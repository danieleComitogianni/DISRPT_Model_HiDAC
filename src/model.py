# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel

class LoRALayer(nn.Module):
    """
    Implements a Low-Rank Adaptation (LoRA) layer.
    This layer is a parameter-efficient fine-tuning technique that injects
    trainable low-rank matrices into a pre-trained model.
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Initialize weights for better stability
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        """Applies the LoRA transformation."""
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling

class MoELoRALayer(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) layer using LoRA adapters.
    A gating network computes a weighted sum of multiple "expert" LoRA layers.
    """
    def __init__(self, hidden_size, num_experts, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        # Create a list of LoRA experts
        self.experts = nn.ModuleList([
            LoRALayer(hidden_size, hidden_size, rank, alpha, dropout) for _ in range(num_experts)
        ])
        # Gating network to determine expert weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts)
        )

    def forward(self, x, formalism_ids=None):
        """
        Forward pass for the MoE-LoRA layer.
        If formalism_ids are provided, it applies a mask to encourage expert specialization.
        """
        gate_logits = self.gate(x)
        
        # The 'formalism_ids' are integer representations of the original 'framework' strings.
        # This conversion is necessary so they can be used as indices in a tensor-based masking operation
        # to guide the gating network's expert selection.
        if formalism_ids is not None:
            large_negative = -1e9
            mask = torch.full_like(gate_logits, large_negative)
            mask[..., 0] = 0  # Always allow the universal expert
            # Unmask the specific expert for each item in the batch
            specific_expert_indices = (formalism_ids.view(-1, 1, 1) + 1).expand(-1, x.shape[1], -1)
            mask.scatter_(-1, specific_expert_indices, 0)
            gate_weights = F.softmax(gate_logits + mask, dim=-1)
        else:
            gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Compute the weighted sum of expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        mixed_output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=2)
        return mixed_output

class TransformerWithMoLA(nn.Module):
    """
    A wrapper for a pre-trained Transformer that injects hierarchical LoRA adapters.
    - Standard LoRA adapters are applied to the lower layers.
    - MoE-LoRA adapters are applied to the upper layers.
    """
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        # Freeze all original parameters of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        self.num_layers = self.base_model.config.num_hidden_layers

        # Initialize MoE-LoRA adapters for the upper layers
        self.moe_layers = nn.ModuleDict()
        for layer_idx in range(self.config.expert_start_layer, self.num_layers):
            num_experts = 1 + self.config.num_formalisms  # Universal expert + specialist experts
            self.moe_layers[str(layer_idx)] = MoELoRALayer(
                self.hidden_size, num_experts, self.config.lora_rank, self.config.lora_alpha, self.config.lora_dropout
            )

        # Initialize standard LoRA adapters for the lower layers
        self.lora_layers = nn.ModuleDict()
        for layer_idx in range(0, 8):
            self.lora_layers[str(layer_idx)] = LoRALayer(
                self.hidden_size, self.hidden_size, self.config.lora_rank, self.config.lora_alpha, self.config.lora_dropout
            )

        self.current_formalism_ids = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward hooks to each Transformer layer to apply the adapters.
        """
        self.hook_handles = []
        def create_hook(layer_idx):
            def hook(_, __, output):
                # The hook adds the adapter's output to the layer's original output
                hidden_states = output[0] if isinstance(output, tuple) else output.last_hidden_state
                
                # Apply MoE-LoRA if the layer is in the upper set
                if str(layer_idx) in self.moe_layers:
                    moe_output = self.moe_layers[str(layer_idx)](hidden_states, formalism_ids=self.current_formalism_ids)
                    new_hidden_states = hidden_states + moe_output
                # Apply standard LoRA if the layer is in the lower set
                elif str(layer_idx) in self.lora_layers:
                    lora_output = self.lora_layers[str(layer_idx)](hidden_states)
                    new_hidden_states = hidden_states + lora_output
                else:
                    new_hidden_states = hidden_states

                # Update the output object with the modified hidden states
                if isinstance(output, tuple):
                    output = (new_hidden_states,) + output[1:]
                else:
                    output.last_hidden_state = new_hidden_states
                return output
            return hook

        for idx, layer in enumerate(self.base_model.encoder.layer):
            handle = layer.register_forward_hook(create_hook(idx))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Removes all registered forward hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def forward(self, formalism_ids=None, **kwargs):
        """
        Forward pass for the adapted Transformer.
        Temporarily sets formalism_ids for use by the MoE hooks.
        """
        self.current_formalism_ids = formalism_ids
        output = self.base_model(**kwargs)
        self.current_formalism_ids = None
        return output

class HiDACEncoder(nn.Module):
    """
    The main encoder module for HiDAC.
    It wraps the adapted Transformer and adds a projection head for the contrastive loss.
    """
    def __init__(self, base_model_name, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerWithMoLA(AutoModel.from_pretrained(base_model_name), config)
        hidden_size = self.encoder.hidden_size
        # Projection head to map intermediate representations to the contrastive space
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.projection_dim)
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, formalism_ids=None, **kwargs):
        """
        Performs a forward pass for both discourse units and extracts representations
        from both the final and intermediate layers.
        """
        # Process both discourse units through the dual-encoder setup
        outputs_1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1, formalism_ids=formalism_ids, output_hidden_states=True)
        outputs_2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2, formalism_ids=formalism_ids, output_hidden_states=True)

        # Get [CLS] token representation from the final layer for the classifier
        final_pooled_1 = outputs_1.last_hidden_state[:, 0]
        final_pooled_2 = outputs_2.last_hidden_state[:, 0]

        # Get [CLS] token representation from an intermediate layer (layer 8) for the contrastive loss
        scl_pooled_1 = outputs_1.hidden_states[8][:, 0]
        scl_pooled_2 = outputs_2.hidden_states[8][:, 0]

        # Project the intermediate representations into the contrastive space
        proj_1 = self.projection(scl_pooled_1)
        proj_2 = self.projection(scl_pooled_2)

        return final_pooled_1, final_pooled_2, proj_1, proj_2

class HiDAC(nn.Module):
    """
    The complete Hierarchical Dual-Adapter Contrastive (HiDAC) model.
    This class orchestrates the encoder, the classifier, and the dual-loss objective.
    """
    def __init__(self, config, num_labels):
        super().__init__()
        self.config = config
        self.encoder = HiDACEncoder(config.model_name, config)
        hidden_size = self.encoder.encoder.hidden_size
        
        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size), # Input is 4d due to feature concatenation
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Learnable embeddings for each class label, used in the contrastive loss
        self.label_embeddings = nn.Embedding(num_labels, config.projection_dim)

    def _compute_scl_loss(self, features, labels):
        """
        Computes the Label-Centered Supervised Contrastive Loss.
        Pulls instance representations closer to their correct label's embedding and
        pushes them away from all other label embeddings.
        """
        # Get the learnable embedding for the correct label of each instance
        positive_label_embs = self.label_embeddings(labels)
        # Get the embeddings for all possible labels to use as negatives
        all_label_embs = self.label_embeddings.weight
        
        # Calculate similarity between instances and their correct label embedding (positive pairs)
        pos_sim = torch.sum(features * positive_label_embs, dim=-1)
        
        # Calculate similarity between instances and all label embeddings (for the denominator)
        neg_sim_matrix = torch.matmul(features, all_label_embs.T)
        
        # Create a mask to exclude the positive pair from the sum of negative similarities
        neg_mask = torch.ones_like(neg_sim_matrix)
        neg_mask.scatter_(1, labels.unsqueeze(1), 0)
        
        # Calculate the InfoNCE-style loss
        pos_exp = torch.exp(pos_sim / self.config.temperature)
        neg_exp_sum = torch.sum(torch.exp(neg_sim_matrix / self.config.temperature) * neg_mask, dim=-1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum.clamp(min=1e-8))).mean()
        
        return loss

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels=None, formalism_ids=None, scl_active=True, **kwargs):
        """
        Main forward pass for the HiDAC model.
        Calculates logits for classification and computes the combined dual-loss.
        """
        # Get representations from both final and intermediate layers
        final_pooled_1, final_pooled_2, proj_1, proj_2 = self.encoder(
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, formalism_ids=formalism_ids
        )
        
        # Create the enhanced feature vector for the classifier
        diff = final_pooled_1 - final_pooled_2
        prod = final_pooled_1 * final_pooled_2
        combined_features = torch.cat([final_pooled_1, final_pooled_2, diff, prod], dim=1)
        
        # Get classification logits
        logits = self.classifier(combined_features)
        
        # Normalize the projected representations for the SCL loss
        proj_1 = F.normalize(proj_1, dim=1)
        proj_2 = F.normalize(proj_2, dim=1)
        
        loss, ce_loss, scl_loss = None, None, None
        if labels is not None:
            # Calculate the primary classification loss
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # If past the warmup period, calculate and add the contrastive loss
            if self.training and scl_active:
                # Average the two projected views for a stable representation
                avg_features = F.normalize((proj_1 + proj_2) / 2, dim=1)
                scl_loss = self._compute_scl_loss(avg_features, labels)
                
                # Combine the two losses using the configured weights
                loss = (self.config.ce_loss_weight * ce_loss) + (self.config.scl_loss_weight * scl_loss)
            else:
                # During warmup or evaluation, only use the CE loss
                loss = ce_loss
                scl_loss = torch.tensor(0.0, device=logits.device)
        
        return {'loss': loss, 'logits': logits, 'ce_loss': ce_loss, 'scl_loss': scl_loss}