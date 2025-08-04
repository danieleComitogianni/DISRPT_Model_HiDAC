HiDAC: Hierarchical Dual-Adapter Contrastive Model
This repository contains the official implementation for the HiDAC model, a novel parameter-efficient fine-tuning framework for discourse relation classification, as submitted to the DISRPT 2025 shared task.

The model introduces a hierarchical adapter strategy that applies different adaptation methods to lower and upper transformer layers, which are trained by a decoupled dual-loss objective to improve both representation quality and final classification accuracy.

Project Structure
hidac_project/
├── data/                 # <-- Place your CSV data files here
├── pretrained_model/     # <-- Place the provided pre-trained model files here
├── src/                  # <-- All source code modules (model, config, etc.)
├── main.py               # <-- Main execution script for training and evaluation
├── requirements.txt      # <-- Project dependencies
└── README.md            

Data Format
The model expects CSV files in the data/ directory with the following columns:

text1: The first discourse unit. (Already swapped based on dir)

text2: The second discourse unit. (Already swapped based on dir)

label: The string representation of the discourse relation.

framework: The annotation framework (e.g., pdtb, rst).

lang: The two-letter language code (e.g., eng, deu).

The default filenames are train.csv and dev.csv, which can be changed in src/config.py.

Setup
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Place your data:
Ensure your training and validation CSV files are located in the data/ directory.

How to Run
Use the main.py script from the root directory to either train the model or run evaluation.

Training

To start the full training process from scratch, run the following command. The script will use the parameters defined in src/config.py and save the best-performing model checkpoint, adapter weights, and configuration to the ./hidac-final-run/ directory.

python main.py train

Evaluation

You can evaluate either a model you have trained yourself or the pre-trained model provided with this repository.

1. Evaluating Your Trained Model

After a training run is complete, you can evaluate the best saved checkpoint on a new test file.

python main.py evaluate --model_dir ./hidac-final-run --test_file ./data/your_test_file.csv

2. Evaluating the Pre-trained Model

To evaluate with the pre-trained HiDAC model provided:

Create a directory named pretrained_model in the root of the project.

Place the provided hidac_adapters.pth and model_info.json files inside the pretrained_model directory.

In main.py, uncomment the following line:

# args.model_dir = './pretrained_model'

Run the evaluation command, pointing to your test file:

python main.py evaluate --test_file ./data/your_test_file.csv

Model Configuration
All model hyperparameters, including LoRA settings, loss weights, and training parameters, are centralized in the Config class within src/config.py for easy modification and experimentation.