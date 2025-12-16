# Small Language Model (SLM) Project

This repository contains the code and resources for training a Small Language Model (SLM) from scratch, based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). 

The goal of this project is to build a GPT-2 (124M parameter) equivalent model trained on high-quality educational data and subsequently fine-tuned for instruction following.

## Project Overview

- **Architecture**: GPT-2 Small (12 layers, 12 heads, 768 embedding dim)
- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (Sample 10B tokens)
- **Fine-Tuning**: Alpaca Dataset (Instruction Tuning)
- **Framework**: PyTorch (nanoGPT)

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**
   The training data is processed using the scripts in the `data/` directory.
   
   > **Note**: The actual datasets (FineWeb-Edu and Alpaca) are not included in this repository to keep it light. You can download and prepare them using the provided `prepare.py` scripts in the respective `data/` subdirectories.

## Usage

### 1. Training
To train the base model:
```bash
python train.py config/train_fineweb.py
```

### 2. Instruction Fine-Tuning
To fine-tune the model on instructions:
```bash
python train.py config/finetune_alpaca.py
```

### 3. Inference / Chat
We provide interactive scripts to test the models:

- **Check Base Model**:
  ```bash
  python check_model.py
  ```
  *Interactively prompts the base model (trained on FineWeb-Edu).*

- **Check Instruction Model**:
  ```bash
  python check_instruct.py
  ```
  *Interactively chats with the instruction-tuned model. Supports commands like `/temp`, `/tokens`.*

## Directory Structure

- `data/`: Data preparation scripts and raw data.
- `config/`: Configuration files for training and fine-tuning.
- `out-fineweb/`: Checkpoints and logs for the base model.
- `out-instruct/`: Checkpoints for the instruction-tuned model.
- `check_model.py`: Inference script for the base model.
- `check_instruct.py`: Inference script for the instruct model.

## Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) Dataset
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) Dataset

## Testing Environment

The code has been tested on **Windows** with **Python 3.11.9**.

For a complete list of installed packages in the testing environment (including specific versions), see `test_env_requirements.txt`.
