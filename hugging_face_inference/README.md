# Hugging Face Inference

This directory contains standalone scripts to run inference using the [LiteGPT-Instruct](https://huggingface.co/koganrath/LiteGPT-Instruct) model hosted on Hugging Face.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

We provide separate scripts for the Instruct and Base models:

### Instruct Model (Alpaca)
Chat with the instruction-tuned model:
```bash
python inference_instruct.py
```

### Base Model (FineWeb)
Complete text with the base pretrained model:
```bash
python inference_base.py
```

## Code Example

You can also use the model programmatically:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("koganrath/LiteGPT-Instruct")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```
