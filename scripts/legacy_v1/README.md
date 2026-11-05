# LiteGPT V1 (preserved baseline)

GPT-2 Small (124M), trained on FineWeb-Edu and fine-tuned on Alpaca. Published as `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct`. It is kept only to reproduce V1 baselines, and `litegpt/` doesn't import it. Tested on Windows with Python 3.11.9; the package list is in `test_env_requirements.txt`.

Run everything from inside this folder, in its own environment:

```bash
pip install -r requirements.txt
```

The datasets aren't included: download and tokenize them with `data/<dataset>/prepare.py`.

```bash
python train.py config/train_fineweb.py
```

```bash
python train.py config/finetune_alpaca.py
```

Chat with the results: `python check_model.py` for the base model, `python check_instruct.py` for the instruct model.

Checkpoints are written to `out-fineweb/` and `out-instruct/` here (git-ignored).

## From Hugging Face

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("koganrath/LiteGPT-Instruct")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

Helpers are in `hugging_face_inference/`.
