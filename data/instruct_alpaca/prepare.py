"""
Prepare the Alpaca dataset for instruction tuning.
Downloads the JSON file, formats it into a single stream of text, tokenizes with GPT-2 tokenizer.
"""
import os
import requests
import json
import numpy as np
import tiktoken

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

def prepare():
    data_dir = os.path.join(os.path.dirname(__file__))
    file_path = os.path.join(data_dir, 'alpaca_data.json')
    data_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    data_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    if not os.path.exists(file_path):
        download_file(data_url, file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instructions.")

    print(f"Loaded {len(data)} instructions.")

    enc = tiktoken.get_encoding("gpt2")
    
    def format_example(example):
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']
        
        if input_text:
            text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}<|endoftext|>"
        else:
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output_text}<|endoftext|>"
        return text

    all_ids = []
    print("Tokenizing...")
    for ex in data:
        text = format_example(ex)
        ids = enc.encode(text, allowed_special={'<|endoftext|>'})
        all_ids.extend(ids)
    
    print(f"Total tokens: {len(all_ids)}")

    print(f"Total tokens: {len(all_ids)}")

    n = len(all_ids)
    train_ids = all_ids[:int(n*0.95)]
    val_ids = all_ids[int(n*0.95):]

    val_ids = all_ids[int(n*0.95):]

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    print(f"Saved train.bin ({len(train_ids)} tokens) and val.bin ({len(val_ids)} tokens)")

    print(f"Saved train.bin ({len(train_ids)} tokens) and val.bin ({len(val_ids)} tokens)")

    meta = {
        'vocab_size': 50257, # GPT-2
    }
    # pickle it
    import pickle
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    prepare()
