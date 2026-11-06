import os
import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "fineweb")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_config = "sample-10BT"
split = "train"


enc = None

def init_worker():
    """Initialize the tokenizer in each worker process."""
    global enc
    enc = tiktoken.get_encoding("gpt2")

def tokenize_batch(texts):
    """Worker function to tokenize a batch of texts."""
    global enc
    eot = enc.eot_token
    results = []
    for text in texts:
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        results.append(tokens)
    return results

def process():
    print(f"Loading {dataset_name} ({dataset_config}) streaming...")
    ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    
    train_filename = os.path.join(DATA_CACHE_DIR, "train.bin")
    val_filename = os.path.join(DATA_CACHE_DIR, "val.bin")
    
    dtype = np.uint16 
    
    print(f"Writing to {train_filename} and {val_filename}...")
    
    print(f"Writing to {train_filename} and {val_filename}...")
    
    BATCH_SIZE = 1000  # Number of documents per batch
    NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2) # Leave some cores for system/IO
    
    print(f"Tokenizing with {NUM_WORKERS} workers...")
    
    total_tokens = 0
    pbar = tqdm.tqdm(desc="Tokenizing", unit="tok")

    batch = []
    
    with open(train_filename, "wb") as f_train, open(val_filename, "wb") as f_val, \
         ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        
        futures = []
        
        def process_futures():
            nonlocal total_tokens
            pass

        
        HUGE_CHUNK = 10000 
        current_texts = []
        
        for i, example in enumerate(ds):
            current_texts.append(example['text'])
            
            if len(current_texts) >= HUGE_CHUNK:
                sub_batches = [current_texts[j:j+BATCH_SIZE] for j in range(0, len(current_texts), BATCH_SIZE)]
                
                results_list = list(executor.map(tokenize_batch, sub_batches))
                
                for batch_results in results_list:
                    for tokens in batch_results:
                        arr = np.array(tokens, dtype=dtype)
                        
                        if np.random.rand() < 0.01: # 1% val
                            f_val.write(arr.tobytes())
                        else:
                            f_train.write(arr.tobytes())
                        
                        count = len(tokens)
                        total_tokens += count
                        pbar.update(count)
                
                current_texts = []

        if current_texts:
            sub_batches = [current_texts[j:j+BATCH_SIZE] for j in range(0, len(current_texts), BATCH_SIZE)]
            results_list = list(executor.map(tokenize_batch, sub_batches))
            for batch_results in results_list:
                for tokens in batch_results:
                    arr = np.array(tokens, dtype=dtype)
                    if np.random.rand() < 0.01:
                        f_val.write(arr.tobytes())
                    else:
                        f_train.write(arr.tobytes())
                    count = len(tokens)
                    total_tokens += count
                    pbar.update(count)
            
    pbar.close()
    print(f"Done. Total tokens: {total_tokens}")

if __name__ == "__main__":
    process()
