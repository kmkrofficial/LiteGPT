"""
Interactive script to chat with the INSTRUCT-tuned model.
Run:
    python nanoGPT/check_instruct.py
"""
import os
import torch
import tiktoken
from model import GPT, GPTConfig

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, 'out-instruct') 
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading INSTRUCT model from {checkpoint_path} on {device}...")
    
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found! (Wait for training to finish 1st epoch)")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    iter_num = checkpoint.get('iter_num', 'N/A')
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    print(f"Model loaded successfully! (Iter: {iter_num}, Best Loss: {best_val_loss})")

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    print("\n" + "="*40)
    print("      INSTRUCT CHAT MODE      ")
    print("      (Prompts are auto-formatted into Alpaca style)")
    print("Commands:")
    print("  /temp X.X   : Set temperature (default 0.8)")
    print("  /top_k N    : Set top_k (default 200)")
    print("  /samples N  : Set num_samples (default 1)")
    print("  /tokens N   : Set max_new_tokens (default 200)")
    print("  quit        : Exit")
    print("="*40 + "\n")

    temp = 0.1
    top_k = 5
    num_samples = 1
    max_tokens = 200

    while True:
        prompt = input(f"[{temp=}, {top_k=}] Instruction: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if prompt.startswith('/temp'):
            try:
                temp = float(prompt.split()[1])
                print(f"Temperature set to {temp}")
            except:
                print("Invalid syntax. Use: /temp 0.8")
            continue
            
        if prompt.startswith('/top_k'):
            try:
                top_k = int(prompt.split()[1])
                print(f"Top_k set to {top_k}")
            except:
                print("Invalid syntax. Use: /top_k 200")
            continue

        if prompt.startswith('/samples'):
            try:
                num_samples = int(prompt.split()[1])
                print(f"Num samples set to {num_samples}")
            except:
                print("Invalid syntax. Use: /samples 3")
            continue

        if prompt.startswith('/tokens'):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Invalid syntax. Use: /tokens 50")
            continue
        
        if not prompt.strip():
            continue

        formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"

        start_ids = encode(formatted_prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        
        stop_token = encode("<|endoftext|>")[0]

        print(f"\nGenerating...", end=" ", flush=True)
        with torch.no_grad():
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens=max_tokens, temperature=temp, top_k=top_k, stop_idx=stop_token)
                
                # Decode output
                output_text = decode(y[0].tolist())
                
                try:
                    response_only = output_text.split("### Response:\n")[1]
                    # Remove the stop token string if it exists in the decoded text
                    response_only = response_only.replace("<|endoftext|>", "")
                except:
                    response_only = output_text # Fallback

                # Print result
                print("-" * 20 + f" Sample {k+1} " + "-" * 20)
                print(response_only)
                print("-" * 48 + "\n")

if __name__ == "__main__":
    main()
