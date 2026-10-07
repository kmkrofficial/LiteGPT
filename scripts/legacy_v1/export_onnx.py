"""
Export trained nanoGPT model to ONNX.
"""
import os
import torch
from model import GPT, GPTConfig

def export_to_onnx():
    # 1. Setup
    out_dir = 'out-fineweb'
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    onnx_path = os.path.join(out_dir, 'gpt2_124m.onnx')
    
    device = 'cpu' # Export is often easier on CPU to avoid device mismatch, but GPU is fine too.
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Needs to exist
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found! Wait for training to save at least one checkpoint.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Fix keys if needed (from train.py)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # 2. Dummy Input
    # Batch size 1, sequence length 1 (will be dynamic)
    dummy_input = torch.randint(0, gptconf.vocab_size, (1, 1), dtype=torch.long, device=device)
    
    # 3. Export
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'loss'], # Model returns (logits, loss) tuple
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Success! Model exported to ONNX.")

if __name__ == "__main__":
    export_to_onnx()
