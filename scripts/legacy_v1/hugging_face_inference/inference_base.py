import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "koganrath/LiteGPT-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} from Hugging Face...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

print("Base Model loaded. Type 'quit' to exit.")

while True:
    prompt = input("\nPrompt: ")
    if prompt.lower() in ["quit", "exit"]:
        break
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("-" * 20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 20)
