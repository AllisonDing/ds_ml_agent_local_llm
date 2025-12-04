# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

print(f"Loading {model_id}...")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True
)

# 2. Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

print("Model loaded successfully!")