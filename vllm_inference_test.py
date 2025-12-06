 # Client: Standard chat and streaming
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="null")

# Simple chat completion
resp = client.chat.completions.create(
    model="/home/allisond/.cache/huggingface/hub/models--nvidia--Llama-3_3-Nemotron-Super-49B-v1_5/snapshots/420ba7d28211abf116b8b103ab700d92619daf98",
    messages=[
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "Give me 3 interesting facts about vLLM."}
    ],
    temperature=0.6,
    max_tokens=256,
)
print(resp.choices[0].message.content)
