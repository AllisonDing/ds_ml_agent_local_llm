import os
from openai import OpenAI
from typing import List, Optional

# vLLM server endpoint
VLLM_API_BASE = "http://localhost:8000/v1"
# MODEL_NAME = "/home/allisond/.cache/huggingface/hub/models--nvidia--Llama-3_3-Nemotron-Super-49B-v1_5/snapshots/420ba7d28211abf116b8b103ab700d92619daf98"  # Super-49B
MODEL_NAME = "/home/allisond/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/bce37e25324449f9be5b6a03c69a15244d27ee6e" # Nano-9B


class LLMClient:
    """Client that connects to vLLM server"""
    
    def __init__(self):
        print(f"Connecting to vLLM server at: {VLLM_API_BASE}")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=VLLM_API_BASE
        )
        print("vLLM client initialized!")
    
    def chat(self, messages: List[dict], **kwargs) -> dict:
        """Generate response from vLLM"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.3),
                top_p=kwargs.get('top_p', 0.95),
            )
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    },
                    "finish_reason": response.choices[0].finish_reason
                }],
                "usage": response.usage.model_dump() if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            print(f"[ERROR] vLLM API call failed: {e}")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    },
                    "finish_reason": "error"
                }]
            }


def create_client() -> LLMClient:
    return LLMClient()