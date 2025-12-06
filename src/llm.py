# src/llm.py
import os
import re
import requests
import json
from typing import List, Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

def create_session():
    """Create a session with robust retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

class LLMClient:
    def __init__(self):
        self.base_url = VLLM_BASE_URL.rstrip("/")
        self.session = create_session()
        self.temperature = 0.0
        self.max_tokens = 4096
        self.model = self._auto_detect_model()

    def _auto_detect_model(self) -> str:
        try:
            print(f"🔌 Connecting to vLLM at {self.base_url}...")
            response = self.session.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_id = data["data"][0]["id"]
                    print(f"✅ Found active model: {model_id}")
                    return model_id
            return "nemotron-9b"
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            return "unknown-model"

    def _clean_content(self, content: str) -> str:
        """Removes <think> tags and other artifacts from the model response."""
        if not content:
            return ""
        
        # Remove the internal monologue block <think>...</think>
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Cleanup extra whitespace resulting from the removal
        return content.strip()

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        if not self.model:
            raise Exception("No model connected.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 400 and "tool" in response.text.lower():
                 raise Exception(f"Tool Error: Ensure vLLM started with '--tool-call-parser llama3_json'. Response: {response.text}")

            response.raise_for_status()
            result = response.json()

            # --- CLEANING STEP (CRITICAL) ---
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                
                # 1. Clean the text content (remove <think> tags)
                if message.get("content"):
                    message["content"] = self._clean_content(message["content"])
                
                # 2. Ensure tool calls are respected
                # (Native parser handles extraction, but we leave it as is)
            
            return result

        except Exception as e:
            raise Exception(f"LLM Error: {str(e)}")

def create_client():
    return LLMClient()