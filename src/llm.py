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
        self.temperature = 1.0
        self.top_p = 0.95
        self.max_tokens = 2048
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

    def _clean_thinking(self, content: str) -> str:
        """Strip <think>...</think> blocks, preserving the summary after them."""
        if not content:
            return ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return content.strip()

    def _parse_xml_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse a <tool_call> XML block from model response.

        Returns a dict with 'name' and 'arguments' (dict), or None if not found.
        The Nemotron chat template uses this format:
          <tool_call>
          <function=func_name>
          <parameter=param1>value1</parameter>
          </function>
          </tool_call>
        """
        if not content or '<tool_call>' not in content:
            return None

        block_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
        if not block_match:
            return None

        block = block_match.group(1)

        func_match = re.search(r'<function=(\w+)>', block)
        if not func_match:
            return None
        func_name = func_match.group(1)

        params: Dict[str, Any] = {}
        for m in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', block, re.DOTALL):
            params[m.group(1)] = m.group(2).strip()

        return {"name": func_name, "arguments": params}

    def chat(self, messages: List[dict]) -> dict:
        """Send messages to the model. Tools are embedded in the system prompt."""
        if not self.model:
            raise Exception("No model connected.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]
                content = message.get("content") or ""

                # Strip <think> blocks; keep the post-think summary in content
                content = self._clean_thinking(content)
                message["content"] = content

                # Parse XML tool call and expose it in OpenAI-compatible format
                # so chat_agent.py can use the same tool_calls check it already has
                parsed = self._parse_xml_tool_call(content)
                if parsed:
                    message["tool_calls"] = [{
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"])
                        }
                    }]

            return result

        except Exception as e:
            raise Exception(f"LLM Error: {str(e)}")

def create_client():
    return LLMClient()
