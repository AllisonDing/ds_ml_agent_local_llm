# # llm.py - Simplified for beginners
# import os
# from typing import List, Optional
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # Configuration
# BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
# MODEL = os.getenv("NIM_MODEL", "nvidia/nvidia-nemotron-nano-9b-v2")
# TEMPERATURE = float(os.getenv("NIM_TEMPERATURE", "0"))
# MAX_TOKENS = int(os.getenv("NIM_MAX_TOKENS", "4096"))

# # Set up HTTP session with retries
# def create_session():
#     """Create a session with retry logic for better reliability."""
#     session = requests.Session()
    
#     # Retry Strategy for Resilient Communication
#     retry_strategy = Retry(
#         total=3,  # Try 3 times
#         backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
#         status_forcelist=[429, 500, 502, 503, 504],  # Retry on these errors
#     )
    
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("https://", adapter)
#     session.mount("http://", adapter)
    
#     return session

# class LLMClient:
#     """Simple client to talk to AI models."""
    
#     def __init__(self):
#         # Get API key from environment
#         self.api_key = os.getenv("NVIDIA_API_KEY")
#         if not self.api_key:
#             raise ValueError("Please set NVIDIA_API_KEY environment variable")
        
#         self.base_url = BASE_URL.rstrip("/")
#         self.model = MODEL
#         self.temperature = TEMPERATURE
#         self.max_tokens = MAX_TOKENS
#         self.session = create_session()
 
#     def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
#         """Send messages to the AI and get a response."""
        
#         # Prepare the request
#         request_data = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": self.temperature,
#             "max_tokens": self.max_tokens,
#         }
        
#         # Add tools if provided
#         if tools:
#             request_data["tools"] = tools
#             request_data["tool_choice"] = "auto"
        
#         # Set up headers
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
        
#         try:
#             # Make the request
#             response = self.session.post(
#                 f"{self.base_url}/chat/completions",
#                 headers=headers,
#                 json=request_data,
#                 timeout=(10, 300)  # 10 seconds to connect, 300 seconds to read
#             )
            
#             # Check if request was successful
#             response.raise_for_status()
#             return response.json()
            
#         except requests.exceptions.RequestException as e:
#             raise RuntimeError(f"Failed to communicate with AI: {e}")

# def create_client() -> LLMClient:
#     """Create a new LLM client."""
#     return LLMClient()







import os
import torch
import json
import ast  # <--- The Regex Killer. Parses code natively.
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
# LOCAL_MODEL_PATH = "/home/allisond/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/7d4e437f6cd878d51f70d715e99d9e1bcc93a462"
LOCAL_MODEL_PATH = "/home/allisond/.cache/huggingface/hub/models--nvidia--Llama-3_3-Nemotron-Super-49B-v1_5/snapshots/420ba7d28211abf116b8b103ab700d92619daf98"

class LLMClient:
    """
    Robust Local Client using AST parsing.
    No Regex. Treats model output as executable Python code structure.
    """
    
    def __init__(self):
        print(f"Initializing Local LLM from: {LOCAL_MODEL_PATH}")
        print("Loading weights...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully!")
 
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """
        Generates response and parses it using Python's Abstract Syntax Tree (AST).
        """
        
        # -------------------------------------------------------
        # STEP 1: PROMPT (NIM Style)
        # -------------------------------------------------------
        conversation = list(messages)
        
        if tools:
            # Build tool signatures for the prompt
            sigs = []
            valid_tool_names = set() # We use this to validate AST results later
            
            for t in tools:
                name = t['function']['name']
                valid_tool_names.add(name)
                # signature for prompt
                params = t['function']['parameters']['properties']
                param_str = ", ".join(params.keys())
                sigs.append(f"- {name}({param_str})")
            
            sig_block = "\n".join(sigs)

            system_prompt = (
                "/think\n"
                "You are an expert AI assistant. You have these Python tools:\n"
                f"{sig_block}\n\n"
                "RULES:\n"
                "1. To use a tool, output ONLY the valid Python code.\n"
                "2. Do NOT add markdown, explanations, or quotes.\n"
                "3. Example output: load_dataset('data.csv')\n"
            )
            conversation.insert(0, {"role": "system", "content": system_prompt})

        # -------------------------------------------------------
        # STEP 2: GENERATE
        # -------------------------------------------------------
        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,       # Low temp for code precision
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # -------------------------------------------------------
        # STEP 3: CLEAN TEXT
        # -------------------------------------------------------
        raw_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        
        # Strip thinking block
        if "</think>" in raw_text:
            clean_text = raw_text.split("</think>")[-1].strip()
        else:
            clean_text = raw_text.strip()
            
        # Strip markdown syntax to leave raw code
        clean_text = clean_text.replace("```python", "").replace("```", "").strip()

        print(f"[DEBUG] Raw Code Candidate: {clean_text}")

        # -------------------------------------------------------
        # STEP 4: AST PARSING (No Regex!)
        # -------------------------------------------------------
        detected_tool = None
        detected_args = {}

        if tools and clean_text:
            try:
                # 1. Parse the string into a Python AST structure
                tree = ast.parse(clean_text)
                
                # 2. Walk through the tree looking for function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Get function name
                        # Handles straightforward calls like func()
                        if hasattr(node.func, 'id'):
                            func_name = node.func.id
                            
                            if func_name in valid_tool_names:
                                detected_tool = func_name
                                print(f"[DEBUG] AST detected valid tool: {func_name}")
                                
                                # 3. Extract Arguments Safely
                                # a. Positional args: func('arg1')
                                if node.args:
                                    # We map the first positional arg to the first tool param
                                    # (Simple heuristic for single-arg tools)
                                    first_arg_val = ast.literal_eval(node.args[0])
                                    
                                    # Find name of first param from tool definition
                                    target_tool = next(t for t in tools if t['function']['name'] == func_name)
                                    props = target_tool['function']['parameters']['properties']
                                    if props:
                                        first_key = list(props.keys())[0]
                                        detected_args[first_key] = first_arg_val
                                
                                # b. Keyword args: func(trials=20)
                                for keyword in node.keywords:
                                    arg_name = keyword.arg
                                    arg_val = ast.literal_eval(keyword.value)
                                    detected_args[arg_name] = arg_val
                                
                                break # Found our tool, stop looking
                                
            except SyntaxError:
                print("[DEBUG] Output was not valid Python code. Treating as text.")
            except Exception as e:
                print(f"[DEBUG] AST parsing error: {e}")

        # -------------------------------------------------------
        # STEP 5: RETURN
        # -------------------------------------------------------
        if detected_tool:
            return {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_local_ast",
                            "type": "function",
                            "function": {
                                "name": detected_tool,
                                "arguments": json.dumps(detected_args)
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": clean_text
                },
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": len(outputs[0])}
        }

def create_client() -> LLMClient:
    return LLMClient()