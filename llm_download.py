# import os
# import transformers
# from huggingface_hub import snapshot_download

# # Configuration
# MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

# def fix_nemotron_download():
#     print(f"--- Starting download for {MODEL_ID} ---")
#     print(f"Current Transformers version: {transformers.__version__}")
    
#     # 1. Download artifacts to cache without loading/executing them
#     # This avoids the ImportError crash during the download phase.
#     try:
#         model_path = snapshot_download(repo_id=MODEL_ID)
#         print(f"\n✅ Download complete. Files located at:\n{model_path}")
#     except Exception as e:
#         print(f"\n❌ Download failed: {e}")
#         return

#     # 2. Patch the broken 'modeling_decilm.py' file
#     # We remove 'NEED_SETUP_CACHE_CLASSES_MAPPING' which was removed in newer Transformers
#     target_file = "modeling_decilm.py"
#     file_path = os.path.join(model_path, target_file)
    
#     if os.path.exists(file_path):
#         print(f"\n🔍 Checking {target_file} for compatibility issues...")
        
#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()
        
#         bad_import = "NEED_SETUP_CACHE_CLASSES_MAPPING"
        
#         if bad_import in content:
#             print(f"⚠️  Found broken import: '{bad_import}'. Patching file...")
            
#             # Remove the variable from the import statements
#             new_content = content.replace(f", {bad_import}", "")
#             new_content = new_content.replace(f"{bad_import}, ", "")
#             new_content = new_content.replace(bad_import, "")
            
#             # Write the fixed content back
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(new_content)
                
#             print(f"✅ Successfully patched {target_file}. It is now compatible with Transformers v4.46+.")
#         else:
#             print(f"✅ {target_file} is already patched or clean.")
            
#     else:
#         print(f"⚠️  Could not find {target_file}. The model structure might be different.")

#     print("\n--- Summary ---")
#     print("1. Ensure you have the latest transformers installed: pip install --upgrade transformers")
#     print("2. Load the model in your main script using:")
#     print(f'   model = AutoModelForCausalLM.from_pretrained("{model_path}", trust_remote_code=True)')

# if __name__ == "__main__":
#     fix_nemotron_download()






import os
import transformers
from huggingface_hub import snapshot_download

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Replace this string with your exact model ID
# Example: "nvidia/Llama-3.1-Nemotron-51B-Instruct" or "nvidia/Nemotron-Mini-4B-Instruct"
MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
# (If you have a specific private 9B-v2 ID, paste it above)

def universal_download_fix():
    print(f"--- Starting download for {MODEL_ID} ---")
    
    # 1. Download artifacts to cache
    try:
        model_path = snapshot_download(repo_id=MODEL_ID)
        print(f"\n✅ Download complete. Files located at:\n{model_path}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return

    # 2. Conditional Patching (Only runs if the broken DeciLM file exists)
    target_file = "modeling_decilm.py"
    file_path = os.path.join(model_path, target_file)
    
    if os.path.exists(file_path):
        print(f"\n🔍 Found custom code file {target_file}. Checking for known bugs...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        bad_import = "NEED_SETUP_CACHE_CLASSES_MAPPING"
        
        if bad_import in content:
            print(f"⚠️  Found broken import: '{bad_import}'. Patching file...")
            new_content = content.replace(f", {bad_import}", "").replace(f"{bad_import}, ", "").replace(bad_import, "")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"✅ Successfully patched {target_file}.")
        else:
            print(f"✅ File exists but does not contain the specific bug. No changes made.")
            
    else:
        # This is what will likely happen for newer/standard models
        print(f"\nℹ️  {target_file} not found. This is normal for standard Llama/Mistral architectures.")
        print("   No patching required.")

    print("\n--- Next Steps ---")
    print(f'1. Use this path in your code: "{model_path}"')

if __name__ == "__main__":
    universal_download_fix()