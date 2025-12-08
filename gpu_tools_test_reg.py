#!/usr/bin/env python
"""
Simple batch runner - assumes files are in current directory
Edit the file paths and target column below
"""

import os
import sys
import time
from src.chat_agent import ChatAgent

# === EDIT THESE TO MATCH YOUR FILES ===
TRAIN_FILE = "data/Titanic-Dataset-1M.csv"      # Your training file
# TEST_FILE = "data/Titanic-Dataset-test.csv"        # Your test file  
TARGET = "Fare"           # Your target column
# =====================================

agent = ChatAgent()

# Set up files
agent.uploaded_files = {}
for f in [TRAIN_FILE]:
    if os.path.exists(f):
        agent.uploaded_files[f] = os.path.abspath(f)
        print(f"✅ Found: {f}")
    else:
        print(f"❌ Not found: {f}")

if not agent.uploaded_files:
    print("No files found! Please check file paths.")
    exit(1)


# preview the first 10 rows
# describe the dataset
# perform HPO for Ridge Regressor

# Commands to run
commands = f"""
load dataset {TRAIN_FILE} target={TARGET}
train a regression model
perform HPO for Random Forest Regressor
perform HPO for Support Vector Regressor
show model history
show the best model
save the best model
make inference on {TRAIN_FILE} and save prediction results in .csv file
""".strip().split('\n')

print(f"\nMode: {'GPU' if 'cudf.pandas' in sys.modules else 'CPU'}")
print("="*50)

for cmd in commands:
    cmd = cmd.strip()
    if not cmd:
        continue
    
    print(f"\n▶ {cmd}")
    start_time = time.time() 
    try:
        result = agent.chat(cmd)
        if isinstance(result, tuple):
            print(f"   ✅ {result[0]}")
        else:
            print(f"   ✅ Done")
        
        end_time = time.time()
        print(f"Time: {end_time-start_time}")

    except Exception as e:
        print(f"   ❌ {e}")

print("\n✅ Batch complete!")