# Data Science ML Agent

This example demonstrates how to build a <b>natural language-driven data science and machine learning agent</b> powered by <b>NVIDIA GPUs</b>.
The agent allows users to perform data exploration, model training, and hyperparameter optimization interactively using <b>RAPIDS cuDF</b> and <b>cuML</b> for GPU acceleration. 

## Overview

The Data Science ML Agent enables natural language interaction for common data science workflows, including dataset loading, target selection, model training, and performance optimization. It combines the flexibility of LLMs with GPU-accelerated computation to simplify and speed up end-to-end machine learning pipelines.

## LLM Used

[NVIDIA Nemotron Cascade-2-30B-A3B](https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B): A compact, open-source large language model optimized for reasoning and data analysis tasks.

## Inference Server Used

vLLM (Versatile Large Language Model) is a high-performance, open-source library designed for the fast inference and serving of Large Language Models (LLMs).

## Key Features

- Natural language interface for running data exploration and ML workflows
- GPU acceleration with cuDF (DataFrame operations) and cuML (ML algorithms)
- Fast inference using the locally downloaded LLM model
- Support for CPU mode using pandas and scikit-learn
- Simple setup and execution through Streamlit interface
- Compatible with both small and large-scale datasets

## Requirements

- RAPIDS 25.10
- Python 3.10, 3.11, 3.12, or 3.13
- vLLM 0.9.x and later
- CUDA 12.0 or 13.0 compatible NVIDIA GPU (for GPU mode)
- streamlit, optuna, joblib, transformers, accelerate

Please refer to the [official RAPIDS installation documentation](https://docs.rapids.ai/install/) for detailed instructions.

## Recommendated Installation:

 For simplicity, it is recommended to use only one environment for both the Inference Server (vLLM) and the Agent (RAPIDS). The example below uses the following compatible versions (important for a successful setup): RAPIDS 25.02, Python 3.11, CUDA 13.0, and vLLM 0.17.1.

```bash
conda create -n rapids-vllm -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.10 python=3.11 'cuda-version=13.0'

conda activate rapids-vllm

pip install vllm==0.17.1 --extra-index-url https://download.pytorch.org/whl/cu130

pip install streamlit optuna joblib transformers accelerate
```

## Example Usage

### Step 1: Start the vLLM Local Server (Terminal 1)
Open a terminal, activate the server environment, and launch the model.

```bash
conda activate rapids-vllm

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.1a"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MODEL_CKPT=nvidia/Nemotron-Cascade-2-30B-A3B

vllm serve nvidia/Nemotron-Cascade-2-30B-A3B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.60 \
    --max-model-len 16384 \
    --kv-cache-dtype fp8 \
    --reasoning-parser nemotron_v3 \
    --mamba-ssm-cache-dtype float32 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

```

### Step 2: Start the User Interface (Terminal 2)
Open a second terminal.

```bash
conda activate rapids-vllm

# cuDF memory allocation
export RMM_ALLOCATOR=managed
export CUDF_PANDAS_LIMIT_MEMORY=25000000000

# GPU-accelerated mode
python -m cudf.pandas -m cuml.accel -m streamlit run user_interface.py

# CPU-based mode
streamlit run user_interface.py
```

## Example queries:
```bash
load dataset Titanic-Dataset.csv/Titanic-Dataset-test.csv
describe the dataset
preview the first n rows
set target variable to be 'Survived'
train classification/regression model
optimize svc with 50 trials
optimize forest regressor with 30 trials
show best model by r2
make inference for the test dataset
...
```

## 📊 Sample Dataset

This project provides sample datasets, you can download the Kaggle Titanic Datasets [train.csv](https://www.kaggle.com/competitions/titanic) and [test.csv](https://www.kaggle.com/competitions/titanic) into the data folder. You can also create **train-1M.csv**, an extrapolated version scaled to 1M rows of the train.csv using the script extrapolation.py. 


**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
