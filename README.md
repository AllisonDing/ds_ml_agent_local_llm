# Data Science ML Agent

A data science and machine learning agent powered by NVIDIA GPUs.  
You can interact with it using natural language to run data exploration and machine learning tasks with minimal setup.

---
## Model Used

[NVIDIA Nemotron Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

## Package Installation

This project requires the following versions to ensure full compatibility:
- RAPIDS: 25.10

Please refer to the [official RAPIDS installation documentation](https://github.com/AllisonDing/cuxfilter_viz_agent#:~:text=RAPIDS%20installation%20documentation) for detailed instructions.

### Installation Example:

```bash
conda create -n rapids-25.10 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.10 python=3.11 'cuda-version=13.0'
```

## Running the Agent

```bash
conda activate rapids-25.10
export NVIDIA_API_KEY=""
```

Then you can run the agent in two different modes:

### 1. **GPU-Accelerated Mode** (NVIDIA cuML + cuDF)
Leverages NVIDIA's RAPIDS libraries for faster data processing and model training.

```bash 
python -m cudf.padnas -m cuml.accel -m streamlit run user_interface.py
```

### 2. **CPU Mode** (scikit-learn + pandas)
Uses standard pandas and scikit-learn for data processing and modeling.

```bash
streamlit run user_interface.py
```

### The agent supports queries such as:<br>
   load data<br>
   preview the head<br>
   describe the data<br>
   train the classification or regression model<br>
   hyperparameter optimization (HPO) for (n) trials<br>
   best model<br>
   show model history<br>
   make inference on the test dataset
   ...

## 📊 Sample Dataset

This project provides sample datasets, the Kaggle [Titanic-Dataset](https://www.kaggle.com/competitions/titanic), [Titanic-Dateset-test](https://www.kaggle.com/competitions/titanic), and **Titanic-Dataset-1M**, an extrapolated version scaled to 1M rows of Titanic-Dataset. They are available in `data/` directory. 

---

**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
