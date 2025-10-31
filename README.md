# Conversational ML Agent

A conversational data science and machine learning agent powered by NVIDIA GPUs.  
You can interact with it using natural language to run data exploration and machine learning tasks with minimal setup.

---

## Running the Agent

You can run the agent in two different modes:

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
   hyperparameter optimization (HPO)<br>
   best model<br>
   show model history<br>
   make inference on the test dataset
   ...

## 📊 Sample Dataset

This project provides sample datasets, the Kaggle [Titanic-Dataset, Titanic-Dateset-test,](https://www.kaggle.com/competitions/titanic) and **Titanic-Dataset-1M**, an extrapolated version scaled to 1M rows of Titanic-Dataset. They are available in `data/` directory. 

---

**Note:**  
- Ensure you have the appropriate dependencies installed for each mode.  
- GPU mode requires a supported NVIDIA GPU and the RAPIDS ecosystem installed.
