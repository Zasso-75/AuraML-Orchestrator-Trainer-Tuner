# 🚀 AuraML: Automated Intelligence Engine

AuraML is a high-performance, modular machine learning framework designed to bridge the gap between raw data and production-ready serving. It automates the "Cold Start" problem of model selection through parallelized tournaments and refines performance using Bayesian hyperparameter optimization.



## 🧠 High-Level Problems Solved

* **The Model Selection Paradox:** Instead of guessing which algorithm fits your data, AuraML runs a **Parallel Tournament**, testing multiple model families (Linear, Trees, Boosting) simultaneously across all CPU cores.
* **Manual Tuning Bottleneck:** It replaces "GridSearch" with **Bayesian Optimization (Optuna)**, which intelligently navigates the hyperparameter space to find the "global minimum" error faster than traditional methods.
* **Infrastructure Fragmentation:** AuraML bundles the preprocessing logic, the trained model, and the deployment endpoints into a single, serializable `.joblib` "Brain," ensuring consistency between training and production.

---

## ✨ Key Features

* **Data Custodian:** Automated task detection (Regression vs. Classification), smart imputation for missing values, and high-cardinality encoding.
* **Parallel Orchestration:** Multi-threaded model evaluation to significantly reduce the time spent on the initial discovery phase.
* **Deep Auditing:** Every run generates a `run_audit/` directory containing:
    * **Tournament Leaderboards:** Visual comparisons of model performance.
    * **Fit Diagnostics:** Residual plots and confusion matrices for the final champion model.
* **Dual-Layer Serving:** Includes both a **FastAPI** backend for machine-to-machine integration and a **Streamlit** UI for human interaction.

---

## 🛠️ Project Structure

```text
AuraML-Engine/
├── core/                # The "Engine Room"
│   ├── handler.py       # Data cleaning & Scaling (DataCustodian)
│   ├── selector.py      # Parallel tournament logic
│   ├── tuner.py         # Bayesian Optimization (Optuna)
│   ├── auditor.py       # Diagnostic charts & Logging
│   └── orchestrator.py  # The "Brain" that connects all modules
├── app.py               # Modern FastAPI Lifespan endpoints
├── ui.py                # Streamlit Dashboard UI
├── train_engine.py      # Entry point to train and save the model
├── requirements.txt     # Dependency list
└── auraml_engine.joblib # The serialized production model
```

---

## 🚀 How to Use

### 1. Training & Saving
To train the engine on your dataset and generate the "Brain":
```python
from core.orchestrator import AuraML
import pandas as pd

df = pd.read_csv("your_data.csv")
engine = AuraML(target_col="price")
engine.fit(df, tuning_trials=50)
engine.save_engine("auraml_engine.joblib")
```

### 2. Running Inference (CLI)
For quick batch processing of a CSV:
```bash
python inference.py new_unseen_data.csv
```

### 3. Launching the API
To serve the model as a REST API:
```bash
uvicorn app:app --reload
```
*Access the interactive docs at: `http://127.0.0.1:8000/docs`*

### 4. Visualizing with the Dashboard
To interact with the model via a browser:
```bash
streamlit run ui.py
```

---

## 📊 Understanding the Output

* **In the UI:** After uploading a CSV and clicking "Predict," the engine appends a column named `AuraML_Prediction`. You can download this enhanced CSV directly.
* **In the Audit Folder:** Look for `tournament_results.csv` to see how different algorithms compared. The `best_model_audit.png` will show you if the model is overfitting or underperforming on specific data segments.

---

## ⚖️ Strengths & Weaknesses

### Strengths
* **Modular Architecture:** You can swap out the `DataCustodian` for a different cleaning strategy without breaking the `Tuner`.
* **Consistency:** Because the preprocessor state is saved *inside* the engine object, you never have to worry about "Data Scaling Leakage" during inference.
* **Speed:** Parallelization makes the exploration of 10+ models happen in the time it usually takes to train one.

### Weaknesses (Known Limitations)
* **Feature Engineering:** Currently, the engine relies on the provided features. It does not yet automatically generate derived features (e.g., polynomial features or ratio features).
* **Small Datasets:** Bayesian optimization and deep trees can lead to overfitting on datasets with fewer than 500 rows.
* **Memory Usage:** Loading many large "Boosting" models in parallel can be memory-intensive for low-RAM machines.

---

## 🛤️ Roadmap
- [ ] Add Automated Feature Engineering (AFE).
- [ ] Implement SHAP values for model explainability in the UI.
- [ ] Add Support for Time-Series forecasting modules.

***

### 🛠️ Installation
```bash
pip install -r requirements.txt
```
*Note: Requires Python 3.9+ for modern FastAPI Lifespan support.*
