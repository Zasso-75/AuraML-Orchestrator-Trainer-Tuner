import pandas as pd
from orchestrator import AuraML

# 1. Load a sample dataset (Replace with your actual CSV)
# For testing, you can use a small public dataset like Iris or Titanic
df = pd.read_csv("your_data.csv") 

# 2. Initialize AuraML with your target column
engine = AuraML(target_col="target_column_name")

# 3. Run the full pipeline (Tournament + Tuning)
# Keep tuning_trials low (5-10) for your first test run
engine.fit(df, tuning_trials=10)

# 4. Save the engine (This creates the file app.py and ui.py need)
engine.save_engine("auraml_engine.joblib")

print("✨ Training complete. You can now run uvicorn app:app or streamlit run ui.py")