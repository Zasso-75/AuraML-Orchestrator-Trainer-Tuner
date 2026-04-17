import pandas as pd 
from orchestrator import AuraML

csv_path= "housing.csv"
target_col = "median_house_value"
df=pd.read_csv(csv_path)


engine = AuraML(target_col=target_col)
engine.fit(df, tuning_trials = 50 )
engine.save_engine('auraml_engine.joblib')
print("Training complete. You can now run uvicorn app:app or streamlit run ui.py")
