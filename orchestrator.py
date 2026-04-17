import pandas as pd 
from handler import DataCustodian
from model_registry import ModelRegistry
from auditor import ModelAuditor
from selector import ModelSelector
from tuner import ModelTuner 
import joblib
import os 


class AuraML :
    def __init__(self,  target_col : str, include_dates : bool=True,  selected_features: list=None):
        self.target_col= target_col
        self.custodian = DataCustodian(target_col, include_dates, selected_features)
        self.registry = ModelRegistry()
        self.auditor =  None
        self.best_model= None
        self.top_families = []

    
    def fit(self, df : pd.DataFrame, tuning_trials:int = 20):
        print("Orchestration begining ...")
        x_train, x_test, y_train, y_test = self.custodian.prepare(df)
        task_type = self.custodian.task_type
        self.auditor = ModelAuditor(task_type)
        selector= ModelSelector(task_type )
        tuner = ModelTuner(task_type, tuning_trials)

        probes= self.registry.get_probes(task_type)
        self.top_families, leaderboard = selector.run_tournament(probes, x_train, y_train)

        self.auditor.log_tournament_results(leaderboard)
        winning_family= self.top_families[0]
        model_class= self.registry.get_search_space(winning_family, task_type)

        self.best_model = tuner.tune(model_class, x_train, y_train)
        print(f"finalising best model : {winning_family}")

        self.best_model.fit(x_train,y_train)

        self.auditor.perform_deep_audit(self.best_model, x_test, y_test, f"Best_{winning_family}")
        print(f"AuraML fit complete :  {winning_family}")


    
    def predict(self, df:pd.DataFrame):
        if not self.best_model:
            raise RuntimeError("model has not been fitted yet")

        processed_data = self.custodian.preprocessor.transform(df)
        return self.best_model.predict(processed_data)
    

    def save_engine(self, filename='auraml_engine.joblib'):
        joblib.dump(self, filename)
        print(f"engine saved successfully : {filename}")

    def load_engine(filename = "auraml_engine.joblib"):
        if not os.path.exists(filename):
            raise FileNotFoundError("file doesn't exist")
        
        return joblib.load(filename)
    