import numpy as np
import optuna
from sklearn.model_selection import cross_val_score


class ModelTuner:
    def __init__(self, task_type: str, n_trials: int=20):
        self.task_type= task_type
        self.n_trials = n_trials
        self.scoring= 'accuracy' if self.task_type=='classification' else 'neg_mean_squared_error'


    def _get_params(self, trial, model_class_name):
        params={}

        if 'XGB' in model_class_name or "LGBM" in model_class_name:
            params= {
                'n_estimators': trial.suggest_int('n_estimators',50,500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth' : trial.suggest_int( 'max_depth', 3,10),
                'subsample' : trial.suggest_float('subsample', 0.5, 1.0) 
            }
        
        elif 'RandomForest' in model_class_name:
            params={
                'n_estimators': trial.suggest_int('n_estimators', 50,300),
                'max_depth': trial.suggest_int('max_depth', 5,20),
                'min_samples_split':trial.suggest_int('min_samples_split', 5,20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
        
        elif "SVC" in model_class_name or "SVR" in model_class_name:
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
                'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf'])
            }
        
        elif "LogisticRegression" in model_class_name:
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True)
            }
        elif "Ridge" in model_class_name:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True)
            }
        elif "ElasticNet" in model_class_name:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
            }


        return params



    def tune(self,model_class,X,y):
        model_name = model_class.__name__
        print(f"starting optimisation loop for model {model_name} ({self.n_trials} trials .. )")

        def objective(trial):
            params = self._get_params(trial,model_name)
            model= model_class(**params)

            scores= cross_val_score(model,X, y, n_jobs=-1, cv=5,scoring=self.scoring)
            return np.mean(scores)
        
    
        direction= 'maximize' 
        study = optuna.create_study(direction= direction)
        study.optimize(objective, n_trials=self.n_trials)

        print(f"best parameters for {model_name} : {study.best_params}")

        return model_class(**study.best_params)
    