import pandas as pd
import numpy as np
from joblib import Parallel, delayed 
from sklearn.model_selection import cross_val_score

class ModelSelector:
    def __init__ (self, task_type:str, n_jobs : int=-1):
        self.task_type= task_type
        self.n_jobs = n_jobs    
        if self.task_type == 'regression':
            self.scoring= 'neg_mean_squared_error'
        else:
            self.scoring= 'accuracy'

    def _evaluate_probe(self, family:str, model_instance, X, y):
        try:
            cv_scores= cross_val_score(model_instance, X, y, scoring= self.scoring)
            mean_score= cv_scores.mean()

            # if self.task_type=='regression':
            #     mean_score = -mean_score
            
            return {
                'family': family,
                'score' : mean_score,
                'model_name': model_instance.__class__.__name__,
                'params': str(model_instance.get_params())
            }
        
        except Exception as e:
            return {'family': family, 'score': -np.inf, 'error': str(e)}
        
    

    def run_tournament(self, probes: dict, X, y):
        tasks = []
        for family, model_list in probes.items():
            for model in model_list:
                tasks.append((family, model))
        
        results = Parallel(n_jobs= self.n_jobs)(
            delayed(self._evaluate_probe)(fam, mod, X, y) for  fam,mod in tasks
        )

        result_df = pd.DataFrame([r for r in results if r['score']!=-np.inf])
        if result_df.empty:
            print('All probes failed, pls check data compatibility')

        leaderboard= result_df.sort_values(by='score', ascending = False)
        top_families= leaderboard.drop_duplicates(subset='family').head(2)

        print("Tournament leaderboard" )
        print(leaderboard[['family', 'model_name', 'score']].head(5))
        return top_families['family'].to_list(), leaderboard
    

