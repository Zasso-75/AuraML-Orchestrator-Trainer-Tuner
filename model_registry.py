from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge 
from sklearn.svm import SVC, SVR 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

'''
Model Registry -  
Random Forest (Regressor, classifier ), 
Gradient Boosting(Regressor, Classifier), 
LogisticRegressor, 
ElasticNet, 
Ridge, 
SVC, 
SVR, 
XGB (classifier, regressor),  
Light gbm (Regressor, Classifier)
'''

class ModelRegistry:
    @staticmethod
    def get_probes(task_type : str):
        """
            Returns a dictionary of 'Probes'. 
            Each key is a Family, and each value is a list of model instances 
            representing diverse configurations to solve the Cold Start problem.
            Detects the task type, returns all the probes - linear, trees, boosting, kernel (linear, rbf with two different c values) as a dict 
        """

        if task_type == 'classification':
            return {
                'linear':[
                    LogisticRegression(max_iter=1000, C=0.1),
                    LogisticRegression(max_iter=1000, C=10)
                ],
                'trees':[
                    RandomForestClassifier(n_estimators=50, max_depth=5),
                    RandomForestClassifier(n_estimators=100, max_depth=None )
                ],
                'kernel': [SVC(kernel='linear', probability=True),
                    SVC(kernel='rbf', C=1, probability=True),
                    SVC(kernel='rbf', C=100, probability=True)
                ] ,
                'boosting':[
                    XGBClassifier(n_estimators=50, learning_rate= 0.1),
                    LGBMClassifier(n_estimators=50, learning_rate=0.05)
                ]
            }
        
        else:
            return{
                'linear':[
                    Ridge(alpha=0.1),
                    ElasticNet(alpha=0.5, l1_ratio=0.5)
                ],
                'trees': [
                    RandomForestRegressor(n_estimators=50, max_depth=20),
                    RandomForestRegressor(n_estimators=100, max_depth=None)
                ],
                'boosting': [
                    XGBRegressor(n_estimators=50, learning_rate=0.1),
                    GradientBoostingRegressor(n_estimators=50)
                ],
                'kernel': [
                    SVR(kernel='linear'),
                    SVR(kernel='rbf', C=1.0),
                    SVR(kernel='rbf', C=100.0)
                ]
            }
        
    @staticmethod
    def get_search_space(family:str, task_type:str):
        mapping={
            ('linear', 'classification'): LogisticRegression,
            ('trees', 'classification'): RandomForestClassifier,
            ('boosting', 'classification'): XGBClassifier,
            ('kernel', 'classification'): SVC,
            ('linear', 'regression'): Ridge,
            ('trees', 'regression'): RandomForestRegressor,
            ('boosting', 'regression'): XGBRegressor,
            ('kernel', 'regression'): SVR,
        }

        return mapping.get((family, task_type))
    