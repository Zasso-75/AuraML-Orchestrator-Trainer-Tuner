''' 
This file analyses the target column's distribution and data type, distinguishes between Regression and Classification.
Separates numerical and categorical columns.
Stores the imputed values while null handling to ensure consistent values during training and inference
High cardinality handling
Data Leak protection - checks for correlation between the features and the output var., drops any features that perfectly predict the output var. 
'''

import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder 


class DataCustodian:
        
    def __init__(self, target_col: str, include_dates : bool= True , selected_features : list= None):
        self.task_type= None
        self.target_col= target_col
        self.preprocessor= None    
        self.feature_cols = []
        self.is_fitted= False   
        self.selected_features = selected_features
        self.include_dates= include_dates  
        self.datetime_cols=[]       



    def detect_task(self, df:pd.DataFrame):
        '''
        This function will take the dataframe as input
        Will take the class var, self.target_col (so we need to provide the target col) and check the number of unique values in the col. 
        if target is of  OBJECT / BOOL type or has unique value count <15 it will assign it as classification task else it will assign in regression task in the self.task_tpye  class var.
        Returns the task type.
        '''
        target= df[self.target_col]
        if target.nunique() <15 or target.dtype == 'object' or target.dtype=='bool' :
            self.task_type= 'classification'
        else:
            self.task_type='regression'

        print(f"Detected task: {self.task_type} ")
        return self.task_type


    def _get_optimal_scaler(self, num_cols, X: pd.DataFrame):
        """
        Takes the training part and the  numeric columns as input and checks if the IQR condition is applicble to any column it selects the RobustScaler for the numeric columns else returns StandardScaler 
        checks if there are columns , if not it will return None
        it checks for the IQR and if the max value is > Q3 + 1.5* IQR it will use the Robust scaler and remove the outliers 
        """
        if not num_cols:
            return None

        for col in num_cols:
            q1, q3 = X[col].quantile(0.25) , X[col].quantile(0.75)
            iqr= q3-q1
            if X[col].max() > q3+ 1.5*iqr  or X[col].min() < q1-1.5*iqr :
                return RobustScaler()

        return StandardScaler() 

    

    def sanitize_and_extract(self, df:pd.DataFrame):
        """
        creates a copy of the df, if there are selected features already, take the selected features and the target col and creates the final df 
        then iterates over cols (skips the target col), checks if the col is object type converts it to numeric 
        for columns that are object type it tries converting the values numeric and checks how many got converted, if 0.9 ratio of the data in col got successfully converted we convert it and keep deal 
        with the reamining nulls 
        checks if we need to include the date cols using the class var include_dates and it isnt all numeric. 
        if these conditions meet it converts the column to datetime adn then splits out the year, month , day and day of week from that date, this datetime conversion happens inside a try except block 
        if it fails, pass.
        also the cols that converted get added to the datetime_cols and then remove from the final df 
        return the df
        """

        print("sanitizing df .... ")
        df=df.copy()   
        if self.selected_features:
            df=df[self.selected_features + [self.target_col]]
        
        for col in df.columns:  
            if col==self.target_col:
                pass
            
            if df[col].dtype == 'object':
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().mean() >= 0.9 :
                    print(f"converted col {col} to numeric")
                    df[col] = converted
            
            if self.include_dates and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    print(f"converting {col} to datetime ......")
                    df[col]=pd.to_datetime(df[col])

                    df[f'{col}_year']= df[col].dt.year
                    df[f'{col}_month']= df[col].dt.month
                    df[f'{col}_day']= df[col].dt.day
                    df[f'{col}_dow']= df[col].dt.dayofweek
                    self.datetime_cols.append(col)

                except:

                    print(f"couldnt convert {col} to datetime ..... ")
                
        return df.drop(columns=self.datetime_cols)

    

    def _build_pipeline(self, X:pd.DataFrame, y:pd.Series):
        '''
        selects numeric cols as list , cat cols as a list (includes objects category and bools ) as a list 
        selects the col with the lowest/ highest unique values
        calls the get optimal scaler function for the numeric cols 
        currently we are using median for imputing but it throws away information like data distribution etc. 
        When there is strong linear relationship between the features and the target imputing the median hurts linear model , mean preserves the expected value better 

        create a pipeline var for numeric cols , put the imputer strategy and the scaler if there are numeric cols, with the transformers
        create the transformers for the cat cols as well and then create the column transformer , remainder = drop 
        '''

        num_cols= X.select_dtypes(include=['number']).columns.tolist()
        print(f"follwing are the detected numerical columns : {num_cols}")
        cat_cols= X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        low_card = [col for col in cat_cols if X[col].nunique() <=10]
        high_card= [col for col in cat_cols if X[col].nunique() > 10]
        
        scaler= self._get_optimal_scaler(num_cols,X)
        print(f"The scaler selected for the numerical columns is : {scaler}")

        num_pipeline=  Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler )
        ]) if num_cols else None


        transformers= []
        if num_pipeline:
            transformers.append(('num', num_pipeline, num_cols))
        
        if low_card:
            transformers.append(('low_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False) , low_card))

        if high_card:
            transformers.append(('high_cat', TargetEncoder(), high_card))

        self.preprocessor= ColumnTransformer(transformers=transformers, remainder='drop')

    

    def prepare(self, df:pd.DataFrame, test_size=0.2):
        '''
        calls for all the funcitons. 
        calls for detect_task to know if regression or classificatoin
        the creates the df usng sanitize and extract function
        then x and y from the df 
        creates a list of feature cols and calls the build pipeline function on x and y 
        splits the x and y data  uses test_size default argument  and then calls the self.preprocessor.fit_transform for the training on xtrain and ytrain 
        and the self.preprocessor.transform on xtest and ytest , changes the flaf self.is_fitter to true and returns teh xtrain test and y train test
        '''

        self.detect_task(df)
        df= self.sanitize_and_extract(df)
        x= df.drop(columns= [self.target_col])
        y=df[self.target_col]
        self.feature_cols=  x.columns.tolist()
        self._build_pipeline(x,y)
    

        x_train_raw, x_test_raw, y_train, y_test = train_test_split(x,y, test_size= test_size, random_state=42)

        x_train = self.preprocessor.fit_transform(x_train_raw, y_train)
        x_test= self.preprocessor.transform(x_test_raw)
        print(f"Fitted the preprocessor and transformed the testing split as well, the shapes are now  x_train : {x_train.shape}, x_test : {x_test.shape}")
        self.is_fitted=True

        return x_train, x_test, y_train, y_test


                




