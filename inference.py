'''This file allows for batch inference on CLI. takes in the inference.py automatically and expects the csv file 
python inference.py data.csv 
This would translate to sys.argv= ['inference.py', 'data.csv']
'''
import pandas as pd
import joblib
import sys

def main(csv_path):
    try:
        engine=joblib.load('auraml_engine.joblib')
        data=pd.read_csv(csv_path)
        predictions= engine.predict(data)

        data['predictions'] =predictions
        output_name= f"result_{csv_path}"
        data.to_csv(output_name, index=False)
        print("Done. Results saved to {output_name}")


    except Exception as e:
        print(f"error : {e}")
    

if __name__== "__main__":
    if len(sys.argv) <2 :
        print('Usage :python inference.py <path to data.csv>')

    else:
        main(sys.argv[1])

    
