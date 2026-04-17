from fastapi import FastAPI, HTTPException 
import pandas as pd 
import joblib
import os
from contextlib import asynccontextmanager
from orchestrator import AuraML


@asynccontextmanager
async def lifespan (app: FastAPI):
    global engine
    model_path = 'auraml_engine.joblib'
    if os.path.exists(model_path):
        try:
            engine = joblib.load(model_path)
            print('Auraml engine loaded successfully via lifespan')
        except Exception as e:
            print(f"error loading engine :  {e}")

    else :
        print("AuraML engine joblib file not found")
    

    yield # app runs here 

    print("Shutting down auraml api")


app = FastAPI(
    title='AuraML API',
    description= "Modern API for Automatic ML",
    lifespan = lifespan
)

engine = None

@app.post("/predict") 
async def predict(data : list[dict]):
    if engine is None:
        raise HTTPException(status_code= 503,status='Model engine not loaded')
    try:
        input_df=pd.DataFrame(data)
        predictions= engine.predict(input_df)

        return {
            'status' : "success",
            'predictions' : predictions.to_list()
        }

    except Exception as e:
        raise HTTPException (status_code= 400, detail= str(e))

@app.get("/health")
def health():
    return {'status': 'online', 'engine_ready' : engine is not None}

