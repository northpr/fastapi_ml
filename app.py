from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get('/hello')
def hi():
    return {'message' : 'Hi how are you?'}


@app.get('/hello/{name}')
def hi_wname(name:str):
    return {'message' : f'hi {name}'}

@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print(classifier.predict([[variance,skewness,curtosis,entropy]])) # 2 bracket
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction = 'Fake note'
    else:
        prediction = 'Its a bank note'
    return {'prediction' : prediction}