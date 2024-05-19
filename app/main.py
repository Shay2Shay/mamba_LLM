'''
Fast API server
'''

from fastapi import FastAPI
from pydantic import BaseModel
from app.model import nshot_question

app = FastAPI()

class ModelInput(BaseModel):
    text: str

class Result(BaseModel):
    answer: str

@app.get('/')
def default():
    return {
        "status" : "OK",
        "api-version" : 0.1
    }

@app.post('/qna', response_model=Result)
def handler(payload: ModelInput):
    text = payload.text
    ans = nshot_question(text)
    return {
        "answer": ans
    }