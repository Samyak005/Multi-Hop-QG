from fastapi import FastAPI
from trained_model import get_inference
from pydantic import BaseModel
from typing import List

app = FastAPI()


@app.get('/')
def read_main():
    return {'message': 'Hello World'}


class Input(BaseModel):
    context: str
    answer: str
    model_name: str


@app.post("/generate_question/")
def generate_question(inp: Input):
    """
    Generates a question using specified context and answer.
    Returns generated question.
    """
    generated_question = get_inference(inp.answer, inp.context, inp.model_name)
    return {"question": generated_question,
            "input": inp}

