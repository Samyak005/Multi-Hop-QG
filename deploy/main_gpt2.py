from fastapi import FastAPI
from trained_gpt_model import get_inference2
from pydantic import BaseModel
from typing import List

app = FastAPI()


@app.get('/')
def read_main():
    return {'message': 'Hello World'}


class Input(BaseModel):
    context: str
    answer: str


@app.post("/generate_question2/")
def generate_question2(inp: Input):
    generated_question2 = get_inference2(inp.answer, inp.context)
    return {"question": generated_question2,
            "input": inp}
