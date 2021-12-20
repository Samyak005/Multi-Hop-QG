# Deploying model using FastAPI
#### Installing Requirements

```bash
# Installing FastAPI related libraries
pip install -r fastapi_requirements.txt
# Install pytorch and transformers 4.9.1 library
pip install transformers==4.9.1
```

## Trained Models

We have trained three models, two of them trained with both supporting facts and full context.

Models details and corresponding names are as follows.

1. `t5_supp` - T5 with supporting facts.
2. `t5_full` - T5 with full context.
3. `gpt2` - GPT2 with full context.
4. `bart_supp` - BART with supporting facts.
5. `bart_full`- BART with full context.

The code for deployment assumes that the models except `gpt2` are in the folder `../trained_models`.

## Deploying Backend

Put the pretrained models on the appropriate directories or modify the path files in the code. 

##### Starting backend

```bash
uvicorn main:app --reload # This will start FastAPI backend
```

#### How to use FastAPI backend to generate questions

Open browser, go to `http://127.0.0.1:8000/docs`

Go to `Try it out`

![Try API](./images/try_out)



Give the `context`, `answer`, and `model_name`.  Then execute.

![Try API](./images/generate_request)

You can find the generated question in the response section below that.

![Try API](./images/generated_question)
