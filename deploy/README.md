# Deploying model using FastAPI
#### Installing Requirements

```bash
# Installing FastAPI related libraries
pip install -r fastapi_requirements.txt
# Install pytorch and transformers 4.9.1 library
pip install transformers==4.9.1
```

#### Deploying backend

Put the pretrained model on the directory that contains `main.py` and `trained_model.py`. The name of the pretrained model should be `model_hotpot_last.pth` or you can edit `trained_model.py` if your pretrained model has a different name.

##### Starting backend

```bash
uvicorn main:app --reload # This will start FastAPI backend
```

#### How to use FastAPI backend to generate questions

Open browser, go to `http://127.0.0.1:8000/docs`

Go to `Try it out`

![Try API](./images/try_out)



Give the `context` and `answer`, then execute.

![Try API](./images/generate_request)



You can find the generated question in the response section below that.

![Try API](./images/generated_question)
