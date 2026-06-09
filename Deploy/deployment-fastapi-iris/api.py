from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("iris.joblib")

app = FastAPI(title="Iris Classification API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Classification API"}

@app.post("/predict")
def predict(data: IrisInput):
    X = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    pred = model.predict(X)[0]

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    return {
        "prediction": int(pred),
        "class_name": classes[pred]
    }

