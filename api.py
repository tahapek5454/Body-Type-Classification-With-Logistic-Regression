from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from pydantic import BaseModel
from typing import List



class Model(BaseModel):
    height : float
    weight: float 
    armCircumference: float 
    chestCircumference: float
    shoulderCircumference: float
    waistCircumference: float
    legCircumference: float

class Item(BaseModel):
    data : List[Model]



class CustomData:
    
    def __init__(self, path="data.csv") -> None:
        self.dataset = pd.read_csv(path);
        self.classifier = None;
        self.sc = StandardScaler()

    def split_data(self):
         X = self.dataset.iloc[:, 0:-1].values
         y = self.dataset.iloc[:, -1].values
         self.X_train,  self.X_test,  self.y_train,  self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    def set_scaler_for_test(self):
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)

    def set_scaler(self, data):
        return self.sc.transform(data);


def convert_to_ndarray(item: Item) -> np.ndarray:
    data_as_ndarray = [
        [
            model.height,
            model.weight,
            model.armCircumference,
            model.chestCircumference,
            model.shoulderCircumference,
            model.waistCircumference,
            model.legCircumference,
        ]
        for model in item.data
    ]
    return np.array(data_as_ndarray)


cd = CustomData()
app = FastAPI()


@app.get("/")
def read_root():
    return "Success"


@app.post("/setData")
def set_data(path: str):
    cd.dataset = pd.read_csv(path);
    return "Success"


@app.get("/fitModel")
def fit_model():
    cd.split_data();
    cd.set_scaler_for_test();
    cd.classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
    cd.classifier.fit(cd.X_train, cd.y_train)

    y_pred = cd.classifier.predict(cd.X_test)
    
    cm = confusion_matrix(cd.y_test, y_pred)
    return {"predData": y_pred.tolist(), "confusionMatrix":cm.tolist()}


@app.post("/predict")
def data_predict(payload: Item):
    items_as_ndarray = convert_to_ndarray(payload) 
    scaled_data = cd.set_scaler(items_as_ndarray)

    y_pred = cd.classifier.predict(scaled_data)

    return {"predData": y_pred.tolist()}


if __name__ == '__main__':
    #for app
    uvicorn.run(app,host="127.0.0.1",port=8000)
    #for docker
    # uvicorn.run(app,host="0.0.0.0",port=8000)


















