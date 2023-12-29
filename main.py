from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException, Depends, UploadFile, File
from typing import Optional ,List ,Annotated
from pydantic import BaseModel
import requests
import httpx
import models
from database import engine,SessionLocal
from sqlalchemy.orm import Session
import pandas as pd
import joblib
from PIL import Image
import io
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi.responses import JSONResponse
import torchvision
import torch.nn as nn
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
device = torch.device('cpu')
pretrain_weight = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_v2_s(weights = pretrain_weight)
model.classifier[1] = nn.Linear(1280, 102)
model = model.to(device)
class CarBase(BaseModel):
    car_year:int | None = None
    brand:str | None = None
    model:str| None = None
    sub_model:str| None = None
    sub_model_name:str| None = None
    car_type:str| None = None
    transmission:str| None = None
    color:str| None = None
    modelyear_start:int| None = None
    modelyear_end:int| None = None
    mile:int| None = None
    cost:int| None = None
def get_db():
    db= SessionLocal()
    try:
        yield db
    finally:
        db.close()
db_dependency = Annotated[Session,Depends(get_db)]    
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.507, 0.487, 0.441], std = [0.267, 0.256, 0.276])
    ])

def preprocess_image(image):
    image = transform(image)
    return image.unsqueeze(0)
def predictModel(model_path,contents):
    global model,device
    currentmodel =model
    currentmodel.load_state_dict(dict(torch.load(model_path,map_location=device)))
    currentmodel.eval()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        predictions = F.softmax(currentmodel(preprocessed_image), dim=1)
    return predictions.numpy()

@app.post("/predict/value")
async def predictValue(Id : int):
    try:
        df = pd.read_csv('preprocessed_data.csv')
        df = df.loc[df['Id'] == Id]
        df.drop(columns=['cost', 'car_model', 'Id'],inplace=True)
        rfModel = joblib.load("random_forest.model")
        predict = rfModel.predict(df)
        result = predict[0]
        return {"prediction":result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
@app.post("/predict/Color")
async def predictColor(file: UploadFile = File(...)):
    contents = await file.read()
    return predictModel("Color.pth",contents)

@app.post("/car/")
async def createNewCar(car:CarBase,db:db_dependency):
    if(car.car_year== None or car.brand== None or car.model== None or car.sub_model== None or car.sub_model_name== None or
        car.car_type== None or car.transmission== None or car.modelyear_start== None or car.modelyear_end== None or
        car.color== None or car.mile== None or car.cost==None):
        raise HTTPException(status_code=404,detail='Invalid car data')
    db_car = models.Car(car_year=car.car_year, brand=car.brand,model=car.model,sub_model=car.sub_model,sub_model_name=car.sub_model_name,
        car_type=car.car_type,transmission=car.transmission,model_year_start=car.modelyear_start,model_year_end=car.modelyear_end,
        color=car.color,mile=car.mile,cost=car.cost)
    print("Test")
    db.add(db_car)
    db.commit()
    return db_car
@app.get("/car/")
async def getAllCar(db:db_dependency):
    result = db.query(models.Car).all()
    return result
@app.get("/car/{car_id}")
async def getCarById(car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
    return result
@app.delete("/car/{car_id}")
async def deleteCarById(car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
    db.delete(result)
    db.commit()
    return []
@app.put("/car/{car_id}")
async def updateCarById(car:CarBase,car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
    if(car.car_year== None):
        car.car_year = result.car_year
    if(car.brand== None):
        car.brand = result.brand
    if(car.model== None):
        car.model = result.model
    if(car.sub_model== None):
        car.sub_model = result.sub_model
    if(car.sub_model_name== None):
        car.sub_model_name = result.sub_model_name
    if(car.car_type== None):
        car.car_type = result.car_type
    if(car.transmission== None):
        car.transmission = result.transmission
    if(car.modelyear_start== None):
        car.model_year_start = result.model_year_start
    if(car.modelyear_end== None):
        car.modelyear_end = result.model_year_end
    if(car.color== None):
        car.color = result.color
    if(car.mile== None):
        car.mile = result.mile
    if(car.cost==None):
        car.cost = result.cost
    result.car_year=car.car_year
    result.brand=car.brand
    result.model=car.model
    result.sub_model=car.sub_model
    result.sub_model_name=car.sub_model_name
    result.car_type=car.car_type
    result.transmission=car.transmission
    result.model_year_start=car.modelyear_start
    result.model_year_end=car.modelyear_end
    result.color=car.color
    result.mile=car.mile
    result.cost=car.cost
    db.add(result)
    db.commit()
    return result
    