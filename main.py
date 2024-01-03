import collections
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
import pickle
app = FastAPI()
models.Base.metadata.create_all(bind=engine)
class CarBase(BaseModel):
    car_year:int 
    brand:str 
    model:str
    sub_model:str
    sub_model_name:str
    car_type:str
    transmission:str
    color:str
    modelyear_start:int
    modelyear_end:int
    mile:int
    cost:int
class CarBaseNoCost(BaseModel):
    car_year:int 
    brand:str 
    model:str
    sub_model:str
    sub_model_name:str
    car_type:str
    transmission:str
    color:str
    modelyear_start:int
    modelyear_end:int
    mile:int
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
def predict_image(model_path,contents):
    device = torch.device('cpu')
    MLmodel = torch.load(model_path,map_location=device)
    MLmodel.eval()    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        image = preprocessed_image.to(device)
        output = MLmodel(image)
    predict =0
    max=output[0][0]
    n=0
    for i in output[0]:
        if max<i:
            max=i
            predict=n
        n+=1
    return predict
def preprocess_image(image):
    image = transform(image)
    return image.unsqueeze(0)
def decoder_model_car(predict):
    car_model ="error"
    car_year = "error"
    car_door = "error"
    if predict==0:
        car_model = "2";car_year = "09-14";car_door=4
    elif predict==1:
        car_model = "2";car_year = "09-14";car_door=5
    elif predict==2:
        car_model = "2";car_year = "14-23";car_door=4
    elif predict==3:
        car_model = "2";car_year = "14-23";car_door=5
    elif predict==4:
        car_model = "3";car_year = "05-10";car_door=4
    elif predict==5:
        car_model = "3";car_year = "05-10";car_door=5
    elif predict==6:
        car_model = "3";car_year = "11-14";car_door=4
    elif predict==7:
        car_model = "3";car_year = "11-14";car_door=5
    elif predict==8:
        car_model = "3";car_year = "14-19";car_door=4
    elif predict==9:
        car_model = "3";car_year = "14-19";car_door=5
    elif predict==10:
        car_model = "3";car_year = "19-23";car_door=4
    elif predict==11:
        car_model = "3";car_year = "19-23";car_door=5
    elif predict==12:
        car_model = "BT50";car_year = "06-10";car_door="-"
    elif predict==13:
        car_model = "BT50";car_year = "11-20";car_door="-"
    elif predict==14:
        car_model = "BT50";car_year = "20-23";car_door="-"
    elif predict==15:
        car_model = "CX3";car_year = "15-23";car_door="-"
    elif predict==16:
        car_model = "CX5";car_year = "13-17";car_door="-"
    elif predict==17:
        car_model = "CX8";car_year = "17-23";car_door="-"
    elif predict==18:
        car_model = "CX30";car_year = "17-23";car_door="-"
    return car_model,car_year,car_door
@app.post("/predict/value")
async def predict_value(car : CarBaseNoCost):
    try:
        d = {'car_year':[car.car_year],'brand':[car.brand],'model':[car.model],'sub_model':[car.sub_model],'sub_model_name':[car.sub_model_name],
        'car_type':[car.car_type],'transmission':[car.transmission],'model_year_start':[car.modelyear_start],'model_year_end':[car.modelyear_end],
        'color':[car.color],'mile':[car.mile]}
        df =pd.DataFrame(data=d)        
        model_file = open('preprocessor.model', 'rb')
        preprocessor = pickle.load(model_file)
        model_file.close()
        df=preprocessor.transform(df)
        model_file = open('catboost.model', 'rb')
        valueModel = pickle.load(model_file)
        model_file.close()
        predict = valueModel.predict(df)
        result = predict[0]
        return {"prediction":result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
@app.post("/predict/front")
async def predict_rear(file: UploadFile = File(...)):
    #Wait for front.pt
    contents = await file.read()
    predict = predict_image("front.pt",contents)
    car_model, car_year, car_door =decoder_model_car(predict)
    return {"prediction":predict,
            "Brand":"Mazda",
            "Model":car_model,
            "ModelYear":car_year,
            "Door":car_door
            }
@app.post("/predict/rear")
async def predict_rear(file: UploadFile = File(...)):
    contents = await file.read()
    predict = predict_image("rear.pt",contents)
    car_model, car_year, car_door =decoder_model_car(predict)
    return {"prediction":predict,
            "Brand":"Mazda",
            "Model":car_model,
            "ModelYear":car_year,
            "Door":car_door
            }
@app.post("/predict/sidefront")
async def predict_sidefront(file: UploadFile = File(...)):
    contents = await file.read()
    predict = predict_image("sidefront.pt",contents)
    car_model, car_year, car_door =decoder_model_car(predict)
    return {"prediction":predict,
            "Brand":"Mazda",
            "Model":car_model,
            "ModelYear":car_year,
            "Door":car_door
            }
@app.post("/predict/siderear")
async def predict_siderear(file: UploadFile = File(...)):
    contents = await file.read()
    predict = predict_image("siderear.pt",contents)
    car_model, car_year, car_door =decoder_model_car(predict)
    return {"prediction":predict,
            "Brand":"Mazda",
            "Model":car_model,
            "ModelYear":car_year,
            "Door":car_door
            }
@app.post("/predict/whole")
async def predict_whole(file: UploadFile = File(...)):
    contents = await file.read()
    predict = predict_image("whole.pt",contents)
    car_model, car_year, car_door =decoder_model_car(predict)
    return {"prediction":predict,
            "Brand":"Mazda",
            "Model":car_model,
            "ModelYear":car_year,
            "Door":car_door
            }
@app.post("/predict/color")
async def predict_color(file: UploadFile = File(...)):
    #Wait for color.pt
    contents = await file.read()
    predict = predict_image("color.pt",contents)
    colors = ["Black","Blue","Brown","Green","Grey","Light Blue","Red","White"]
    return {"prediction":predict,
            "color":colors[predict]
            }
@app.post("/car/")
async def create_new_car(car:CarBase,db:db_dependency):
    db_car = models.Car(car_year=car.car_year, brand=car.brand,model=car.model,sub_model=car.sub_model,sub_model_name=car.sub_model_name,
        car_type=car.car_type,transmission=car.transmission,model_year_start=car.modelyear_start,model_year_end=car.modelyear_end,
        color=car.color,mile=car.mile,cost=car.cost)
    print("Test")
    db.add(db_car)
    db.commit()
    return db_car
@app.get("/car/")
async def get_all_car(db:db_dependency):
    result = db.query(models.Car).all()
    return result
@app.get("/car/{car_id}")
async def get_car_by_id(car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
    return result
@app.delete("/car/{car_id}")
async def delete_car_by_id(car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
    db.delete(result)
    db.commit()
    return []
@app.put("/car/{car_id}")
async def update_car_by_id(car:CarBase,car_id:int,db:db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Car not found')
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
    