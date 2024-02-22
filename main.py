from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from typing import Annotated
from pydantic import BaseModel
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import pandas as pd
from PIL import Image
import io
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
import pickle
from sqlalchemy import func
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
app = FastAPI()
models.Base.metadata.create_all(bind=engine)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarBase(BaseModel):
    car_year: int
    brand: str
    model: str
    sub_model: str
    sub_model_name: str
    car_type: str
    transmission: str
    color: str
    modelyear_start: int
    modelyear_end: int
    mile: int
    cost: int


class CarBaseNoCost(BaseModel):
    car_year: int
    brand: str
    model: str
    sub_model: str
    sub_model_name: str
    car_type: str
    transmission: str
    color: str
    modelyear_start: int
    modelyear_end: int
    mile: int

class CarPredictRequest(BaseModel):
    front: Optional[int] = None
    rear: Optional[int] = None
    sidefront: Optional[int] = None
    siderear: Optional[int] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 


db_dependency = Annotated[Session, Depends(get_db)]
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[
                             0.267, 0.256, 0.276])
     ])


def predict_image(model_path, contents):
    device = torch.device('cpu')
    if isinstance(torch.load(model_path, map_location=device), dict):
        pretrain_weight = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        MLmodel = torchvision.models.efficientnet_v2_s(weights=pretrain_weight)
        MLmodel.classifier[1] = nn.Linear(1280, 102)
        MLmodel.load_state_dict(torch.load(model_path, map_location=device))
    else:
        MLmodel = torch.load(model_path, map_location=device)
    MLmodel.eval()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    preprocessed_image = preprocess_image(image)
    with torch.no_grad():
        image = preprocessed_image.to(device)
        output = MLmodel(image)
    predict = 0
    max = output[0][0]
    n = 0
    for i in output[0]:
        if max < i:
            max = i
            predict = n
        n += 1
    return predict


def preprocess_image(image):
    image = transform(image)
    return image.unsqueeze(0)


def decoder_model_car(predict, i):
    car_model = "error"
    car_year = "error"
    car_door = "error"
    if predict == 0:
        car_model = "2"
        car_year = "09-14"
        car_door = 4
    elif predict == 1:
        car_model = "2"
        car_year = "09-14"
        car_door = 5
    elif predict == 2:
        car_model = "2"
        car_year = "14-23"
        car_door = 4
    elif predict == 3:
        car_model = "2"
        car_year = "14-23"
        car_door = 5
    elif predict == 4:
        car_model = "3"
        car_year = "05-10"
        car_door = 4
    elif predict == 5:
        car_model = "3"
        car_year = "05-10"
        car_door = 5
    elif predict == 6:
        car_model = "3"
        car_year = "11-14"
        car_door = 4
    elif predict == 7:
        car_model = "3"
        car_year = "11-14"
        car_door = 5
    elif predict == 8:
        car_model = "3"
        car_year = "14-19"
        car_door = 4
    elif predict == 9:
        car_model = "3"
        car_year = "14-19"
        car_door = 5
    elif predict == 10:
        car_model = "3"
        car_year = "19-23"
        car_door = 4
    elif predict == 11:
        car_model = "3"
        car_year = "19-23"
        car_door = 5
    elif predict == 12:
        car_model = "BT50"
        car_year = "06-10"
        car_door = "-"
    elif predict == 13:
        car_model = "BT50"
        car_year = "11-20"
        car_door = "-"
    elif predict == 14:
        car_model = "BT50"
        car_year = "20-23"
        car_door = "-"
    elif predict == 15:
        car_model = "CX3"
        car_year = "15-23"
        car_door = "-"
    elif predict == 16:
        car_model = "CX5"
        car_year = "13-17"
        car_door = "-"
    elif predict == 17:
        car_model = "CX8"
        car_year = "17-23"
        car_door = "-"
    elif predict == 18:
        car_model = "CX30"
        car_year = "17-23"
        car_door = "-"
    return {"prediction": predict,
            "Brand": "Mazda",
            "ImageId": i,
            "Model": car_model,
            "ModelYear": car_year,
            "Door": car_door
            }
    
def predict_one(_1:int|None=None,_2:int|None=None,_3:int|None=None,_4:int|None=None):
    L = list()
    L.append(_1)
    L.append(_2)
    L.append(_3)
    L.append(_4)
    max_count =0
    cnt=0
    ans= None
    for i in range(4):
        if L[i] == None:continue
        cnt=0
        for j in L:
            if j == L[i]:cnt+=1
        if cnt>max_count:
            max_count=cnt
            ans=L[i]
    return ans

@app.post("/predict/value")
async def predict_value(car: CarBaseNoCost):
    try:
        d = {'car_year': [car.car_year], 'brand': [car.brand], 'model': [car.model], 'sub_model': [car.sub_model], 'sub_model_name': [car.sub_model_name],
             'car_type': [car.car_type], 'transmission': [car.transmission], 'model_year_start': [car.modelyear_start], 'model_year_end': [car.modelyear_end],
             'color': [car.color], 'mile': [car.mile]}
        df = pd.DataFrame(data=d)
        model_file = open('preprocessor.model', 'rb')
        preprocessor = pickle.load(model_file)
        model_file.close()
        df = preprocessor.transform(df)
        model_file = open('catboost.model', 'rb')
        valueModel = pickle.load(model_file)
        model_file.close()
        predict = valueModel.predict(df)
        result = predict[0]
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making predictions: {str(e)}")


@app.post("/predict/front")
async def predict_front(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("front.pt", contents)
        predictions.append(decoder_model_car(predict, i))
        i += 1
    return {"All prediction": predictions}


@app.post("/predict/rear")
async def predict_rear(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("rear.pt", contents)
        predictions.append(decoder_model_car(predict, i))
        i += 1
    return {"All prediction": predictions}


@app.post("/predict/sidefront")
async def predict_sidefront(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("sidefront.pt", contents)
        predictions.append(decoder_model_car(predict, i))
        i += 1
    return {"All prediction": predictions}


@app.post("/predict/siderear")
async def predict_siderear(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("siderear.pt", contents)
        predictions.append(decoder_model_car(predict, i))
        i += 1
    return {"All prediction": predictions}


@app.post("/predict/fourside")
async def predict_fourside(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("4side.pt", contents)
        predictions.append(decoder_model_car(predict, i))
        i += 1
    return {"All prediction": predictions}


@app.post("/predict/color")
async def predict_color(files: list[UploadFile] = File(...)):
    predictions = []
    i = 0
    for file in files:
        contents = await file.read()
        predict = predict_image("color.pt", contents)
        colors = ["Black", "Blue", "Brown", "Green",
                  "Grey", "Light Blue", "Red", "White"]
        predictions.append({"prediction": predict,
                            "color": colors[predict],
                            "ImageId": i
                            })
        i += 1
    return {"All prediction": predictions}

@app.post("/predict/onecar")
async def predict_onecar(request_body: CarPredictRequest):
    _1 = request_body.front
    _2 = request_body.rear
    _3 = request_body.sidefront
    _4 = request_body.siderear
    car = predict_one(_1, _2, _3, _4)
    predictions = decoder_model_car(car, 0)
    
    return {"Prediction": predictions}
    
@app.post("/predict/onecolor")
async def predict_onecar(request_body: CarPredictRequest):
    _1 = request_body.front
    _2 = request_body.rear
    _3 = request_body.sidefront
    _4 = request_body.siderear
    car = predict_one(_1,_2,_3,_4)
    colors = ["Black", "Blue", "Brown", "Green",
                  "Grey", "Light Blue", "Red", "White"]
    predictions = {"prediction": car,
                            "color": colors[car]}
    return {"Prediction":predictions
            }

@app.post("/car/")
async def create_new_car(car: CarBase, db: db_dependency):
    db_car = models.Car(car_year=car.car_year, brand=car.brand, model=car.model, sub_model=car.sub_model, sub_model_name=car.sub_model_name,
                        car_type=car.car_type, transmission=car.transmission, model_year_start=car.modelyear_start, model_year_end=car.modelyear_end,
                        color=car.color, mile=car.mile, cost=car.cost)
    print("Test")
    db.add(db_car)
    db.commit()
    return db_car


@app.get("/car/")
async def get_all_car(db: db_dependency):
    result = db.query(models.Car).all()
    return result


@app.get("/car/{car_id}")
async def get_car_by_id(car_id: int, db: db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='Car not found')
    return result


@app.delete("/car/{car_id}")
async def delete_car_by_id(car_id: int, db: db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='Car not found')
    db.delete(result)
    db.commit()
    return []


@app.put("/car/{car_id}")
async def update_car_by_id(car: CarBase, car_id: int, db: db_dependency):
    result = db.query(models.Car).filter(models.Car.id == car_id).first()
    if not result:
        raise HTTPException(status_code=404, detail='Car not found')
    result.car_year = car.car_year
    result.brand = car.brand
    result.model = car.model
    result.sub_model = car.sub_model
    result.sub_model_name = car.sub_model_name
    result.car_type = car.car_type
    result.transmission = car.transmission
    result.model_year_start = car.modelyear_start
    result.model_year_end = car.modelyear_end
    result.color = car.color
    result.mile = car.mile
    result.cost = car.cost
    db.add(result)
    db.commit()
    return result


@app.get("/car_market_detail")
async def get_car_market_detail(db: db_dependency, car_year: str, brand: str | None = None, model: str | None = None, sub_model: str | None = None, sub_model_name: str | None = None, car_type: str | None = None, predict_value: float | None = None):
    db_query = db.query(models.Car)
    if brand != None:
        db_query = db_query.filter(models.Car.brand == brand)
    if model != None:
        db_query = db_query.filter(models.Car.model == model)
    if sub_model != None:
        db_query = db_query.filter(models.Car.sub_model == sub_model)
    if sub_model_name != None:
        db_query = db_query.filter(models.Car.sub_model_name == sub_model_name)
    if car_type != None:
        db_query = db_query.filter(models.Car.car_type == car_type)

    avg_cost = db_query.with_entities(
        func.avg(models.Car.cost).label('avg_cost')).scalar()
    sd_cost = db_query.with_entities(func.stddev(
        models.Car.cost).label('sd_cost')).scalar()
    avg_mile = db_query.with_entities(
        func.avg(models.Car.mile).label('avg_cost')).scalar()
    count_car = db_query.with_entities(func.count(
        models.Car.id).label('record_count')).scalar()
    try:
        car_year = int(car_year)
        sub_model = float(sub_model)
    except:
        sub_model
    df_one2car = pd.read_csv('one2car_map.csv')
    df_one2car_filter = df_one2car[
        (df_one2car['brand'] == brand) &
        (df_one2car['car_year'] == car_year) &
        (df_one2car['model'] == model) &
        (df_one2car['sub_model'] == sub_model) &
        (df_one2car['sub_model_name'] == sub_model_name) &
        (df_one2car['car_type'] == car_type)
    ]
    id = df_one2car_filter.iloc[0]
    id = id['ttb_bluebook_id']
    df_ttb = pd.read_csv('ttb_map.csv')
    first_car_cost = df_ttb['1st_hand_price'][id].item()
    all_query = db_query.all()
    data = 1
    json_all_data = list()
    for query in all_query:
        json_data = query.__dict__
        json_all_data.append(json_data)
    sort_data = sorted(json_all_data, key=lambda x: x['cost'], reverse=True)
    for i, item in enumerate(sort_data, start=1):
        item['rank'] = i
    max_price = None
    min_price = None
    try:
        max_price = sort_data[0]['cost']
        min_price = sort_data[-1]['cost']
    except:
        max_price
    car_show = list()
    if (len(sort_data) <= 5):
        car_show = sort_data
    else:
        car_show.append(sort_data[0])
        closest = abs(sort_data[0]['cost']-predict_value)
        i = 0
        index = 0
        for data in sort_data:
            if closest > abs(data['cost']-predict_value):
                closest = abs(data['cost']-predict_value)
                index = i
            i += 1
        car_show.append(sort_data[index-1])
        car_show.append(sort_data[index])
        car_show.append(sort_data[index+1])
        car_show.append(sort_data[-1])
    return {
        "First car cost": first_car_cost,
        "Average Cost": avg_cost,
        "SD Cost": sd_cost,
        "Average Mile": avg_mile,
        "Number of Cars": count_car,
        "Max price": max_price,
        "Min price": min_price,
        "Car show": car_show
    }
