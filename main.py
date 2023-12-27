from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException, Depends
from typing import Optional ,List ,Annotated
from pydantic import BaseModel
import requests
import httpx
import models
from database import engine,SessionLocal
from sqlalchemy.orm import Session
import pandas as pd
import joblib
app = FastAPI()
models.Base.metadata.create_all(bind=engine)

class ChoiceBase(BaseModel):
    choice_text:str
    is_correct:bool
class QuestionBase(BaseModel):
    question_text:str
    choices: List[ChoiceBase]
def get_db():
    db= SessionLocal()
    try:
        yield db
    finally:
        db.close()
db_dependency = Annotated[Session,Depends(get_db)]    

@app.post("/questions/")
async def create_questions(question:QuestionBase,db:db_dependency):
    db_question = models.Question(question_text=question.question_text)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    for choice in question.choices:
        db_choice = models.Choices(choice_text=choice.choice_text,is_correct=choice.is_correct,question_id = db_question.id)
        db.add(db_choice)
    db.commit()
    return db_question
@app.get("/questions/")
async def read_question(db:db_dependency):
    result = db.query(models.Question).all()
    return result
@app.get("/questions/{question_id}")
async def read_question(question_id:int,db:db_dependency):
    result = db.query(models.Question).filter(models.Question.id == question_id).first()
    if not result:
        raise HTTPException(status_code=404,detail='Question not found')
    return result
@app.delete("/question/{question_id}")
async def delete_question(question_id:int,db:db_dependency):
    result = db.query(models.Choices).filter(models.Choices.question_id == question_id).all()
    for res in result:
        db.delete(res)
    db.commit()
    result = db.query(models.Question).filter(models.Question.id == question_id).first()
    db.delete(result)
    db.commit()
@app.put("/questions/{question_id}")
async def update_question(question_text:str,question_id:int,db:db_dependency):
    result = db.query(models.Question).filter(models.Question.id == question_id).first()
    result.question_text = question_text
    db.add(result)
    db.commit()
@app.get("/choices/{question_id}")
async def read_choices(question_id:int,db:db_dependency):
    result = db.query(models.Choices).filter(models.Choices.question_id == question_id).all()
    if not result:
        raise HTTPException(status_code=404,detail='Question not found')
    return result

@app.post("/predict")
async def predict(Id : int):
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

@app.get('/get_firstuser')
def first_user():
    #Synchronous
    api_url = "https://jsonplaceholder.typicode.com/users"
    all_users = requests.get(api_url).json()
    user1 = all_users[0]
    name = user1["name"]
    email = user1["email"]
    return {'name': name, "email": email}
@app.get('/get_seconduser')
async def second_user():
    #Asynchronous
    api_url = "https://jsonplaceholder.typicode.com/users"
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        all_users = response.json()
        user2 = all_users[1]
        name = user2["name"]
        email = user2["email"]
        return {'name': name, "email": email}
    