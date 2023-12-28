from sqlalchemy import Boolean,Column,ForeignKey,Integer,String
from database import Base

class Question(Base):
    __tablename__ = 'questions'

    id= Column(Integer,primary_key=True,index=True)
    question_text = Column(String, index=True)
class Choices(Base):
    __tablename__ = 'choices'
    id= Column(Integer,primary_key=True,index=True)
    choice_text = Column(String, index=True)
    is_correct =Column(Boolean,default=False)
    question_id = Column(Integer,ForeignKey("questions.id"))

class Car(Base):
    __tablename__ = 'cars'
    id=Column(Integer,primary_key=True,index=True)
    car_year = Column(Integer)
    brand = Column(String, default="Mazda")
    model = Column(String)
    sub_model = Column(String)
    sub_model_name =Column(String)
    car_type = Column(String)
    transmission = Column(String)
    color = Column(String)
    model_year_start = Column(Integer)
    model_year_end = Column(Integer)
    mile = Column(Integer)
    cost = Column(Integer)

