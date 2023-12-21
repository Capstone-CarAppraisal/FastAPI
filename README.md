# This is how to use FastAPI
Firstly, you must to install fastapi and some other packages. This project use FastAPI and postgresql database(on elephantsql cloud)
## Installation
If you use Mac, you must to use command below this
```
python3 -m venv env
source env/bin/activate
```
Next, install package(FastAPI, )
```
pip install fastapi sqlalchemy psycopg2-binary
```
## Add database file
Next, you must create database file with code below this, you must to edit code in line URL_DATABASE = REPLACE_URL_DATABASE with your url of database
```
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
URL_DATABASE = REPLACE_URL_DATABASE
engine = create_engine(URL_DATABASE)
SessionLocal =sessionmaker(autocommit=False, autoflush=False,bind=engine)
Base=declarative_base()
```
## Run FastAPI
If you didn't have uvicorn you must to install it before
```
pip install uvicorn
```
After you create database.py file. Next, you can run FastAPI with command below this. 
```
uvicorn main:app --reload
```
After run command you can use api request via http://127.0.0.1:8000/docs
## Edit code
If you want to change model or add more model you must edit on models.py file

If you want to edit API you must edit on main.py file