from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

URL_DATABASE = 'postgresql://umrxgyay:xGMZMOffE9epo8PmX6dm4s0oIRldGNlr@rain.db.elephantsql.com/umrxgyay'
engine = create_engine(URL_DATABASE)
SessionLocal =sessionmaker(autocommit=False, autoflush=False,bind=engine)
Base=declarative_base()