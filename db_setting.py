from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

ENGINE = create_engine('sqlite:///db.sqlite3')
Base = declarative_base()
