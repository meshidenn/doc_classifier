from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from db_setting import Base, ENGINE


class Config(Base):
    __tablename__ = 'config'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    algo = Column(String)
    hyper = Column(Text)

    def toDict(self):
        return {
            'id': self.id,
            'name': self.name,
            'algo': self.algo,
            'hyper': self.hyper
        }


def main():
    Base.metadata.create_all(bind=ENGINE)


if __name__ == "__main__":
    main()