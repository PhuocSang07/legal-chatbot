from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class SQLDocument(Base):
    __tablename__ = "docstore"
    key = Column(String, primary_key=True)
    value = Column(JSONB)

    def __repr__(self):
        return f"<SQLDocument(key='{self.key}', value='{self.value}')>"