from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from sqlalchemy.orm import Session


SQLALCHEMY_DATABASE_URL = "postgresql://postgres:Jx7GvARRBIQMeJ1Nh7N7Lay0LyjwjMDY@a6b1f99f-f52f-4ae3-9034-fed8186b24a9.hsvc.ir:32123"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Image(Base):
    __tablename__ = "Image"
    id = Column(Integer, primary_key=True, index=True)
    image = Column(String)
    class_name_predicted = Column(String)
    class_name_gt = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=None)


class Image_record(BaseModel):
    image: str
    class_name_predicted: str
    class_name_gt: str
    
class DB:
    def __init__(self):
        self.db = SessionLocal()
    
    def add_image(self, record: Image_record):
        db_image = Image(
            image = record.image, 
            class_name_predicted = record.class_name_predicted,
            class_name_gt = record.class_name_gt,
            )
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        return db_image


    def get_image(self, skip: int = 0, limit: int = 100):
        return self.db.query(Image).offset(skip).limit(limit).all()
    