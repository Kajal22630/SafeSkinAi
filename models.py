from sqlalchemy import Column, Integer, String, Float
from database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)

    patient_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    doctor_name = Column(String)

    diagnosis = Column(String)
    probability = Column(Float)
    risk = Column(String)

    original_path = Column(String)
    gradcam_path = Column(String)
    comparison_path = Column(String)