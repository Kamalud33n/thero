"""
models.py
All SQLAlchemy ORM models for the Rehabilitation AI System.

Kept separate from app.py so the data layer (tables/columns/relationships)
is easy to find, review, and migrate independently of the API/route code.

Import models from here wherever they're needed, e.g.:
    from models import Patient, SessionModel, JointAngle, ExerciseResult, Report, Setting, History
"""

import uuid

from sqlalchemy import (
    Column, String, Integer, Float,
    DateTime, Text, Boolean, ForeignKey, JSON, LargeBinary
)
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


# ─── ID generators ──────────────────────────────────────────────────────────
def new_patient_id() -> str:
    return f"PAT-{uuid.uuid4().hex[:8].upper()}"


def new_session_id() -> str:
    return f"SES-{uuid.uuid4().hex[:8].upper()}"


# ─── Models ──────────────────────────────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"
    id                 = Column(String(50), primary_key=True, default=new_patient_id)
    name               = Column(String(100), nullable=False)
    age                = Column(Integer, nullable=False)
    gender             = Column(String(10), nullable=False)
    weight             = Column(Float, nullable=True)
    height             = Column(Float, nullable=True)
    diagnosis          = Column(String(200), nullable=True)
    affected_body_part = Column(String(100), nullable=True)
    doctor_name        = Column(String(100), nullable=True)
    therapist_name     = Column(String(100), nullable=True)
    phone              = Column(String(20), nullable=True)
    email              = Column(String(100), nullable=True)
    medical_history    = Column(Text, nullable=True)
    previous_injury    = Column(Text, nullable=True)
    current_treatment  = Column(Text, nullable=True)
    exercise_plan      = Column(Text, nullable=True)
    photo              = Column(LargeBinary().with_variant(LONGBLOB, "mysql"), nullable=True)
    date_created       = Column(DateTime, default=func.now())
    is_active          = Column(Boolean, default=True)
    sessions = relationship("SessionModel", back_populates="patient", cascade="all, delete-orphan")


class SessionModel(Base):
    __tablename__ = "sessions"
    id                  = Column(String(50), primary_key=True, default=new_session_id)
    patient_id          = Column(String(50), ForeignKey("patients.id"), nullable=False)
    exercise_type       = Column(String(100), nullable=False)
    start_time          = Column(DateTime, default=func.now())
    end_time            = Column(DateTime, nullable=True)
    duration_seconds    = Column(Integer, nullable=True)
    total_reps          = Column(Integer, default=0)
    completed_reps      = Column(Integer, default=0)
    accuracy_percentage = Column(Float, default=0.0)
    average_rom         = Column(Float, default=0.0)
    incorrect_movements = Column(Integer, default=0)
    stability_score     = Column(Float, default=0.0)
    balance_score       = Column(Float, default=0.0)
    movement_smoothness = Column(Float, default=0.0)
    fatigue_estimation  = Column(Float, default=0.0)
    recovery_score      = Column(Float, default=0.0)
    session_data        = Column(JSON, nullable=True)
    patient          = relationship("Patient", back_populates="sessions")
    joint_angles     = relationship("JointAngle", back_populates="session", cascade="all, delete-orphan")
    exercise_results = relationship("ExerciseResult", back_populates="session", cascade="all, delete-orphan")


class JointAngle(Base):
    __tablename__ = "joint_angles"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    timestamp    = Column(DateTime, default=func.now())
    joint_name   = Column(String(50), nullable=False)
    angle_value  = Column(Float, nullable=False)
    target_angle = Column(Float, nullable=True)
    deviation    = Column(Float, nullable=True)
    is_correct   = Column(Boolean, default=True)
    session = relationship("SessionModel", back_populates="joint_angles")


class ExerciseResult(Base):
    __tablename__ = "exercise_results"
    id                 = Column(Integer, primary_key=True, autoincrement=True)
    session_id         = Column(String(50), ForeignKey("sessions.id"), nullable=False)
    exercise_name      = Column(String(100), nullable=False)
    repetition_number  = Column(Integer, nullable=False)
    accuracy           = Column(Float, default=0.0)
    rom_achieved       = Column(Float, default=0.0)
    speed              = Column(Float, default=0.0)
    hold_duration      = Column(Float, default=0.0)
    compensation_score = Column(Float, default=0.0)
    is_completed       = Column(Boolean, default=False)
    feedback           = Column(Text, nullable=True)
    timestamp          = Column(DateTime, default=func.now())
    session = relationship("SessionModel", back_populates="exercise_results")


class Report(Base):
    __tablename__ = "reports"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    patient_id     = Column(String(50), ForeignKey("patients.id"), nullable=False)
    report_type    = Column(String(50), nullable=False)
    generated_date = Column(DateTime, default=func.now())
    file_path      = Column(String(200), nullable=True)
    report_data    = Column(JSON, nullable=True)


class Setting(Base):
    __tablename__ = "settings"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    key         = Column(String(50), unique=True, nullable=False)
    value       = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    updated_at  = Column(DateTime, default=func.now(), onupdate=func.now())


class History(Base):
    __tablename__ = "history"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.id"), nullable=False)
    action     = Column(String(100), nullable=False)
    details    = Column(Text, nullable=True)
    timestamp  = Column(DateTime, default=func.now())