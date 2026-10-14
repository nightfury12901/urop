import os
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "parking.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}?check_same_thread=False"

from sqlalchemy import event
from sqlalchemy.pool import NullPool

engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=False)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False)           # 'STUDENT', 'TEACHER', 'VIP'
    vehicle_number = Column(String(20), nullable=False)
    vehicle_type = Column(String(20), nullable=False)   # 'TWO_WHEELER', 'FOUR_WHEELER'
    created_at = Column(DateTime, default=get_utc_now)
    
    reservations = relationship("Reservation", back_populates="user")

class Slot(Base):
    __tablename__ = "slots"
    
    id = Column(String(10), primary_key=True)           # e.g. 'P1-01', 'P3-07'
    area_id = Column(String(10), nullable=False, index=True) # 'P1', 'P2', 'P3', 'P4'
    slot_number = Column(Integer, nullable=False)
    slot_type = Column(String(20), nullable=False)      # 'BIKES_ONLY', 'CAR_OR_2BIKES'
    state = Column(String(20), default="FREE")          # 'FREE', 'RESERVED', 'OCCUPIED', 'MAINTENANCE'
    bike_capacity = Column(Integer, default=2)
    current_bikes = Column(Integer, default=0)
    has_car = Column(Boolean, default=False)
    distance_from_gate = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)
    
    reservations = relationship("Reservation", back_populates="slot")

class Reservation(Base):
    __tablename__ = "reservations"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    slot_id = Column(String(10), ForeignKey("slots.id"), nullable=False)
    vehicle_type = Column(String(20), nullable=False)
    vehicle_number = Column(String(20), nullable=False)
    status = Column(String(20), default="ACTIVE")       # 'ACTIVE', 'COMPLETED', 'CANCELLED', 'EXPIRED'
    reserved_at = Column(DateTime, default=get_utc_now)
    expires_at = Column(DateTime, nullable=False)
    extensions_used = Column(Integer, default=0)
    idempotency_key = Column(String(64), unique=True, nullable=True)
    
    user = relationship("User", back_populates="reservations")
    slot = relationship("Slot", back_populates="reservations")

class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slot_id = Column(String(10), nullable=False, index=True)
    detected_state = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=get_utc_now)

def init_db(config_path: str):
    """Initializes tables and synchronizes the parking slots from campus_config.json."""
    import json
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        slots_data = cfg.get("slots", [])
        expected_ids = {s["id"] for s in slots_data}

        # Remove obsolete slots (e.g. from previous 50-slot config)
        obsolete_slots = db.query(Slot).filter(~Slot.id.in_(expected_ids)).all()
        for obs in obsolete_slots:
            db.query(Reservation).filter(Reservation.slot_id == obs.id).delete()
            db.delete(obs)
        if obsolete_slots:
            db.commit()

        # Insert or update valid slots
        for s in slots_data:
            existing = db.query(Slot).filter(Slot.id == s["id"]).first()
            if not existing:
                slot_obj = Slot(
                    id=s["id"],
                    area_id=s["area_id"],
                    slot_number=s["slot_number"],
                    slot_type=s["slot_type"],
                    state="FREE",
                    bike_capacity=s.get("bike_capacity", 2),
                    current_bikes=0,
                    has_car=False,
                    distance_from_gate=s["distance_from_gate"]
                )
                db.add(slot_obj)
            else:
                existing.area_id = s["area_id"]
                existing.slot_number = s["slot_number"]
                existing.slot_type = s["slot_type"]
                existing.distance_from_gate = s["distance_from_gate"]
        db.commit()
        print(f"[Database] Synchronized {len(slots_data)} parking slots in database.")
    finally:
        db.close()
