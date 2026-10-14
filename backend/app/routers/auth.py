from fastapi import APIRouter, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session

from ..models.database import SessionLocal, User
from ..schemas.parking import UserRegisterRequest, UserResponse

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Users"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register", response_model=UserResponse)
def register_user(req: UserRegisterRequest, db: Session = Depends(get_db)):
    role = req.role.upper()
    v_type = req.vehicle_type.upper()
    
    if role not in ("STUDENT", "TEACHER", "VIP"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be STUDENT, TEACHER, or VIP.")
    if v_type not in ("TWO_WHEELER", "FOUR_WHEELER"):
        raise HTTPException(status_code=400, detail="Invalid vehicle type. Must be TWO_WHEELER or FOUR_WHEELER.")
    if role == "STUDENT" and v_type == "FOUR_WHEELER":
        raise HTTPException(status_code=400, detail="Policy violation: Students are only permitted 2-wheelers.")

    # Check if vehicle number already registered - update role/vehicle type if changed
    existing = db.query(User).filter(User.vehicle_number == req.vehicle_number.strip().upper()).first()
    if existing:
        existing.role = role
        existing.vehicle_type = v_type
        existing.name = req.name.strip()
        db.commit()
        db.refresh(existing)
        return existing

    new_user = User(
        name=req.name.strip(),
        role=role,
        vehicle_number=req.vehicle_number.strip().upper(),
        vehicle_type=v_type
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.get("/users", response_model=List[UserResponse])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).limit(20).all()

@router.get("/user/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
