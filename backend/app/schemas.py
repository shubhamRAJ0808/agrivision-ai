from pydantic import BaseModel, EmailStr
from typing import Optional
import datetime


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    farm_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    farm_name: Optional[str] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ScanHistoryOut(BaseModel):
    id: int
    module: str
    result_label: str
    confidence: str
    severity: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True
