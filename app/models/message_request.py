# app/models.py
from pydantic import BaseModel
from typing import Optional

class MessageRequest(BaseModel):
    message: str
    userId: str
    timestamp: int  # ← Bắt buộc, không optional