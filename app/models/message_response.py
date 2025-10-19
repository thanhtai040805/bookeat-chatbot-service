# app/models.py
from pydantic import BaseModel
from typing import Optional

class MessageResponse(BaseModel):
    response: str  # ← Chỉ cần field này