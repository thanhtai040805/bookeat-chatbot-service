# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from app.models import MessageRequest, MessageResponse
from app.agents.restaurant_agent import handle_message

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("", response_model=MessageResponse)
async def chat_endpoint(payload: MessageRequest):
    """
    AI Chat endpoint - nhận message từ Spring Boot và trả về AI response
    """
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    if not payload.userId.strip():
        raise HTTPException(status_code=400, detail="UserId is required.")
    
    return await handle_message(payload)