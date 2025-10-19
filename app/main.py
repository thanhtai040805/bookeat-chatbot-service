from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import load_environment
from app.routers import chat, health

# Load configuration once so environment-specific settings are ready.
_settings = load_environment()

app = FastAPI(
    title="Restaurant Chatbot Service",
    version="2.0.0",
    description="Conversational AI agent for restaurant support.",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers.
app.include_router(chat.router)
app.include_router(health.router)


@app.get("/")
async def root():
    return {"message": "Restaurant chatbot service is running.", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
