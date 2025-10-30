from fastapi import APIRouter, HTTPException
from app.agents.restaurant_agent import restaurant_agent

router = APIRouter(prefix="/vector", tags=["Vector Database"])

@router.post("/initialize")
async def initialize_vector_database():
    """
    Initialize Vector Database với restaurant data từ Spring API
    """
    try:
        await restaurant_agent.initialize_vector_database()
        return {
            "status": "success",
            "message": "Vector Database initialized successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Vector Database: {str(e)}"
        )

@router.get("/stats")
async def get_vector_database_stats():
    """
    Get statistics về Vector Database
    """
    try:
        stats = await restaurant_agent.get_vector_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Vector Database stats: {str(e)}"
        )

@router.get("/health")
async def vector_database_health():
    """
    Health check cho Vector Database
    """
    try:
        stats = await restaurant_agent.get_vector_database_stats()
        return {
            "status": "healthy" if stats.get("status") == "healthy" else "unhealthy",
            "message": "Vector Database is operational" if stats.get("status") == "healthy" else "Vector Database has issues",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Vector Database error: {str(e)}"
        }
