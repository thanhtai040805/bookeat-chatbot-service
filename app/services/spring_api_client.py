import requests
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_restaurants() -> List[Dict[str, Any]]:
    """
    Lấy danh sách nhà hàng từ Spring API.
    
    Returns:
        List các nhà hàng hoặc empty list nếu có lỗi
    """
    try:
        url = f"{settings.SPRING_API_URL}/restaurants"
        logger.info(f"Đang gọi API: {url}")
        
        # Thêm timeout và headers
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RestaurantChatbot/1.0'
        }
        
        resp = requests.get(
            url, 
            headers=headers,
            timeout=10  # 10 giây timeout
        )
        
        if resp.ok:
            data = resp.json()
            logger.info(f"Lấy được {len(data)} nhà hàng từ API")
            return data
        else:
            logger.error(f"API trả về status {resp.status_code}: {resp.text}")
            return []
            
    except requests.exceptions.Timeout:
        logger.error("API timeout - không thể kết nối đến Spring API")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Lỗi kết nối đến Spring API - có thể service đang down")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi request đến Spring API: {e}")
        return []
    except Exception as e:
        logger.error(f"Lỗi không xác định khi gọi Spring API: {e}")
        return []

def test_spring_api() -> bool:
    """
    Test kết nối đến Spring API.
    
    Returns:
        True nếu API hoạt động, False nếu có lỗi
    """
    try:
        url = f"{settings.SPRING_API_URL}/health"  # Giả sử có health endpoint
        resp = requests.get(url, timeout=5)
        if resp.ok:
            logger.info("Spring API health check thành công")
            return True
        else:
            logger.warning(f"Spring API health check failed: {resp.status_code}")
            return False
    except Exception as e:
        logger.error(f"Spring API health check failed: {e}")
        return False