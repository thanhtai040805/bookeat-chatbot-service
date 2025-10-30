# app/services/spring_api_client.py
import requests
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class SpringAPIClient:
    """Enhanced Spring API Client với tất cả APIs mới"""
    
    def __init__(self):
        self.base_url = settings.SPRING_API_URL
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RestaurantChatbot/1.0'
        }
        self.timeout = 10
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Helper method để gọi API với error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            logger.info(f"Đang gọi API: {method} {url}")
            
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                timeout=self.timeout,
                **kwargs
            )
            
            if response.ok:
                return response.json()
            else:
                logger.error(f"API trả về status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"API timeout - {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Lỗi kết nối - {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Lỗi không xác định - {endpoint}: {e}")
            return None
    
    # ==================== RESTAURANT APIs ====================
    
    async def get_all_restaurants(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả nhà hàng"""
        result = self._make_request('GET', '/api/booking/restaurants')
        return result if result else []
    
    async def get_restaurant_details(self, restaurant_id: int) -> Optional[Dict[str, Any]]:
        """Lấy chi tiết nhà hàng"""
        return self._make_request('GET', f'/api/booking/restaurants/{restaurant_id}')
    
    async def get_restaurant_menu(self, restaurant_id: int) -> List[Dict[str, Any]]:
        """Lấy menu nhà hàng"""
        result = self._make_request('GET', f'/api/booking/restaurants/{restaurant_id}/dishes')
        return result if result else []
    
    async def get_restaurant_services(self, restaurant_id: int) -> List[Dict[str, Any]]:
        """Lấy dịch vụ nhà hàng"""
        result = self._make_request('GET', f'/api/booking/restaurants/{restaurant_id}/services')
        return result if result else []
    
    async def get_restaurant_tables(self, restaurant_id: int) -> List[Dict[str, Any]]:
        """Lấy danh sách bàn nhà hàng"""
        result = self._make_request('GET', f'/api/booking/restaurants/{restaurant_id}/tables')
        return result if result else []
    
    async def get_table_layouts(self, restaurant_id: int) -> List[Dict[str, Any]]:
        """Lấy table layouts của nhà hàng"""
        result = self._make_request('GET', f'/api/booking/restaurants/{restaurant_id}/table-layouts')
        return result if result else []
    
    # ==================== BOOKING APIs ====================
    
    async def check_availability(self, restaurant_id: int, booking_time: str, guest_count: int, selected_table_ids: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """Kiểm tra bàn trống"""
        params = {
            'restaurantId': restaurant_id,
            'bookingTime': booking_time,
            'guestCount': guest_count
        }
        if selected_table_ids:
            params['selectedTableIds'] = ','.join(map(str, selected_table_ids))
        
        return self._make_request('GET', '/api/booking/availability-check', params=params)
    
    async def get_available_time_slots(self, table_id: int, date: str) -> Optional[Dict[str, Any]]:
        """Lấy danh sách time slots khả dụng cho một bàn"""
        params = {'date': date}
        return self._make_request('GET', f'/api/booking/conflicts/available-slots/{table_id}', params=params)
    
    # ==================== PUBLIC APIs ONLY ====================
    # Chỉ giữ lại các API public không cần authentication
    
    async def get_demo_vouchers(self, restaurant_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lấy demo vouchers (public API)"""
        params = {}
        if restaurant_id:
            params['restaurantId'] = restaurant_id
        
        result = self._make_request('GET', '/api/vouchers/demo', params=params)
        return result if result else []
    
    # ==================== LEGACY METHODS (Backward Compatibility) ====================
    
    def get_restaurants(self) -> List[Dict[str, Any]]:
        """Legacy method - backward compatibility"""
        import asyncio
        return asyncio.run(self.get_all_restaurants())
    
    def test_spring_api(self) -> bool:
        """Test kết nối đến Spring API"""
        try:
            result = self._make_request('GET', '/health')
            if result is not None:
                logger.info("Spring API health check thành công")
                return True
            else:
                logger.warning("Spring API health check failed")
                return False
        except Exception as e:
            logger.error(f"Spring API health check failed: {e}")
            return False


# ==================== INSTANCE CREATION ====================

# Tạo instance global để sử dụng
spring_api_client = SpringAPIClient()

# Legacy functions để backward compatibility
def get_restaurants() -> List[Dict[str, Any]]:
    return spring_api_client.get_restaurants()

def test_spring_api() -> bool:
    return spring_api_client.test_spring_api()