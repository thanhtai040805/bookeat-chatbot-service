import logging
from typing import Dict, List, Optional, Any
from app.services.spring_api_client import spring_api_client

logger = logging.getLogger(__name__)

class FunctionService:
    """Service để execute functions và trả về formatted responses"""
    
    def __init__(self):
        self.spring_client = spring_api_client
        
    async def execute_function(self, function_name: str, arguments: Dict[str, Any], user_id: str = None) -> str:
        """
        Execute function và trả về formatted response
        
        Args:
            function_name: Tên function cần execute
            arguments: Arguments cho function
            user_id: ID của user (cho authentication)
            
        Returns:
            Formatted response string
        """
        try:
            logger.info(f"Executing function: {function_name} with arguments: {arguments}")
            
            if function_name == "search_restaurants":
                return await self._search_restaurants(arguments)
            
            elif function_name == "get_restaurant_menu":
                return await self._get_restaurant_menu(arguments)
            
            elif function_name == "get_demo_vouchers":
                return await self._get_demo_vouchers(arguments)
            
            elif function_name == "get_messages":
                return await self._get_messages(arguments, user_id)
            
            elif function_name == "get_table_layouts":
                return await self._get_table_layouts(arguments)
            
            elif function_name == "get_restaurant_details":
                return await self._get_restaurant_details(arguments)
            
            # AI Actions - chỉ tạo responses, không gọi API
            
            
            else:
                return f"Xin lỗi, tôi chưa thể thực hiện chức năng '{function_name}'. Vui lòng thử lại sau."
                
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
    
    async def _search_restaurants(self, arguments: Dict[str, Any]) -> str:
        """Search restaurants với filters"""
        try:
            restaurants = await self.spring_client.get_all_restaurants()
            
            if not restaurants:
                return "Xin lỗi, không tìm thấy nhà hàng nào. Vui lòng thử lại sau."
            
            # Filter by cuisine if specified
            if arguments.get("cuisine_type"):
                cuisine_filter = arguments["cuisine_type"].lower()
                restaurants = [
                    r for r in restaurants 
                    if cuisine_filter in r.get("cuisineType", "").lower()
                ]
            
            # Filter by location if specified
            if arguments.get("location"):
                location_filter = arguments["location"].lower()
                restaurants = [
                    r for r in restaurants 
                    if location_filter in r.get("address", "").lower()
                ]
            
            if not restaurants:
                return f"Xin lỗi, không tìm thấy nhà hàng phù hợp với tiêu chí của bạn. Bạn có thể thử tìm kiếm với tiêu chí khác."
            
            # Format response
            response = f"🍽️ **Tìm thấy {len(restaurants)} nhà hàng phù hợp:**\n\n"
            
            for i, restaurant in enumerate(restaurants[:5], 1):  # Show top 5
                response += f"**{i}. {restaurant.get('name', 'N/A')}**\n"
                response += f"📍 {restaurant.get('address', 'N/A')}\n"
                response += f"🍽️ {restaurant.get('cuisineType', 'N/A')}\n"
                response += f"⭐ {restaurant.get('rating', 'N/A')}/5\n"
                response += f"💰 {restaurant.get('priceRange', 'N/A')}\n\n"
            
            if len(restaurants) > 5:
                response += f"... và {len(restaurants) - 5} nhà hàng khác.\n\n"
            
            response += "Bạn muốn xem chi tiết nhà hàng nào hoặc cần hỗ trợ gì thêm?"
            return response
            
        except Exception as e:
            logger.error(f"Error in _search_restaurants: {e}")
            return "Xin lỗi, không thể tìm kiếm nhà hàng. Vui lòng thử lại sau."
    
    async def _get_restaurant_menu(self, arguments: Dict[str, Any]) -> str:
        """Get restaurant menu"""
        try:
            restaurant_id = arguments.get("restaurant_id", 1)  # Default to restaurant 1
            menu = await self.spring_client.get_restaurant_menu(restaurant_id)
            
            if not menu:
                return "Xin lỗi, không thể lấy được thực đơn. Vui lòng thử lại sau."
            
            response = f"🍽️ **Thực đơn nhà hàng:**\n\n"
            
            for dish in menu[:10]:  # Show top 10 dishes
                response += f"• **{dish.get('name', 'N/A')}** - {dish.get('price', 'N/A')} VNĐ\n"
                if dish.get('description'):
                    response += f"  _{dish.get('description', '')}_\n"
                response += "\n"
            
            if len(menu) > 10:
                response += f"... và {len(menu) - 10} món khác.\n\n"
            
            response += "Bạn có muốn đặt bàn hoặc cần thông tin gì khác không?"
            return response
            
        except Exception as e:
            logger.error(f"Error in _get_restaurant_menu: {e}")
            return "Xin lỗi, không thể lấy thực đơn. Vui lòng thử lại sau."
    
    
    async def _get_demo_vouchers(self, arguments: Dict[str, Any]) -> str:
        """Get demo vouchers"""
        try:
            restaurant_id = arguments.get("restaurant_id")
            vouchers = await self.spring_client.get_demo_vouchers(restaurant_id)
            
            if not vouchers:
                return "Hiện tại không có voucher nào khả dụng."
            
            response = f"🎫 **Vouchers khả dụng:**\n\n"
            
            for voucher in vouchers[:5]:
                response += f"• **{voucher.get('name', 'N/A')}**\n"
                response += f"  💰 Giảm: {voucher.get('discountAmount', 'N/A')} VNĐ\n"
                if voucher.get('description'):
                    response += f"  📝 {voucher.get('description', '')}\n"
                response += "\n"
            
            if len(vouchers) > 5:
                response += f"... và {len(vouchers) - 5} voucher khác.\n\n"
            
            response += "Bạn có muốn sử dụng voucher nào không?"
            return response
            
        except Exception as e:
            logger.error(f"Error in _get_demo_vouchers: {e}")
            return "Xin lỗi, không thể lấy thông tin voucher. Vui lòng thử lại sau."
    
    async def _get_messages(self, arguments: Dict[str, Any], user_id: str) -> str:
        """Get chat messages"""
        try:
            room_id = arguments.get("room_id", "default")
            messages = await self.spring_client.get_messages(room_id, user_id or "user123")
            
            if not messages:
                return "Không có tin nhắn nào trong lịch sử chat."
            
            response = f"💬 **Lịch sử chat gần đây:**\n\n"
            
            for message in messages[-5:]:  # Show last 5 messages
                sender = "Bạn" if message.get("sender") == user_id else "AI Assistant"
                response += f"**{sender}:** {message.get('content', 'N/A')}\n"
                response += f"🕐 {message.get('timestamp', 'N/A')}\n\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in _get_messages: {e}")
            return "Xin lỗi, không thể lấy lịch sử chat. Vui lòng thử lại sau."
    
    async def _get_table_layouts(self, arguments: Dict[str, Any]) -> str:
        """Get table layouts"""
        try:
            restaurant_id = arguments.get("restaurant_id", 1)
            layouts = await self.spring_client.get_table_layouts(restaurant_id)
            
            if not layouts:
                return "Không thể lấy được sơ đồ bàn. Vui lòng thử lại sau."
            
            response = f"🗺️ **Sơ đồ bàn nhà hàng:**\n\n"
            response += "📋 **Các loại bàn có sẵn:**\n"
            
            for layout in layouts:
                response += f"• {layout.get('tableType', 'N/A')}: {layout.get('capacity', 'N/A')} người\n"
            
            response += "\nBạn có muốn đặt bàn hoặc cần thông tin gì khác không?"
            return response
            
        except Exception as e:
            logger.error(f"Error in _get_table_layouts: {e}")
            return "Xin lỗi, không thể lấy sơ đồ bàn. Vui lòng thử lại sau."
    
    async def _get_restaurant_details(self, arguments: Dict[str, Any]) -> str:
        """Get restaurant details"""
        try:
            restaurant_id = arguments.get("restaurant_id", 1)
            details = await self.spring_client.get_restaurant_details(restaurant_id)
            
            if not details:
                return "Không thể lấy thông tin nhà hàng. Vui lòng thử lại sau."
            
            response = f"🏪 **Thông tin nhà hàng:**\n\n"
            response += f"**Tên:** {details.get('name', 'N/A')}\n"
            response += f"**Địa chỉ:** {details.get('address', 'N/A')}\n"
            response += f"**Loại ẩm thực:** {details.get('cuisineType', 'N/A')}\n"
            response += f"**Đánh giá:** {details.get('rating', 'N/A')}/5\n"
            response += f"**Khoảng giá:** {details.get('priceRange', 'N/A')}\n"
            
            if details.get('description'):
                response += f"**Mô tả:** {details.get('description')}\n"
            
            if details.get('openingHours'):
                response += f"**Giờ mở cửa:** {details.get('openingHours')}\n"
            
            response += "\nBạn có muốn xem menu hoặc đặt bàn không?"
            return response
            
        except Exception as e:
            logger.error(f"Error in _get_restaurant_details: {e}")
            return "Xin lỗi, không thể lấy thông tin nhà hàng. Vui lòng thử lại sau."
    
    # ==================== AI ACTIONS RESPONSE TEMPLATES ====================
    
    
    
    # Không cần response template nữa
