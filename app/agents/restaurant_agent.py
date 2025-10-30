import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from app.models import MessageRequest, MessageResponse
from app.services.vector_intent_service import vector_intent_service
from app.services.function_service import FunctionService
from app.services.vector_service import vector_service
from app.core.config import settings
from openai import OpenAI

logger = logging.getLogger("restaurant_agent")

class RestaurantAgent:
    """Anti-Hallucination Restaurant Agent với Vector DB First + Multi-Collection Search"""
    
    def __init__(self):
        self.intent_service = vector_intent_service
        self.function_service = FunctionService()
        self.vector_service = vector_service
        self.openai_client = None
        self.conversations = {}
        # TurnState memory để track context
        self.turn_states = {}  # {user_id: {"last_restaurant_id": ..., "last_restaurant_name": ...}}
        
        # STRICT System Prompt - Ngăn hallucination
        self.strict_system_prompt = """You are RestaurantBot, a friendly AI concierge for restaurant booking system.

CRITICAL RULES - NEVER VIOLATE:
1. ONLY mention restaurants/dishes/services that are EXPLICITLY listed in the provided data
2. NEVER create or invent restaurant names, dish names, or any information
3. If data is not available, say "Không tìm thấy thông tin" - DO NOT make up information
4. Always verify information exists in provided data before mentioning it

Your capabilities:
- Provide accurate information based on REAL DATA provided
- Format responses naturally and friendly
- Use Vietnamese by default, English if user uses English
- Keep responses concise but informative"""

    async def handle_message(self, payload: MessageRequest) -> MessageResponse:
        """Anti-Hallucination message handling với Vector DB First + Multi-Collection Search"""
        try:
            conversation_id = payload.userId
            user_id = payload.userId
            
            # 0. Initialize turn state nếu chưa có
            if user_id not in self.turn_states:
                self.turn_states[user_id] = {
                    "last_restaurant_id": None,
                    "last_restaurant_name": None,
                    "last_intent": None
                }
            
            # 1. Intent Recognition với turn state context
            intent_result = await self.intent_service.recognize_intent_with_context(
                payload.message, payload.userId
            )
            entities = await self.intent_service.extract_entities_with_context(
                payload.message, intent_result["intent"], intent_result.get("context", "")
            )
            
            # 1.1. Resolve restaurant reference từ turn state
            entities = await self._resolve_restaurant_reference(payload.message, entities, user_id)
            
            logger.info(f"Intent: {intent_result['intent']}, Confidence: {intent_result['confidence']}")
            logger.info(f"Entities: {entities}")
            
            # 2. Check for complex queries (restaurant search + availability)
            if self._is_complex_availability_query(payload.message, intent_result["intent"]):
                response = await self._handle_complex_availability_query(
                    payload.message, entities, payload.userId
                )
                response_success = True
            
            # 3. Data Retrieval Strategy - Vector DB First
            elif intent_result["intent"] == "restaurant_search":
                # VECTOR DB FIRST - Multi-collection search
                response = await self._handle_restaurant_search(
                    payload.message, entities, payload.userId
                )
                response_success = True
                
            elif intent_result["intent"] == "menu_inquiry":
                # VECTOR DB FIRST - Multi-collection search
                response = await self._handle_menu_inquiry(
                    payload.message, entities, payload.userId
                )
                response_success = True
                
                
            else:
                # Check for booking/waitlist requests and redirect
                if ("đặt bàn" in payload.message.lower() or "booking" in payload.message.lower() or 
                    "reservation" in payload.message.lower() or "waitlist" in payload.message.lower() or 
                    "xếp hàng" in payload.message.lower() or "chờ bàn" in payload.message.lower()):
                    response = "Tôi có thể giúp bạn tìm nhà hàng phù hợp và kiểm tra bàn trống. Để đặt bàn hoặc tham gia waitlist, vui lòng truy cập trang đặt bàn của chúng tôi hoặc liên hệ trực tiếp với nhà hàng. Bạn muốn tôi tìm nhà hàng nào cho bạn?"
                    response_success = True
                else:
                    # General inquiry - Try Vector DB first, then OpenAI với strict context
                    response = await self._handle_general_inquiry(
                        payload.message, payload.userId
                    )
                    response_success = True
            
            # 3. Store conversation
            await self.vector_service.store_conversation(
                payload.userId, payload.message, response, intent_result["intent"]
            )
            
            # 4. Update turn state với restaurant đã gợi ý
            await self._update_turn_state(user_id, intent_result["intent"], entities, response)
            
            # 5. Learn from interaction
            await self.intent_service.learn_from_interaction(
                payload.userId, payload.message, intent_result["intent"], 
                entities, response_success
            )
            
            # 6. Store in memory
            self._store_conversation(conversation_id, payload.message, response)
            
            return MessageResponse(response=response)
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return MessageResponse(response="Xin lỗi, có lỗi xảy ra khi xử lý tin nhắn của bạn.")
    
    async def _resolve_restaurant_reference(self, user_message: str, entities: Dict, user_id: str) -> Dict:
        """Resolve restaurant reference từ turn state"""
        try:
            reference_keywords = [
                "nhà hàng đó", "nhà hàng bạn gợi ý", "nhà hàng phía trên", 
                "nhà hàng vừa nói", "nhà hàng trước đó", "restaurant đó",
                "nhà hàng kia", "nhà hàng vừa rồi", "nhà hàng trên"
            ]
            
            if not any(kw in user_message.lower() for kw in reference_keywords):
                return entities
            
            turn_state = self.turn_states.get(user_id, {})
            last_restaurant_id = turn_state.get("last_restaurant_id")
            last_restaurant_name = turn_state.get("last_restaurant_name")
            
            if last_restaurant_id:
                entities["restaurant_id"] = last_restaurant_id
                logger.info(f"Resolved restaurant reference: ID={last_restaurant_id}, Name={last_restaurant_name}")
            else:
                logger.warning(f"No restaurant reference found for user {user_id}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error resolving restaurant reference: {e}")
            return entities
    
    async def _update_turn_state(self, user_id: str, intent: str, entities: Dict, response: str):
        """Update turn state với restaurant đã gợi ý"""
        try:
            if user_id not in self.turn_states:
                self.turn_states[user_id] = {
                    "last_restaurant_id": None,
                    "last_restaurant_name": None,
                    "last_intent": None
                }
            
            # Update last intent
            self.turn_states[user_id]["last_intent"] = intent
            
            # Extract restaurant info từ response nếu có
            if intent in ["restaurant_search", "menu_inquiry"] and entities.get("restaurant_id"):
                self.turn_states[user_id]["last_restaurant_id"] = entities["restaurant_id"]
                
                # Extract restaurant name từ response
                restaurant_name = self._extract_restaurant_name_from_response(response)
                if restaurant_name:
                    self.turn_states[user_id]["last_restaurant_name"] = restaurant_name
                    logger.info(f"Updated turn state for user {user_id}: restaurant_id={entities['restaurant_id']}, name={restaurant_name}")
            
        except Exception as e:
            logger.error(f"Error updating turn state: {e}")
    
    def _extract_restaurant_name_from_response(self, response: str) -> Optional[str]:
        """Extract restaurant name từ AI response"""
        import re
        
        # Patterns để tìm restaurant name
        patterns = [
            r"nhà hàng\s+([A-Za-z\s]+?)(?:\s|,|\.|$|chuyên)",
            r"tại\s+([A-Za-z\s]+?)(?:\s|,|\.|$|chuyên)",
            r"([A-Za-z\s]+?(?:BBQ|Premium|Restaurant|Restaurants))(?:\s|,|\.|$|chuyên)",
            r"NHÀ HÀNG\s*-\s*Tên:\s*([A-Za-z\s]+?)(?:\s|,|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common words
                if len(name) > 3 and name.lower() not in ['có', 'là', 'nào', 'và', 'cho']:
                    return name
        
        return None
    
    def _is_complex_availability_query(self, user_message: str, intent: str) -> bool:
        """Detect complex queries that combine restaurant search + availability check"""
        message_lower = user_message.lower()
        
        # Keywords that indicate both restaurant search and availability
        restaurant_keywords = ['nhà hàng', 'restaurant', 'quán', 'chỗ ăn']
        availability_keywords = ['có bàn', 'bàn trống', 'availability', 'còn chỗ', 'đặt được']
        cuisine_keywords = ['nhật', 'japanese', 'hàn', 'korean', 'ý', 'italian', 'việt', 'vietnamese']
        time_keywords = ['ngày mai', 'hôm nay', 'tối nay', 'trưa nay', 'chiều nay']
        
        has_restaurant = any(kw in message_lower for kw in restaurant_keywords)
        has_availability = any(kw in message_lower for kw in availability_keywords)
        has_cuisine = any(kw in message_lower for kw in cuisine_keywords)
        has_time = any(kw in message_lower for kw in time_keywords)
        
        # Complex query if it has restaurant + availability + (cuisine or time)
        is_complex = has_restaurant and has_availability and (has_cuisine or has_time)
        
        logger.info(f"Complex query detection: restaurant={has_restaurant}, availability={has_availability}, cuisine={has_cuisine}, time={has_time}, is_complex={is_complex}")
        return is_complex
    
    async def _handle_complex_availability_query(self, user_message: str, entities: Dict, user_id: str) -> str:
        """Handle complex queries that combine restaurant search + availability check"""
        try:
            logger.info(f"Handling complex availability query: {user_message}")
            
            # 1. First, search for restaurants matching the criteria
            collections = await self._detect_required_collections(user_message, {"intent": "restaurant_search"})
            
            search_results = await self._multi_collection_search(
                user_message,
                collections,
                distance_threshold=0.6,  # Higher threshold for complex queries
                limit_per_collection=30
            )
            
            aggregated = await self._aggregate_search_results(search_results)
            restaurants_enriched = aggregated.get("restaurants", [])
            
            if not restaurants_enriched:
                return "Xin lỗi, tôi không tìm thấy nhà hàng nào phù hợp với tiêu chí của bạn."
            
            # 2. Check availability for each restaurant
            availability_results = []
            
            for restaurant in restaurants_enriched[:5]:  # Limit to top 5 restaurants
                restaurant_id = self._extract_restaurant_id_from_metadata(restaurant)
                if not restaurant_id:
                    continue
                
                # Normalize entities and extract booking_time coherently
                normalized_entities = self._normalize_booking_entities(entities, user_message)
                booking_time = normalized_entities.get("booking_time", self._extract_booking_time_from_message(user_message))
                
                # Check availability
                try:
                    availability_result = await self.function_service.execute_function(
                        "check_availability",
                        {
                            "restaurant_id": restaurant_id,
                            "booking_time": booking_time,
                            "guest_count": entities.get("guest_count", 2)
                        },
                        user_id
                    )
                    
                    # Parse availability result
                    has_availability = "có bàn trống" in availability_result.lower() or "available" in availability_result.lower()
                    
                    availability_results.append({
                        "restaurant": restaurant,
                        "has_availability": has_availability,
                        "availability_text": availability_result
                    })
                    
                except Exception as e:
                    logger.error(f"Error checking availability for restaurant {restaurant_id}: {e}")
                    availability_results.append({
                        "restaurant": restaurant,
                        "has_availability": False,
                        "availability_text": "Không thể kiểm tra khả dụng"
                    })
            
            # 3. Format response
            available_restaurants = [r for r in availability_results if r["has_availability"]]
            unavailable_restaurants = [r for r in availability_results if not r["has_availability"]]
            
            if available_restaurants:
                response_parts = [f"🎉 **Tìm thấy {len(available_restaurants)} nhà hàng có bàn trống:**\n"]
                
                for i, result in enumerate(available_restaurants, 1):
                    restaurant = result["restaurant"]
                    response_parts.append(
                        f"**{i}. {restaurant.get('restaurantName', restaurant.get('name', 'N/A'))}**\n"
                        f"📍 {restaurant.get('address', 'N/A')}\n"
                        f"🍽️ {restaurant.get('cuisineType', 'N/A')}\n"
                        f"✅ {result['availability_text']}\n"
                    )
                
                if unavailable_restaurants:
                    response_parts.append(f"\n❌ **{len(unavailable_restaurants)} nhà hàng khác không có bàn trống**")
                
                return "\n".join(response_parts)
            
            else:
                response_parts = ["😔 **Tất cả nhà hàng phù hợp đều không có bàn trống:**\n"]
                
                for i, result in enumerate(availability_results, 1):
                    restaurant = result["restaurant"]
                    response_parts.append(
                        f"**{i}. {restaurant.get('restaurantName', restaurant.get('name', 'N/A'))}**\n"
                        f"📍 {restaurant.get('address', 'N/A')}\n"
                        f"❌ {result['availability_text']}\n"
                    )
                
                response_parts.append("\n💡 **Gợi ý:** Bạn có thể thử thời gian khác hoặc liên hệ trực tiếp với nhà hàng.")
                return "\n".join(response_parts)
                
        except Exception as e:
            logger.error(f"Error handling complex availability query: {e}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu phức tạp của bạn."
    
    def _extract_booking_time_from_message(self, user_message: str) -> str:
        """Extract booking time from user message"""
        from datetime import datetime, timedelta
        
        message_lower = user_message.lower()
        current_date = datetime.now()
        tomorrow_date = current_date + timedelta(days=1)
        
        # Default to tomorrow 19:00
        default_time = tomorrow_date.strftime("%Y-%m-%d 19:00")
        
        # Check for "ngày mai"
        if "ngày mai" in message_lower:
            if "12 giờ trưa" in message_lower or "12h trưa" in message_lower:
                return tomorrow_date.strftime("%Y-%m-%d 12:00")
            elif "tối" in message_lower:
                return tomorrow_date.strftime("%Y-%m-%d 19:00")
            elif "sáng" in message_lower:
                return tomorrow_date.strftime("%Y-%m-%d 08:00")
            else:
                return tomorrow_date.strftime("%Y-%m-%d 19:00")
        
        # Check for "hôm nay"
        elif "hôm nay" in message_lower:
            if "12 giờ trưa" in message_lower or "12h trưa" in message_lower:
                return current_date.strftime("%Y-%m-%d 12:00")
            elif "tối" in message_lower:
                return current_date.strftime("%Y-%m-%d 19:00")
            else:
                return current_date.strftime("%Y-%m-%d 19:00")
        
        return default_time

    def _normalize_booking_entities(self, entities: Dict, user_message: str) -> Dict:
        """Normalize date/time + compose booking_time consistently from entities and raw text."""
        from datetime import datetime, timedelta
        try:
            normalized = dict(entities) if entities else {}

            # Parse base date
            now = datetime.now()
            base_date = None

            date_value = normalized.get("date")
            if isinstance(date_value, str):
                lower = date_value.lower()
                if "mai" in lower or "tomorrow" in lower:
                    base_date = now + timedelta(days=1)
                elif "hôm nay" in lower or "today" in lower:
                    base_date = now
                else:
                    # Try YYYY-MM-DD
                    try:
                        base_date = datetime.strptime(date_value[:10], "%Y-%m-%d")
                    except Exception:
                        base_date = None

            # Derive hour/minute
            hour = None
            minute = 0

            # If booking_time exists, try extract time portion
            bt = normalized.get("booking_time")
            if isinstance(bt, str):
                # Try parse time part HH:MM
                try:
                    if len(bt) >= 16:
                        hour = int(bt[11:13])
                        minute = int(bt[14:16])
                    elif len(bt) >= 5 and ":" in bt:
                        hour = int(bt.split(":")[0][-2:])
                        minute = int(bt.split(":")[1][:2])
                except Exception:
                    hour = None

            # Infer from raw user text if needed
            text = (user_message or "").lower()
            if hour is None:
                if "12 giờ trưa" in text or "12h trưa" in text or "12 gio trua" in text:
                    hour = 12
                elif "2h" in text or "2 giờ" in text:
                    hour = 14
                elif "sáng" in text or "morning" in text:
                    hour = 8
                elif "tối" in text or "evening" in text or "night" in text:
                    hour = 19
                else:
                    hour = 19  # default

            if base_date is None:
                # Fallback: if we can't determine date, use helper that considers text
                normalized["booking_time"] = self._extract_booking_time_from_message(user_message)
                return normalized

            # Compose final booking_time from base_date + time
            final_dt = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            normalized["booking_time"] = final_dt.strftime("%Y-%m-%d %H:%M")
            return normalized
        except Exception:
            return entities
    
    async def _detect_required_collections(self, user_message: str, intent_result: Dict) -> List[str]:
        """
        Detect các collections cần query dựa trên user message
        
        Returns:
            List of collection names: ['restaurants', 'menus', 'tables', 'image_url', ...]
        """
        collections = []
        message_lower = user_message.lower()
        
        # Detect keywords
        restaurant_keywords = ['nhà hàng', 'restaurant', 'quán', 'địa chỉ', 'cuisine', 'ẩm thực']
        menu_keywords = ['menu', 'thực đơn', 'món', 'dish', 'food', 'ăn gì', 'đồ ăn']
        service_keywords = ['dịch vụ', 'service', 'phục vụ', 'tiện ích', 'tổ chức tiệc', 'sinh nhật', 'birthday', 'party', 'event', 'sự kiện']
        table_keywords = ['bàn', 'table', 'availability', 'available', 'chỗ ngồi', 'seatcount', 'sức chứa', 'loại bàn']
        layout_keywords = ['sơ đồ', 'layout', 'ảnh', 'image', 'hình ảnh', 'phòng', 'room', 'không gian', 'vị trí', 'location']
        
        # Check intent first
        intent = intent_result.get("intent", "")
        
        if intent == "restaurant_search":
            collections.append("restaurants")
            # Nếu message có menu keywords → thêm menus
            if any(kw in message_lower for kw in menu_keywords):
                collections.append("menus")
            # Nếu message có service keywords → thêm menus (services stored in menus collection)
            if any(kw in message_lower for kw in service_keywords):
                if "menus" not in collections:
                    collections.append("menus")
            # Nếu message có table keywords → thêm menus (tables stored in menus collection)
            if any(kw in message_lower for kw in table_keywords):
                if "menus" not in collections:
                    collections.append("menus")
            # Nếu message có layout keywords → thêm image_url
            if any(kw in message_lower for kw in layout_keywords):
                collections.append("image_url")
        
        elif intent == "menu_inquiry":
            collections.append("menus")
            # Nếu message có restaurant keywords → thêm restaurants
            if any(kw in message_lower for kw in restaurant_keywords):
                collections.append("restaurants")
            # Nếu message có table keywords → thêm menus tables
            if any(kw in message_lower for kw in table_keywords):
                pass  # tables đã trong menus
            # Nếu message có layout keywords → thêm image_url
            if any(kw in message_lower for kw in layout_keywords):
                collections.append("image_url")
        
        elif intent == "check_availability":
            collections.append("restaurants")
            if "menus" not in collections:
                collections.append("menus")  # tables nằm trong menus
            if any(kw in message_lower for kw in layout_keywords):
                collections.append("image_url")
        
        else:
            # General query - detect multi-intent
            if any(kw in message_lower for kw in restaurant_keywords):
                collections.append("restaurants")
            if any(kw in message_lower for kw in menu_keywords):
                collections.append("menus")
            if any(kw in message_lower for kw in service_keywords):
                if "menus" not in collections:
                    collections.append("menus")  # Services stored in menus collection
            if any(kw in message_lower for kw in table_keywords):
                if "menus" not in collections:
                    collections.append("menus")  # Tables stored in menus collection
            if any(kw in message_lower for kw in layout_keywords):
                collections.append("image_url")
        
        # Nếu không detect được → default to restaurants
        if not collections:
            collections.append("restaurants")
        
        logger.debug(f"Detected collections for query: {collections}")
        return list(set(collections))
    
    async def _multi_collection_search(
        self, 
        user_message: str, 
        collections: List[str],
        distance_threshold: float = 0.5,
        limit_per_collection: int = 5,
        restaurant_id: int = None
    ) -> Dict[str, List[Dict]]:
        """
        Search nhiều collections cùng lúc
        
        Returns:
            {
                "restaurants": [...],
                "menus": [...],
                "tables": [...],
                "image_url": [...]
            }
        """
        results = {}
        
        # Parallel search các collections
        search_tasks = []
        collection_names = []
        
        if "restaurants" in collections:
            search_tasks.append(
                self.vector_service.search_restaurants(
                    user_message, 
                    limit=limit_per_collection,
                    distance_threshold=distance_threshold
                )
            )
            collection_names.append("restaurants")
        
        if "menus" in collections:
            # ✅ FIX: Pass restaurant_id để filter menus theo nhà hàng cụ thể
            search_tasks.append(
                self.vector_service.search_menus(
                    user_message,
                    restaurant_id=restaurant_id,  # ✅ Filter by restaurant_id
                    limit=limit_per_collection * 3,  # Menus có thể nhiều hơn - tăng multiplier
                    distance_threshold=distance_threshold
                )
            )
            collection_names.append("menus")
        
        # Search tables (từ menus collection, nhưng filter type=table)
        if "menus" in collections:
            # ✅ FIX: Pass restaurant_id để filter tables theo nhà hàng cụ thể
            search_tasks.append(
                self.vector_service.search_tables(
                    user_message,
                    restaurant_id=restaurant_id,  # ✅ Filter by restaurant_id
                    limit=limit_per_collection,
                    distance_threshold=distance_threshold
                )
            )
            collection_names.append("tables")
        
        # Search table layouts/images
        if "image_url" in collections:
            # ✅ FIX: Pass restaurant_id để filter layouts theo nhà hàng cụ thể
            search_tasks.append(
                self.vector_service.search_table_layouts(
                    user_message,
                    restaurant_id=restaurant_id,  # ✅ Filter by restaurant_id
                    limit=limit_per_collection,
                    distance_threshold=distance_threshold
                )
            )
            collection_names.append("image_url")
        
        # Execute parallel
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Map results
            for idx, collection_name in enumerate(collection_names):
                if not isinstance(search_results[idx], Exception):
                    results[collection_name] = search_results[idx]
                else:
                    logger.error(f"Error searching {collection_name}: {search_results[idx]}")
                    results[collection_name] = []
        
        return results

    def _extract_restaurant_id_from_metadata(self, metadata: Dict[str, Any]) -> Optional[Any]:
        if not metadata:
            return None
        candidate_keys = [
            "restaurant_id",
            "restaurantId",
            "restaurantID",
            "id",
        ]
        for key in candidate_keys:
            if key in metadata and metadata[key] not in (None, ""):
                value = metadata[key]
                if isinstance(value, str) and value.isdigit():
                    try:
                        return int(value)
                    except Exception:
                        return value
                return value
        return None

    def _detect_collection_item_type(self, metadata: Dict[str, Any]) -> str:
        if not metadata:
            return "menu"
        lowered_keys = {key.lower() for key in metadata.keys()}
        service_indicators = {"serviceid", "servicename", "service_name", "servicecategory", "servicetype", "duration"}
        table_indicators = {"tableid", "tablename", "table_name", "capacity", "tabletype", "tablelayout"}
        if lowered_keys & service_indicators:
            return "service"
        if lowered_keys & table_indicators:
            return "table"
        return "menu"

    def _simplify_matched_item(self, metadata: Dict[str, Any], distance: float, item_type: str) -> Dict[str, Any]:
        meta_copy = dict(metadata or {})
        name = (
            meta_copy.get("name")
            or meta_copy.get("dishName")
            or meta_copy.get("serviceName")
            or meta_copy.get("tableName")
            or meta_copy.get("title")
            or "N/A"
        )
        description = (
            meta_copy.get("description")
            or meta_copy.get("serviceDescription")
            or meta_copy.get("details")
            or meta_copy.get("note")
        )
        price = meta_copy.get("price") or meta_copy.get("cost") or meta_copy.get("amount")
        return {
            "name": name,
            "description": description,
            "price": price,
            "distance": distance,
            "type": item_type,
            "metadata": meta_copy,
        }

    async def _aggregate_search_results(
        self,
        search_results: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        missing_restaurant_ids: set[Any] = set()

        restaurant_entries = search_results.get("restaurants", []) or []
        menu_entries = search_results.get("menus", []) or []
        table_entries = search_results.get("tables", []) or []
        image_entries = search_results.get("image_url", []) or []

        for entry in restaurant_entries:
            metadata = entry.get("metadata") or {}
            restaurant_id = self._extract_restaurant_id_from_metadata(metadata)
            if restaurant_id is None:
                continue
            key = str(restaurant_id)
            aggregator = aggregated.get(key)
            if not aggregator:
                aggregator = {
                    "restaurant_id": restaurant_id,
                    "restaurant": dict(metadata),
                    "matched_menus": [],
                    "matched_services": [],
                    "matched_tables": [],
                    "matched_images": [],
                    "score": 0.0,
                    "sources": set(["restaurants"]),
                }
                aggregated[key] = aggregator
            else:
                if not aggregator.get("restaurant"):
                    aggregator["restaurant"] = dict(metadata)
                aggregator["sources"].add("restaurants")
            aggregator["score"] += max(0.0, 1.0 - entry.get("distance", 1.0))

        for entry in menu_entries:
            metadata = entry.get("metadata") or {}
            restaurant_id = self._extract_restaurant_id_from_metadata(metadata)
            if restaurant_id is None:
                continue
            key = str(restaurant_id)
            aggregator = aggregated.get(key)
            if not aggregator:
                aggregator = {
                    "restaurant_id": restaurant_id,
                    "restaurant": None,
                    "matched_menus": [],
                    "matched_services": [],
                    "matched_tables": [],
                    "matched_images": [],
                    "score": 0.0,
                    "sources": set(),
                }
                aggregated[key] = aggregator
            if aggregator.get("restaurant") is None:
                missing_restaurant_ids.add(restaurant_id)

            item_type = self._detect_collection_item_type(metadata)
            simplified_item = self._simplify_matched_item(
                metadata, entry.get("distance", 1.0), item_type
            )
            if item_type == "service":
                aggregator["matched_services"].append(simplified_item)
                aggregator["sources"].add("services")
            elif item_type == "table":
                aggregator["matched_tables"].append(simplified_item)
                aggregator["sources"].add("tables")
            else:
                aggregator["matched_menus"].append(simplified_item)
                aggregator["sources"].add("menus")
            aggregator["score"] += max(0.0, 1.0 - entry.get("distance", 1.0))

        # Xử lý tables search results riêng
        for entry in table_entries:
            metadata = entry.get("metadata") or {}
            restaurant_id = self._extract_restaurant_id_from_metadata(metadata)
            if restaurant_id is None:
                continue
            key = str(restaurant_id)
            aggregator = aggregated.get(key)
            if not aggregator:
                aggregator = {
                    "restaurant_id": restaurant_id,
                    "restaurant": None,
                    "matched_menus": [],
                    "matched_services": [],
                    "matched_tables": [],
                    "matched_images": [],
                    "score": 0.0,
                    "sources": set(),
                }
                aggregated[key] = aggregator
            simplified_item = self._simplify_matched_item(
                metadata, entry.get("distance", 1.0), "table"
            )
            aggregator["matched_tables"].append(simplified_item)
            aggregator["sources"].add("tables")
            aggregator["score"] += max(0.0, 1.0 - entry.get("distance", 1.0))

        # Xử lý image_url search results
        for entry in image_entries:
            metadata = entry.get("metadata") or {}
            restaurant_id = metadata.get("restaurant_id")
            if restaurant_id is None:
                continue
            key = str(restaurant_id)
            aggregator = aggregated.get(key)
            if not aggregator:
                aggregator = {
                    "restaurant_id": restaurant_id,
                    "restaurant": None,
                    "matched_menus": [],
                    "matched_services": [],
                    "matched_tables": [],
                    "matched_images": [],
                    "score": 0.0,
                    "sources": set(),
                }
                aggregated[key] = aggregator
            image_item = {
                "url": metadata.get("url"),
                "mediaId": metadata.get("mediaId"),
                "type": metadata.get("type", "table_layout"),
                "distance": entry.get("distance", 1.0),
                "metadata": metadata,
            }
            aggregator["matched_images"].append(image_item)
            aggregator["sources"].add("image_url")
            aggregator["score"] += max(0.0, 1.0 - entry.get("distance", 1.0))

        if missing_restaurant_ids:
            fetched = await self.vector_service.get_restaurants_by_ids(list(missing_restaurant_ids))
            for fetched_id, payload in fetched.items():
                key = str(fetched_id)
                if key in aggregated and payload:
                    aggregated[key]["restaurant"] = dict(payload)

        aggregated_restaurants: List[Dict[str, Any]] = []
        aggregated_menus: List[Dict[str, Any]] = []
        aggregated_services: List[Dict[str, Any]] = []
        aggregated_tables: List[Dict[str, Any]] = []
        aggregated_images: List[Dict[str, Any]] = []

        for key, aggregator in aggregated.items():
            restaurant_meta = aggregator.get("restaurant") or {"id": aggregator.get("restaurant_id")}
            restaurant_copy = dict(restaurant_meta)
            restaurant_copy.setdefault("id", aggregator.get("restaurant_id"))
            restaurant_copy["_matchedMenus"] = aggregator.get("matched_menus", [])
            restaurant_copy["_matchedServices"] = aggregator.get("matched_services", [])
            restaurant_copy["_matchedTables"] = aggregator.get("matched_tables", [])
            restaurant_copy["_matchedImages"] = aggregator.get("matched_images", [])
            restaurant_copy["_matchScore"] = round(aggregator.get("score", 0.0), 4)
            restaurant_copy["_matchSources"] = list(aggregator.get("sources", set()))
            aggregated_restaurants.append(restaurant_copy)

            restaurant_name = (
                restaurant_copy.get("restaurantName")
                or restaurant_copy.get("name")
                or restaurant_copy.get("title")
                or ""
            )
            restaurant_identifier = (
                restaurant_copy.get("id")
                or restaurant_copy.get("restaurantId")
                or restaurant_copy.get("restaurant_id")
                or aggregator.get("restaurant_id")
            )

            for item in aggregator.get("matched_menus", []):
                meta = dict(item.get("metadata") or {})
                meta["_restaurantName"] = restaurant_name
                meta["_restaurantId"] = restaurant_identifier
                meta["_matchDistance"] = item.get("distance")
                aggregated_menus.append(meta)

            for item in aggregator.get("matched_services", []):
                meta = dict(item.get("metadata") or {})
                meta["_restaurantName"] = restaurant_name
                meta["_restaurantId"] = restaurant_identifier
                meta["_matchDistance"] = item.get("distance")
                aggregated_services.append(meta)

            for item in aggregator.get("matched_tables", []):
                meta = dict(item.get("metadata") or {})
                meta["_restaurantName"] = restaurant_name
                meta["_restaurantId"] = restaurant_identifier
                meta["_matchDistance"] = item.get("distance")
                aggregated_tables.append(meta)

            for item in aggregator.get("matched_images", []):
                image_meta = dict(item.get("metadata") or {})
                image_meta["_restaurantName"] = restaurant_name
                image_meta["_restaurantId"] = restaurant_identifier
                image_meta["_matchDistance"] = item.get("distance")
                aggregated_images.append(image_meta)

        aggregated_restaurants.sort(key=lambda r: r.get("_matchScore", 0.0), reverse=True)

        return {
            "restaurants": aggregated_restaurants,
            "menus": aggregated_menus,
            "services": aggregated_services,
            "tables": aggregated_tables,
            "images": aggregated_images,
        }

    def _render_match_summary(
        self,
        items: List[Dict[str, Any]],
        label: str,
        max_items: int = 3
    ) -> Optional[str]:
        if not items:
            return None
        highlights = []
        for item in items[:max_items]:
            name = item.get("name") or "N/A"
            description = item.get("description")
            price = item.get("price")
            snippet = name
            if description:
                trimmed = description.strip()
                if len(trimmed) > 60:
                    trimmed = trimmed[:60].rstrip() + "..."
                snippet += f" ({trimmed})"
            if price not in (None, "", "N/A"):
                snippet += f" - {price}"
            highlights.append(snippet)
        if len(items) > max_items:
            highlights.append(f"... và {len(items) - max_items} {label.lower()} khác")
        return f"{label}: " + "; ".join(highlights)
    
    async def _handle_restaurant_search(
        self, user_message: str, entities: Dict, user_id: str
    ) -> str:
        """Restaurant search - Two-Step Search: Find Restaurants → Search Related Data by Restaurant IDs"""
        try:
            # ✅ STEP 1: Extract restaurant_id nếu có
            restaurant_id = entities.get("restaurant_id")
            
            # 1. Detect các collections cần query
            intent_result = {"intent": "restaurant_search"}
            collections = await self._detect_required_collections(
                user_message, intent_result
            )
            
            # ✅ STEP 2: Multi-collection search WITH restaurant_id filter (nếu có)
            search_results = await self._multi_collection_search(
                user_message,
                collections,
                distance_threshold=0.5,  # CHỈ lấy results gần
                limit_per_collection=20,
                restaurant_id=restaurant_id  # ✅ Pass restaurant_id để filter (None nếu generic search)
            )
            
            # ✅ STEP 3: Aggregate cross-collection data để liên kết món ăn/dịch vụ ↔ nhà hàng
            aggregated = await self._aggregate_search_results(search_results)
            restaurants_enriched = aggregated.get("restaurants", [])
            menus_enriched = aggregated.get("menus", [])
            services_enriched = aggregated.get("services", [])

            # ✅ STEP 4: Apply entity filters trên aggregated data
            if entities.get("cuisine_type"):
                cuisine = entities["cuisine_type"].lower()
                restaurants_enriched = [
                    r for r in restaurants_enriched
                    if cuisine in (r.get("cuisineType", "") or "").lower()
                ]

            if entities.get("restaurant_id") is not None:
                target_id = entities["restaurant_id"]
                restaurants_enriched = [
                    r for r in restaurants_enriched
                    if self._extract_restaurant_id_from_metadata(r) == target_id
                ]

            allowed_ids = {
                self._extract_restaurant_id_from_metadata(r)
                for r in restaurants_enriched
            }
            allowed_ids.discard(None)

            if allowed_ids:
                menus_enriched = [
                    m for m in menus_enriched
                    if self._extract_restaurant_id_from_metadata(m) in allowed_ids
                ]
                services_enriched = [
                    s for s in services_enriched
                    if self._extract_restaurant_id_from_metadata(s) in allowed_ids
                ]

            # ✅ STEP 5: Nếu không có data → Không hallucinate
            if not restaurants_enriched and not menus_enriched and not services_enriched:
                return "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Bạn có thể thử tìm kiếm với từ khóa khác không?"

            # ✅ STEP 6: Format với AI - Đưa TẤT CẢ data vào
            return await self._format_multi_data_with_ai(
                user_message,
                restaurants=restaurants_enriched,
                menus=menus_enriched,
                services=services_enriched,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error in _handle_restaurant_search: {e}", exc_info=True)
            return "Xin lỗi, không thể tìm kiếm. Vui lòng thử lại sau."
    
    async def _handle_menu_inquiry(
        self, user_message: str, entities: Dict, user_id: str
    ) -> str:
        """Menu inquiry - Two-Step Search: Find Restaurant → Search Menus by Restaurant ID"""
        try:
            # ✅ STEP 1: Find restaurant first to get restaurant_id
            restaurant_id = None
            
            if entities.get("restaurant_id"):
                restaurant_id = entities["restaurant_id"]
            elif entities.get("restaurant_name"):
                # Search for restaurant by name
                restaurant_results = await self.vector_service.search_restaurants(
                    entities["restaurant_name"], limit=1, distance_threshold=0.7
                )
                if restaurant_results:
                    restaurant_id = self._extract_restaurant_id_from_metadata(
                        restaurant_results[0]["metadata"]
                    )
            
            # ✅ STEP 2: Detect collections
            intent_result = {"intent": "menu_inquiry"}
            collections = await self._detect_required_collections(
                user_message, intent_result
            )
            
            # ✅ STEP 3: Multi-collection search WITH restaurant_id filter
            search_results = await self._multi_collection_search(
                user_message,
                collections,
                distance_threshold=0.5,
                limit_per_collection=30,
                restaurant_id=restaurant_id  # ✅ Pass restaurant_id để filter
            )
            
            aggregated = await self._aggregate_search_results(search_results)
            restaurants_enriched = aggregated.get("restaurants", [])
            menus_enriched = aggregated.get("menus", [])
            services_enriched = aggregated.get("services", [])

            # ✅ Additional filter nếu có restaurant_id từ entities
            if restaurant_id is not None:
                target_id = restaurant_id
                restaurants_enriched = [
                    r for r in restaurants_enriched
                    if self._extract_restaurant_id_from_metadata(r) == target_id
                ]
                menus_enriched = [
                    m for m in menus_enriched
                    if self._extract_restaurant_id_from_metadata(m) == target_id
                ]
                services_enriched = [
                    s for s in services_enriched
                    if self._extract_restaurant_id_from_metadata(s) == target_id
                ]

            if not menus_enriched and not services_enriched:
                return "Xin lỗi, tôi không tìm thấy thực đơn phù hợp. Bạn có thể thử tìm kiếm với từ khóa khác không?"

            return await self._format_multi_data_with_ai(
                user_message,
                restaurants=restaurants_enriched,
                menus=menus_enriched,
                services=services_enriched,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error in _handle_menu_inquiry: {e}", exc_info=True)
            return "Xin lỗi, không thể lấy thực đơn. Vui lòng thử lại sau."
    
    
    async def _handle_general_inquiry(self, user_message: str, user_id: str) -> str:
        """General inquiry - Two-Step Search: Find Restaurant → Search Related Data by Restaurant ID"""
        try:
            # ✅ STEP 1: Try to extract restaurant_id from context or previous turn state
            restaurant_id = None
            
            # Check turn state (follow-up questions)
            turn_state = self.turn_states.get(user_id)
            if turn_state and turn_state.get("last_restaurant_id"):
                restaurant_id = turn_state["last_restaurant_id"]
            
            # ✅ STEP 2: Detect collections
            intent_result = {"intent": "general_inquiry"}
            collections = await self._detect_required_collections(
                user_message, intent_result
            )
            
            # ✅ STEP 3: Multi-collection search WITH restaurant_id filter
            search_results = await self._multi_collection_search(
                user_message,
                collections,
                distance_threshold=0.5,
                limit_per_collection=10,
                restaurant_id=restaurant_id  # ✅ Pass restaurant_id để filter
            )
            
            aggregated = await self._aggregate_search_results(search_results)
            restaurants_enriched = aggregated.get("restaurants", [])
            menus_enriched = aggregated.get("menus", [])
            services_enriched = aggregated.get("services", [])

            if not restaurants_enriched and not menus_enriched and not services_enriched:
                return self._get_fallback_response(user_message)

            return await self._format_multi_data_with_ai(
                user_message,
                restaurants=restaurants_enriched,
                menus=menus_enriched,
                services=services_enriched,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error in _handle_general_inquiry: {e}", exc_info=True)
            return self._get_fallback_response(user_message)
    
    async def _format_multi_data_with_ai(
        self,
        user_message: str,
        restaurants: List[Dict] = None,
        menus: List[Dict] = None,
        services: List[Dict] = None,
        user_id: str = None
    ) -> str:
        """Format nhiều loại data với AI - STRICT DATA INJECTION"""
        
        restaurants = restaurants or []
        menus = menus or []
        services = services or []
        
        # Build STRICT data context cho TẤT CẢ loại data
        data_context_parts = []
        
        if restaurants:
            # Không limit vì đã filter theo distance - chỉ giới hạn token (max 20 restaurants)
            restaurants_to_show = restaurants[:20] if len(restaurants) > 20 else restaurants
            restaurant_lines: List[str] = []
            for index, r in enumerate(restaurants_to_show, 1):
                base_line = (
                    f"{index}. NHÀ HÀNG - Tên: {r.get('restaurantName', r.get('name', 'N/A'))}, "
                    f"Địa chỉ: {r.get('address', 'N/A')}, "
                    f"Loại: {r.get('cuisineType', 'N/A')}, "
                    f"Rating: {r.get('rating', 'N/A')}/5"
                )
                detail_lines = [base_line]

                menu_summary = self._render_match_summary(r.get("_matchedMenus", []), "Món phù hợp")
                if menu_summary:
                    detail_lines.append(f"   • {menu_summary}")

                service_summary = self._render_match_summary(r.get("_matchedServices", []), "Dịch vụ nổi bật")
                if service_summary:
                    detail_lines.append(f"   • {service_summary}")

                table_summary = self._render_match_summary(r.get("_matchedTables", []), "Bố trí bàn", max_items=2)
                if table_summary:
                    detail_lines.append(f"   • {table_summary}")

                restaurant_lines.append("\n".join(detail_lines))

            if len(restaurants) > 20:
                restaurant_lines.append(f"... và {len(restaurants) - 20} nhà hàng khác.")

            data_context_parts.append("DANH SÁCH NHÀ HÀNG:\n" + "\n".join(restaurant_lines))
        
        if menus:
            # Không limit vì đã filter theo distance - chỉ giới hạn token (max 30 dishes)
            menus_to_show = menus[:30] if len(menus) > 30 else menus
            menu_lines: List[str] = []
            for dish in menus_to_show:
                dish_name = dish.get("name") or dish.get("dishName") or "N/A"
                price = dish.get("price")
                restaurant_name = dish.get("_restaurantName") or dish.get("restaurantName")
                description = dish.get("description")
                line = f"- MÓN - {dish_name}"
                if restaurant_name:
                    line += f" (Nhà hàng: {restaurant_name})"
                if price in (None, "", "N/A"):
                    line += ": N/A"
                else:
                    price_str = str(price)
                    if "vnđ" not in price_str.lower():
                        price_str += " VNĐ"
                    line += f": {price_str}"
                if description:
                    trimmed_desc = description.strip()
                    if len(trimmed_desc) > 80:
                        trimmed_desc = trimmed_desc[:80].rstrip() + "..."
                    line += f" - {trimmed_desc}"
                menu_lines.append(line)
            if len(menus) > 30:
                menu_lines.append(f"... và {len(menus) - 30} món khác.")
            data_context_parts.append("DANH SÁCH MÓN ĂN:\n" + "\n".join(menu_lines))
        
        if services:
            # Không limit vì đã filter theo distance - chỉ giới hạn token (max 15 services)
            services_to_show = services[:15] if len(services) > 15 else services
            service_lines: List[str] = []
            for service in services_to_show:
                service_name = service.get("name") or service.get("serviceName") or "N/A"
                description = service.get("description") or service.get("serviceDescription") or "N/A"
                restaurant_name = service.get("_restaurantName") or service.get("restaurantName")
                line = f"- DỊCH VỤ - {service_name}"
                if restaurant_name:
                    line += f" (Nhà hàng: {restaurant_name})"
                line += f": {description}"
                service_lines.append(line)
            if len(services) > 15:
                service_lines.append(f"... và {len(services) - 15} dịch vụ khác.")
            data_context_parts.append("DANH SÁCH DỊCH VỤ:\n" + "\n".join(service_lines))
        
        data_context = "\n\n".join(data_context_parts)
        
        if not data_context:
            return "Không tìm thấy thông tin phù hợp."
        
        # STRICT System Prompt
        strict_prompt = f"""{self.strict_system_prompt}

DỮ LIỆU THỰC TẾ (CHỈ ĐƯỢC ĐỀ CẬP ĐẾN CÁC THÔNG TIN NÀY):
{data_context}

QUAN TRỌNG:
- CHỈ được đề cập đến thông tin trong danh sách trên
- KHÔNG được tự tạo tên nhà hàng, món ăn, dịch vụ, hoặc thông tin nào khác
- Nếu user hỏi về thông tin không có trong danh sách → Nói "Không tìm thấy"
- Format response tự nhiên và tổng hợp các loại thông tin một cách hợp lý"""
        
        messages = [
            {"role": "system", "content": strict_prompt},
            {"role": "user", "content": user_message}
        ]
        
        history = self.conversations.get(user_id, [])
        if history:
            messages = [messages[0]] + history[-4:] + [messages[1]]
        
        response = await self._call_openai(messages)
        
        # Fallback nếu AI không hoạt động
        if not response:
            return self._format_multi_data_fallback(restaurants, menus, services)
        
        return response
    
    async def _format_api_response_with_ai(
        self, user_message: str, api_response: str, response_type: str, user_id: str
    ) -> str:
        """Format API response với AI để tự nhiên hơn"""
        strict_prompt = f"""{self.strict_system_prompt}

DỮ LIỆU TỪ API:
{api_response}

QUAN TRỌNG:
- Format lại response tự nhiên và thân thiện hơn
- GIỮ NGUYÊN thông tin từ API response
- KHÔNG được thay đổi hoặc thêm thông tin không có trong API response"""
        
        messages = [
            {"role": "system", "content": strict_prompt},
            {"role": "user", "content": f"Format lại response này một cách tự nhiên: {api_response}"}
        ]
        
        response = await self._call_openai(messages)
        return response or api_response  # Fallback to original if AI fails
    
    def _format_multi_data_fallback(
        self, restaurants: List[Dict], menus: List[Dict], services: List[Dict]
    ) -> str:
        """Fallback formatting nếu AI không hoạt động"""
        response_parts = []
        
        if restaurants:
            response_parts.append(f"🍽️ **Tìm thấy {len(restaurants)} nhà hàng:**\n\n")
            # Show tất cả restaurants (đã filter theo distance)
            restaurants_to_show = restaurants[:20] if len(restaurants) > 20 else restaurants
            for i, r in enumerate(restaurants_to_show, 1):
                lines = [
                    f"**{i}. {r.get('restaurantName', r.get('name', 'N/A'))}**",
                    f"📍 {r.get('address', 'N/A')}",
                    f"🍽️ {r.get('cuisineType', 'N/A')}"
                ]
                menu_summary = self._render_match_summary(r.get("_matchedMenus", []), "Món phù hợp")
                if menu_summary:
                    lines.append(f"• {menu_summary}")
                service_summary = self._render_match_summary(r.get("_matchedServices", []), "Dịch vụ nổi bật")
                if service_summary:
                    lines.append(f"• {service_summary}")
                response_parts.append("\n".join(lines) + "\n\n")
            if len(restaurants) > 20:
                response_parts.append(f"... và {len(restaurants) - 20} nhà hàng khác.\n\n")
        
        if menus:
            response_parts.append(f"🍽️ **Thực đơn:**\n\n")
            # Show tất cả menus (đã filter theo distance)
            menus_to_show = menus[:30] if len(menus) > 30 else menus
            for dish in menus_to_show:
                dish_name = dish.get("name") or dish.get("dishName") or "N/A"
                price = dish.get("price")
                restaurant_name = dish.get("_restaurantName") or dish.get("restaurantName")
                description = dish.get("description")
                line = f"• **{dish_name}**"
                if restaurant_name:
                    line += f" (Nhà hàng: {restaurant_name})"
                if price in (None, "", "N/A"):
                    line += " - N/A"
                else:
                    price_str = str(price)
                    if "vnđ" not in price_str.lower():
                        price_str += " VNĐ"
                    line += f" - {price_str}"
                if description:
                    trimmed_desc = description.strip()
                    if len(trimmed_desc) > 80:
                        trimmed_desc = trimmed_desc[:80].rstrip() + "..."
                    line += f" - {trimmed_desc}"
                response_parts.append(line + "\n")
            if len(menus) > 30:
                response_parts.append(f"... và {len(menus) - 30} món khác.\n")
        
        return "".join(response_parts) if response_parts else "Không tìm thấy thông tin."
    
    def _build_messages(self, conversation_id: str, user_message: str) -> List[Dict[str, str]]:
        """Build conversation messages with context"""
        history = self.conversations.get(conversation_id, [])
        
        # Keep only last 10 messages for context
        recent_history = history[-10:] if history else []
        
        messages = [
            {"role": "system", "content": self.strict_system_prompt}
        ]
        
        # Add conversation history
        messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _build_enhanced_messages(self, conversation_id: str, user_message: str, context: str) -> List[Dict[str, str]]:
        """Build enhanced conversation messages với Vector Database context"""
        history = self.conversations.get(conversation_id, [])
        
        # Keep only last 10 messages for context
        recent_history = history[-10:] if history else []
        
        # Enhanced system prompt với context
        enhanced_system_prompt = self.strict_system_prompt
        if context:
            enhanced_system_prompt += f"\n\nRelevant context from previous conversations:\n{context}"
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt}
        ]
        
        # Add conversation history
        messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _call_openai(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Call OpenAI API với strict settings để giảm hallucination"""
        if not self.openai_client:
            if settings.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                return None

        try:
            completion = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.3,  # Lower temperature để giảm hallucination
                max_tokens=500,  # Limit tokens để tránh dài dòng
            )
            
            if completion.choices:
                return completion.choices[0].message.content.strip()
            return None

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def _store_conversation(self, conversation_id: str, user_message: str, response: str):
        """Store conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        # Keep conversation history bounded to avoid unbounded memory growth.
        if len(self.conversations[conversation_id]) > 20:
            self.conversations[conversation_id] = self.conversations[conversation_id][-20:]
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Fallback responses for common queries"""
        message_lower = user_message.lower()
        
        if any(keyword in message_lower for keyword in ["giờ", "open", "hour"]):
            return "Nhà hàng mở cửa từ 10:00 đến 22:00 mỗi ngày. Bạn muốn đặt bàn khung giờ nào?"
        
        elif any(keyword in message_lower for keyword in ["địa chỉ", "address", "ở đâu"]):
            return "Nhà hàng nằm tại trung tâm thành phố. Bạn có thể tìm kiếm nhà hàng gần vị trí của bạn không?"
        
        elif any(keyword in message_lower for keyword in ["giá", "price", "cost"]):
            return "Giá cả tùy thuộc vào món ăn và nhà hàng. Bạn muốn xem menu của nhà hàng nào?"
        
        else:
            return "Tôi là trợ lý đặt bàn nhà hàng. Tôi có thể giúp bạn tìm nhà hàng, xem menu, đặt bàn hoặc kiểm tra voucher. Bạn cần hỗ trợ gì?"
    
    async def initialize_vector_database(self):
        """Initialize Vector Database với restaurant data + Intent Embeddings"""
        try:
            logger.info("Initializing Vector Database...")
            
            # Clear old intent embeddings first (xóa các intent cũ)
            await self.vector_service.clear_all_intent_embeddings()
            logger.info("Cleared old intent embeddings")
            
            # Initialize Intent Embeddings FIRST (nhanh)
            await self.intent_service.initialize_intent_embeddings()
            logger.info("Intent embeddings initialized")
            
            # Get restaurant data từ Spring API
            from app.services.spring_api_client import spring_api_client
            
            restaurants = await spring_api_client.get_all_restaurants()
            if restaurants:
                # Store restaurant data in vector database
                await self.vector_service.store_restaurant_data(restaurants)
                logger.info(f"Stored {len(restaurants)} restaurants in Vector Database")
                
                # Store additional data for ALL restaurants
                total_restaurants = len(restaurants)
                for i, restaurant in enumerate(restaurants):
                    restaurant_id = (
                        restaurant.get('id')
                        or restaurant.get('restaurantId')
                        or restaurant.get('restaurantID')
                    )
                    if restaurant_id:
                        try:
                            # Store menu data
                            menu = await spring_api_client.get_restaurant_menu(restaurant_id)
                            if menu:
                                logger.debug(
                                    "Fetched %d menu items for restaurant %s. Sample: %s",
                                    len(menu),
                                    restaurant_id,
                                    menu[0] if isinstance(menu, list) and menu else menu,
                                )
                                await self.vector_service.store_menu_data(restaurant_id, menu)
                                logger.info(f"Stored menu for restaurant {restaurant_id} ({i+1}/{total_restaurants})")
                            else:
                                logger.warning(f"No menu data for restaurant {restaurant_id}")
                            
                            # Store restaurant services
                            services = await spring_api_client.get_restaurant_services(restaurant_id)
                            if services:
                                logger.debug(
                                    "Fetched %d services for restaurant %s. Sample: %s",
                                    len(services),
                                    restaurant_id,
                                    services[0] if isinstance(services, list) and services else services,
                                )
                                await self.vector_service.store_services_data(restaurant_id, services)
                                logger.info(f"Stored services for restaurant {restaurant_id}")
                            
                            tables = await spring_api_client.get_restaurant_tables(restaurant_id)
                            if tables:
                                logger.debug(
                                    "Fetched %d tables for restaurant %s. Sample: %s",
                                    len(tables),
                                    restaurant_id,
                                    tables[0] if isinstance(tables, list) and tables else tables,
                                )
                                await self.vector_service.store_tables_data(restaurant_id, tables)
                                logger.info(f"Stored tables for restaurant {restaurant_id}")
                            
                            # Store table layouts
                            table_layouts = await spring_api_client.get_table_layouts(restaurant_id)
                            if table_layouts:
                                logger.debug(
                                    "Fetched %d table layouts for restaurant %s. Sample: %s",
                                    len(table_layouts),
                                    restaurant_id,
                                    table_layouts[0] if isinstance(table_layouts, list) and table_layouts else table_layouts,
                                )
                                await self.vector_service.store_table_layouts_data(restaurant_id, table_layouts)
                                logger.info(f"Stored table layouts for restaurant {restaurant_id}")
                                
                        except Exception as e:
                            logger.error(f"Error storing data for restaurant {restaurant_id}: {e}")
                            continue
            
            logger.info("Vector Database initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing Vector Database: {e}")
    
    async def get_vector_database_stats(self) -> Dict[str, any]:
        """Get statistics về Vector Database"""
        try:
            stats = self.vector_service.get_collection_stats()
            return {
                "status": "healthy",
                "collections": stats,
                "total_items": sum(stats.values())
            }
        except Exception as e:
            logger.error(f"Error getting Vector Database stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global instance
restaurant_agent = RestaurantAgent()

# Backward compatibility
async def handle_message(payload: MessageRequest) -> MessageResponse:
    return await restaurant_agent.handle_message(payload)
