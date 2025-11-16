import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional
from app.models import MessageRequest, MessageResponse
from app.services.vector_intent_service import vector_intent_service
from app.services.function_service import FunctionService
from app.services.vector_service import vector_service
from app.services.menu_reasoning_service import menu_reasoning_service
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
3. ✅ If data IS available in the provided list → MUST mention at least some restaurants/dishes
4. ✅ Chỉ được trả "Không tìm thấy thông tin" khi không có bất kỳ món hoặc nhà hàng nào trong danh sách dữ liệu
5. Always verify information exists in provided data before mentioning it

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
            # ✅ FIX: Không học sai khi pattern đã match nhưng LLM trả general_inquiry
            intent_method = intent_result.get("method", "")
            intent_name = intent_result["intent"]
            
            # Nếu pattern đã override LLM general_inquiry → không học với general_inquiry
            if (intent_method in ["pattern_override_llm_general", "pattern_override_llm_low_confidence"] and
                intent_name == "general_inquiry"):
                logger.info(
                    f"Skipping learning: pattern overrode LLM general_inquiry, "
                    f"not learning incorrect intent"
                )
                # Không gọi learn_from_interaction để tránh học sai
            else:
                await self.intent_service.learn_from_interaction(
                    payload.userId, payload.message, intent_name, 
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
        Detect các collections cần query dựa trên INTENT (semantic-first)
        
        ✅ ĐÃ BỎ KEYWORD DETECTION - Dựa vào intent đã được recognize
        Chỉ giữ keyword detection như fallback cuối cùng nếu intent không rõ
        
        Returns:
            List of collection names: ['restaurants', 'menus', 'tables', 'image_url', ...]
        """
        collections = []
        intent = intent_result.get("intent", "")
        confidence = intent_result.get("confidence", 0.0)
        
        # ✅ SEMANTIC-FIRST: Dựa vào intent đã được recognize (không dùng keywords)
        if intent == "restaurant_search":
            collections.append("restaurants")
            # Menu inquiry thường đi kèm với restaurant search → thêm menus để có context
            collections.append("menus")
            # Layout/table info thường cần cho restaurant search → thêm image_url
            collections.append("image_url")
        
        elif intent == "menu_inquiry":
            collections.append("menus")
            # Menu cần restaurant context → thêm restaurants
            collections.append("restaurants")
            # Có thể có table/layout info → thêm image_url
            collections.append("image_url")
        
        elif intent == "table_inquiry":
            collections.append("menus")  # Tables stored in menus collection
            collections.append("restaurants")  # Need restaurant context
            collections.append("image_url")  # Table layouts
        
        elif intent == "voucher_inquiry":
            collections.append("restaurants")  # Vouchers associated with restaurants
        
        else:
            # General query hoặc intent không rõ → search tất cả collections
            # (Semantic search sẽ tự filter kết quả phù hợp)
            collections = ["restaurants", "menus", "image_url"]
        
        # ✅ FALLBACK: Nếu intent confidence quá thấp (<0.3) → search tất cả
        if confidence < 0.3:
            logger.debug(f"Low intent confidence ({confidence}), searching all collections")
            collections = ["restaurants", "menus", "image_url"]
        
        # Đảm bảo có ít nhất 1 collection
        if not collections:
            collections.append("restaurants")
        
        # Remove duplicates và return
        unique_collections = list(set(collections))
        logger.debug(f"Detected collections (intent-based): {unique_collections} for intent: {intent} (confidence: {confidence})")
        return unique_collections
    
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

    def _extract_forbidden_tags(self, reasoning_profile: Dict[str, Any]) -> List[str]:
        """
        Extract forbidden tags từ reasoning profile
        
        ✅ CẢI THIỆN: Detect condition để chỉ filter đúng những gì cần thiết, không quá gắt.
        
        Strategy:
        1. Detect condition từ summary/constraints_text (sẹo, ăn chay, gout, tiểu đường...)
        2. Map condition → forbidden tags cụ thể (ví dụ: sẹo chỉ kiêng bò + hải sản, KHÔNG kiêng tất cả thịt)
        
        Args:
            reasoning_profile: Profile từ LLM reasoning
            
        Returns:
            List of forbidden tags (e.g., ["beef", "seafood", "tôm", "cua", ...])
        """
        forbidden_tags = []
        constraints_text = reasoning_profile.get("constraints_text", [])
        summary = (reasoning_profile.get("summary", "") or "").lower()
        constraints_text_str = " ".join(str(c) for c in constraints_text).lower()
        
        # ✅ Detect condition từ summary và constraints để map đúng
        is_scar_condition = any(keyword in summary for keyword in [
            "sẹo", "vết mổ", "mới phẫu thuật", "phẫu thuật", "vết thương"
        ]) or any(keyword in constraints_text_str for keyword in ["sẹo", "vết mổ"])
        
        is_vegetarian = any(keyword in summary for keyword in [
            "ăn chay", "đồ chay", "vegetarian", "vegan"
        ]) or any(keyword in constraints_text_str for keyword in [
            "ăn chay", "đồ chay", "không ăn thịt"
        ])
        
        is_gout = any(keyword in summary for keyword in ["gout", "đau khớp"]) or any(
            keyword in constraints_text_str for keyword in ["gout", "đau khớp", "thịt đỏ"]
        )
        
        # Map constraint text → tags (theo từng constraint cụ thể)
        for constraint in constraints_text:
            constraint_lower = str(constraint).lower()
            
            # ========== THỊT BÒ ==========
            if any(keyword in constraint_lower for keyword in ["tránh thịt bò", "kiêng bò", "không bò", "tránh bò"]):
                forbidden_tags.extend(["beef", "thịt bò", "bò"])
            
            # ========== HẢI SẢN (tổng quát) ==========
            if any(keyword in constraint_lower for keyword in ["tránh hải sản", "kiêng hải sản", "không hải sản"]):
                forbidden_tags.extend([
                    "seafood", "hải sản",
                    "shrimp", "tôm",
                    "crab", "cua",
                    "squid", "mực",
                    "clam", "nghêu",
                    "scallop", "sò", "sò điệp",
                    "snail", "ốc",
                    "oyster", "hàu"
                ])
            
            # Tôm cụ thể
            if "tránh tôm" in constraint_lower or "kiêng tôm" in constraint_lower:
                forbidden_tags.extend(["shrimp", "tôm"])
            
            # Cua cụ thể
            if "tránh cua" in constraint_lower or "kiêng cua" in constraint_lower:
                forbidden_tags.extend(["crab", "cua"])
            
            # Mực cụ thể
            if "tránh mực" in constraint_lower or "kiêng mực" in constraint_lower:
                forbidden_tags.extend(["squid", "mực"])
            
            # ========== CAY ==========
            if any(keyword in constraint_lower for keyword in [
                "không cay", "tránh cay", "kiêng cay", "không quá cay", 
                "ít cay", "hạn chế cay", "không ớt"
            ]):
                forbidden_tags.extend(["spicy", "cay", "ớt", "pepper", "chili"])
            
            # ========== ĐỒ CHIÊN XÀO ==========
            if any(keyword in constraint_lower for keyword in [
                "tránh đồ chiên", "tránh đồ xào", "kiêng chiên xào",
                "hạn chế đồ chiên xào", "không chiên", "không xào"
            ]):
                forbidden_tags.extend(["fried", "deep_fried", "chiên", "xào", "pan_fried"])
            
            # ========== ĐƯỜNG / NGỌT (tiểu đường) ==========
            if any(keyword in constraint_lower for keyword in [
                "ít đường", "không đường", "tránh đường", "kiêng đường",
                "tránh đồ ngọt", "ít ngọt", "không ngọt", "tiểu đường"
            ]):
                forbidden_tags.extend(["sweet", "dessert", "sugar", "đường", "ngọt", "caramel"])
            
            # ========== MUỐI / MẶN (huyết áp cao) ==========
            if any(keyword in constraint_lower for keyword in [
                "ít muối", "không muối", "tránh muối", "huyết áp cao",
                "tăng huyết áp", "ít mặn", "không mặn"
            ]):
                forbidden_tags.extend(["mặn", "muối", "salty"])  # Fallback cho text matching
            
            # ========== RƯỢU ==========
            if any(keyword in constraint_lower for keyword in [
                "tránh rượu", "kiêng rượu", "không rượu", "gan yếu", "viêm gan"
            ]):
                forbidden_tags.extend(["alcohol", "rượu", "beer", "bia", "wine"])
        
        # ========== ĐIỀU KIỆN ĐẶC BIỆT ==========
        # ✅ SẸO / VẾT MỔ: Chỉ kiêng bò + hải sản (KHÔNG kiêng tất cả thịt)
        if is_scar_condition:
            # Chỉ add bò + hải sản nếu chưa có (tránh duplicate)
            if "beef" not in forbidden_tags and "thịt bò" not in forbidden_tags:
                forbidden_tags.extend(["beef", "thịt bò", "bò"])
            if "seafood" not in forbidden_tags and "hải sản" not in forbidden_tags:
                forbidden_tags.extend([
                    "seafood", "hải sản",
                    "shrimp", "tôm",
                    "crab", "cua",
                    "squid", "mực",
                    "clam", "nghêu",
                    "scallop", "sò", "sò điệp",
                    "snail", "ốc",
                    "oyster", "hàu"
                ])
            # ✅ QUAN TRỌNG: Không add "meat", "thịt", "pork", "chicken" cho sẹo
        
        # ✅ ĂN CHAY: Kiêng tất cả thịt
        if is_vegetarian:
            forbidden_tags.extend([
                "beef", "pork", "chicken", "meat",
                "thịt bò", "thịt heo", "thịt gà", "thịt"
            ])
        
        # ✅ GOUT / ĐAU KHỚP: Chỉ kiêng thịt đỏ
        if is_gout:
            forbidden_tags.extend(["beef", "thịt bò", "pork", "thịt heo", "lamb", "thịt cừu"])
        
        # ✅ Note: "tránh thịt" generic - chỉ apply nếu không phải sẹo/gout (để tránh conflict)
        if not is_scar_condition and not is_gout:
            for constraint in constraints_text:
                constraint_lower = str(constraint).lower()
                if any(keyword in constraint_lower for keyword in [
                    "không thịt", "tránh thịt", "kiêng thịt", "không có thịt"
                ]):
                    if "meat" not in forbidden_tags and "thịt" not in forbidden_tags:
                        forbidden_tags.extend(["beef", "pork", "chicken", "meat", "thịt bò", "thịt heo", "thịt gà", "thịt"])
        
        # Remove duplicates và return
        return list(set(forbidden_tags))
    
    def _filter_by_forbidden_tags(
        self, search_results: Dict[str, List[Dict]], forbidden_tags: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Filter search results loại bỏ món có forbidden tags
        
        Strategy:
        - Check tags trong metadata
        - Check name, description, ingredients cho forbidden keywords
        - Chỉ filter menus, giữ nguyên restaurants/services
        
        Args:
            search_results: Dict với keys là collection names, values là list of results
            forbidden_tags: List of tags/keywords cần tránh
            
        Returns:
            Filtered search_results
        """
        if not forbidden_tags:
            return search_results
        
        filtered = {}
        
        for collection_name, results in search_results.items():
            if collection_name == "menus":
                # Filter menus - loại bỏ món có forbidden tags/keywords
                filtered_items = []
                
                for item in results:
                    metadata = item.get("metadata", {})
                    
                    # Get tags (ensure it's a list)
                    tags = metadata.get("tags", [])
                    if isinstance(tags, str):
                        try:
                            import ast
                            tags = ast.literal_eval(tags) if tags.startswith("[") else [tags]
                        except:
                            tags = [tags] if tags else []
                    if not isinstance(tags, list):
                        tags = []
                    
                    # ✅ Get ingredient_tags (MỚI - dùng cho dị ứng/kiêng khem)
                    ingredient_tags = metadata.get("ingredient_tags", [])
                    if isinstance(ingredient_tags, str):
                        try:
                            import ast
                            ingredient_tags = ast.literal_eval(ingredient_tags) if ingredient_tags.startswith("[") else [ingredient_tags]
                        except:
                            ingredient_tags = [ingredient_tags] if ingredient_tags else []
                    if not isinstance(ingredient_tags, list):
                        ingredient_tags = []
                    
                    # Get text fields để check (fallback nếu không có tags)
                    name = (metadata.get("name") or "").lower()
                    description = (metadata.get("description") or "").lower()
                    ingredients = metadata.get("ingredients", "")
                    if isinstance(ingredients, str):
                        ingredients = ingredients.lower()
                    elif isinstance(ingredients, list):
                        ingredients = " ".join(str(i) for i in ingredients).lower()
                    else:
                        ingredients = ""
                    
                    # Check nếu có forbidden tag/keyword
                    has_forbidden = False
                    for forbidden in forbidden_tags:
                        forbidden_lower = forbidden.lower()
                        
                        # ✅ Priority 1: Check trong ingredient_tags (CHÍNH XÁC NHẤT)
                        if any(forbidden_lower == tag.lower() for tag in ingredient_tags):
                            has_forbidden = True
                            break
                        
                        # ✅ Priority 2: Check trong tags (health/lifestyle tags)
                        if any(forbidden_lower == tag.lower() for tag in tags):
                            has_forbidden = True
                            break
                        
                        # ✅ Priority 3: Check trong name, description, ingredients (fallback)
                        if forbidden_lower in name or forbidden_lower in description or forbidden_lower in ingredients:
                            has_forbidden = True
                            break
                    
                    # Chỉ giữ món không có forbidden tags
                    if not has_forbidden:
                        filtered_items.append(item)
                    else:
                        logger.debug(
                            f"Filtered out menu item '{metadata.get('name', 'N/A')}' "
                            f"due to forbidden tags: {forbidden_tags}"
                        )
                
                filtered[collection_name] = filtered_items
                logger.info(
                    f"Filtered menus: {len(results)} → {len(filtered_items)} "
                    f"(removed {len(results) - len(filtered_items)} items with forbidden tags)"
                )
            else:
                # Giữ nguyên restaurants, services, và các collections khác
                filtered[collection_name] = results
        
        return filtered
    
    def _normalize_menu_item(self, dish: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize menu item từ search result format về flat format
        
        Handles:
        - Search result format: {"metadata": {...}, "distance": ..., "score": ...}
        - Aggregated format: {"name": ..., "description": ..., "_restaurantName": ...}
        - Direct format: {"name": ..., "description": ...}
        
        Returns:
            Normalized dict với tất cả fields ở root level
        """
        # Extract metadata nếu có (search result format)
        if "metadata" in dish and isinstance(dish.get("metadata"), dict):
            metadata = dish.get("metadata", {})
            normalized = dict(metadata)  # Copy metadata fields
            # Preserve enriched fields từ aggregation (như _restaurantName, _restaurantId, distance, score)
            normalized.update({
                k: v for k, v in dish.items() 
                if k not in ["metadata", "document", "id"] and k.startswith("_")
            })
            # Preserve distance, score nếu có
            if "distance" in dish:
                normalized["distance"] = dish["distance"]
            if "score" in dish:
                normalized["score"] = dish["score"]
            return normalized
        
        # Nếu không có metadata → dùng dish trực tiếp (đã là flat format)
        return dish

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
    
    async def _extract_restaurants_from_history(self, user_id: str) -> List[Dict]:
        """
        Extract restaurants từ conversation history của user
        
        Tìm trong assistant messages trong conversation history để extract:
        - Restaurant names (từ format "1. **Seoul BBQ Premium**")
        - Restaurant IDs (nếu có)
        """
        try:
            # Lấy recent conversations
            conversations = await self.vector_service.get_user_conversations_recent(user_id, limit=10)
            if not conversations:
                logger.info(f"No conversation history found for user {user_id}")
                return []
            
            restaurants = []
            seen_names = set()
            
            # Patterns để extract restaurant names từ assistant responses
            # ✅ FIX: Chỉ extract từ list restaurants (có số thứ tự), không extract món ăn
            patterns = [
                # Format: "1. **Seoul BBQ Premium**" (có số thứ tự ở đầu)
                r"\n\s*\d+\.\s+\*\*([^*]+?)\*\*",
                # Format: "**Seoul BBQ Premium**" (không có "- " hoặc "( " phía trước, không có VNĐ sau)
                r"(?<![-(\s])\*\*([A-Za-z0-9\s\-]+(?:Restaurant|BBQ|Premium|Restaurants|Cafe|Café|Bar|Bistro)?)\*\*(?!\s*[-\()]|\s*-\s*\d+\.?\d*\s*VN?Đ)",
                # Format: "NHÀ HÀNG - Tên: Seoul BBQ Premium"
                r"NHÀ HÀNG\s*-\s*Tên:\s*([A-Za-z0-9\s\-]+?)(?:\s|,|$|Địa)",
            ]
            
            # Tìm trong assistant messages (những message có format list restaurants)
            for conv in conversations:
                document = conv.get("document", "")
                if not document:
                    continue
                
                # Chỉ tìm trong assistant responses (có format markdown với **)
                if "**" not in document and "NHÀ HÀNG" not in document:
                    continue
                
                # Extract restaurant names
                for pattern in patterns:
                    matches = re.finditer(pattern, document, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        name = match.group(1).strip()
                        
                        # ✅ FIX: Validate để loại bỏ món ăn
                        # 1. Check context xung quanh để loại bỏ món ăn (có VNĐ, giá, "- 50.000")
                        match_start = match.start()
                        match_end = match.end()
                        context_before = document[max(0, match_start - 30):match_start].lower()
                        context_after = document[match_end:min(len(document), match_end + 50)].lower()
                        
                        # Loại bỏ nếu có pattern giá cả (VNĐ, đồng, giá, price) trong context
                        if any(keyword in context_before or keyword in context_after for keyword in [
                            "vnđ", "đồng", "giá", "price", "- 50", "- 100", "000", "triệu", "k"
                        ]):
                            # Nhưng giữ lại nếu có pattern nhà hàng (Restaurant, BBQ, Premium)
                            if not any(keyword in name.lower() for keyword in [
                                "restaurant", "bbq", "premium", "cafe", "café", "bar", "bistro"
                            ]):
                                logger.debug(f"Skipping '{name}' - looks like a dish (has price context)")
                                continue
                        
                        # 2. Filter out common words và validate
                        if (len(name) > 3 and 
                            name.lower() not in ['có', 'là', 'nào', 'và', 'cho', 'món', 'địa chỉ', 'loại', 'rating', 
                                                  'canh', 'cá', 'cơm', 'bánh', 'lẩu', 'gỏi', 'chả', 'súp', 'salad'] and
                            # ✅ Loại bỏ tên món ăn phổ biến (thường ngắn và có từ khóa món ăn)
                            not any(dish_keyword in name.lower() for dish_keyword in [
                                'canh', 'súp', 'gỏi', 'chả', 'nem', 'bánh', 'cơm', 'lẩu', 'salad'
                            ]) and
                            name not in seen_names):
                            seen_names.add(name)
                            restaurants.append({
                                "name": name,
                                "id": None  # ID sẽ được tìm sau
                            })
                            logger.debug(f"Extracted restaurant name from history: {name}")
            
            # Limit to top 5 restaurants (thường user chỉ hỏi về 2-3 restaurants)
            restaurants = restaurants[:5]
            logger.info(f"Extracted {len(restaurants)} restaurants from history: {[r['name'] for r in restaurants]}")
            return restaurants
            
        except Exception as e:
            logger.error(f"Error extracting restaurants from history: {e}", exc_info=True)
            return []
    
    async def _search_restaurants_by_names_or_ids(
        self, restaurant_names: List[str] = None, restaurant_ids: List[Any] = None
    ) -> List[Dict]:
        """
        Search restaurants theo names hoặc IDs
        
        Args:
            restaurant_names: List of restaurant names
            restaurant_ids: List of restaurant IDs
            
        Returns:
            List of restaurant dicts
        """
        try:
            restaurants = []
            
            # 1. Search by IDs nếu có
            if restaurant_ids:
                restaurant_ids = [rid for rid in restaurant_ids if rid is not None]
                if restaurant_ids:
                    restaurants_by_ids = await self.vector_service.get_restaurants_by_ids(restaurant_ids)
                    restaurants.extend(restaurants_by_ids.values())
                    logger.info(f"Found {len(restaurants_by_ids)} restaurants by IDs")
            
            # 2. Search by names nếu có
            if restaurant_names:
                restaurant_names = [name for name in restaurant_names if name and name.strip()]
                if restaurant_names:
                    # Search từng name với threshold cao hơn (0.6) để catch variations
                    seen_ids = {self._extract_restaurant_id_from_metadata(r) for r in restaurants}
                    
                    for name in restaurant_names:
                        # Search với name + keywords để tăng accuracy
                        query = f"{name} nhà hàng restaurant"
                        results = await self.vector_service.search_restaurants(
                            query, limit=3, distance_threshold=0.6
                        )
                        
                        for result in results:
                            result_id = self._extract_restaurant_id_from_metadata(result)
                            # Check nếu name match (fuzzy)
                            result_name = result.get("restaurantName") or result.get("name", "")
                            if (result_id and result_id not in seen_ids and
                                (name.lower() in result_name.lower() or result_name.lower() in name.lower())):
                                seen_ids.add(result_id)
                                restaurants.append(result)
                                logger.debug(f"Found restaurant by name '{name}': {result_name} (ID: {result_id})")
            
            # Remove duplicates
            seen_ids = set()
            unique_restaurants = []
            for r in restaurants:
                r_id = self._extract_restaurant_id_from_metadata(r)
                if r_id and r_id not in seen_ids:
                    seen_ids.add(r_id)
                    unique_restaurants.append(r)
            
            logger.info(f"Found {len(unique_restaurants)} unique restaurants by names/IDs")
            return unique_restaurants
            
        except Exception as e:
            logger.error(f"Error searching restaurants by names/IDs: {e}", exc_info=True)
            return []
    
    def _normalize_restaurant_item(self, restaurant: Dict) -> Dict:
        """
        Normalize restaurant item để extract fields từ metadata nếu cần
        
        Tương tự _normalize_menu_item, flatten metadata và ensure fields accessible
        """
        if not restaurant:
            return restaurant
        
        # Nếu đã có fields ở top level → return as is
        if restaurant.get("restaurantName") or restaurant.get("name"):
            return restaurant
        
        # ✅ FIX: Extract từ metadata nếu cần
        metadata = restaurant.get("metadata", {})
        if metadata:
            # Flatten metadata fields lên top level
            restaurant_normalized = dict(restaurant)
            
            # Extract restaurant fields từ metadata
            if not restaurant_normalized.get("restaurantName") and not restaurant_normalized.get("name"):
                restaurant_normalized["restaurantName"] = (
                    metadata.get("restaurantName") 
                    or metadata.get("name")
                    or restaurant.get("name")
                    or "N/A"
                )
            
            if not restaurant_normalized.get("address"):
                restaurant_normalized["address"] = (
                    metadata.get("address")
                    or metadata.get("restaurantAddress")
                    or restaurant.get("address")
                    or "N/A"
                )
            
            if not restaurant_normalized.get("cuisineType"):
                restaurant_normalized["cuisineType"] = (
                    metadata.get("cuisineType")
                    or metadata.get("cuisine_type")
                    or metadata.get("restaurant_cuisine")
                    or restaurant.get("cuisineType")
                    or "N/A"
                )
            
            if not restaurant_normalized.get("rating"):
                restaurant_normalized["rating"] = (
                    metadata.get("rating")
                    or metadata.get("restaurantRating")
                    or restaurant.get("rating")
                    or "N/A"
                )
            
            # Preserve metadata for other uses
            restaurant_normalized["metadata"] = metadata
            
            return restaurant_normalized
        
        return restaurant
    
    async def _format_comparison_response(
        self, user_message: str, restaurants: List[Dict], user_id: str
    ) -> str:
        """
        Format response để so sánh restaurants
        
        Args:
            user_message: User message
            restaurants: List of restaurants to compare
            user_id: User ID
            
        Returns:
            Formatted comparison response
        """
        try:
            if not restaurants:
                return "Xin lỗi, tôi không tìm thấy thông tin về các nhà hàng bạn muốn so sánh."
            
            # ✅ FIX: Normalize restaurants trước khi format
            restaurants_to_compare = []
            for r in restaurants[:5]:  # Limit to top 5 restaurants
                normalized = self._normalize_restaurant_item(r)
                restaurants_to_compare.append(normalized)
            
            # Get menus cho các restaurants này
            restaurant_ids = [
                self._extract_restaurant_id_from_metadata(r) 
                for r in restaurants_to_compare
            ]
            restaurant_ids = [rid for rid in restaurant_ids if rid is not None]
            
            # Get menus for each restaurant
            menus_by_restaurant = {}
            for rid in restaurant_ids:
                menus = await self.vector_service.search_menus(
                    "", restaurant_id=rid, limit=10, distance_threshold=1.0  # High threshold để lấy tất cả
                )
                menus_by_restaurant[rid] = menus
            
            # Aggregate menus into restaurants
            for r in restaurants_to_compare:
                rid = self._extract_restaurant_id_from_metadata(r)
                if rid and rid in menus_by_restaurant:
                    r["_matchedMenus"] = menus_by_restaurant[rid]
            
            # Format với AI để so sánh
            return await self._format_multi_data_with_ai(
                user_message,
                restaurants=restaurants_to_compare,
                menus=[],  # Menus đã được aggregate vào restaurants
                services=[],
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Error formatting comparison response: {e}", exc_info=True)
            # Fallback to simple format
            return self._format_multi_data_fallback(restaurants, [], [])
    
    async def _handle_restaurant_search(
        self, user_message: str, entities: Dict, user_id: str
    ) -> str:
        """
        Restaurant search với reasoning layer
        
        Strategy:
        0. Detect follow-up questions và extract restaurants từ conversation history
        1. LLM Reasoning - Phân tích nhu cầu ăn uống (health, diet, constraints)
        2. Enhanced query - Dùng search_query từ reasoning thay vì raw user_message
        3. Semantic search với enhanced query
        4. Filter theo forbidden_tags (nếu có từ reasoning)
        """
        try:
            # ✅ STEP 0: Detect follow-up questions (so sánh, bạn vừa gợi ý, etc.)
            message_lower = user_message.lower()
            is_follow_up = any(keyword in message_lower for keyword in [
                "so sánh", "bạn vừa gợi ý", "những nhà hàng trên", "nhà hàng trên",
                "2 nhà hàng", "hai nhà hàng", "các nhà hàng", "những nhà hàng"
            ])
            
            if is_follow_up:
                logger.info(f"Detected follow-up question: {user_message[:100]}...")
                # Extract restaurants từ conversation history
                restaurants_from_history = await self._extract_restaurants_from_history(user_id)
                if restaurants_from_history:
                    logger.info(f"Found {len(restaurants_from_history)} restaurants from history: {[r.get('name', 'N/A') for r in restaurants_from_history]}")
                    # Search restaurants theo names/IDs từ history
                    restaurants = await self._search_restaurants_by_names_or_ids(
                        [r.get("name") for r in restaurants_from_history if r.get("name")],
                        [r.get("id") for r in restaurants_from_history if r.get("id")]
                    )
                    if restaurants:
                        # Format comparison response
                        return await self._format_comparison_response(
                            user_message, restaurants, user_id
                        )
                    else:
                        logger.warning(f"Could not find restaurants by names/IDs from history")
                        # Fallback to normal search
            
            # ✅ STEP 1: LLM Reasoning - Phân tích nhu cầu ăn uống
            reasoning_profile = await menu_reasoning_service.universal_query_reasoning(user_message)
            logger.info(
                f"Restaurant search reasoning: summary='{reasoning_profile.get('summary', 'N/A')}', "
                f"constraints_text={reasoning_profile.get('constraints_text', [])}, "
                f"diet_profile={reasoning_profile.get('diet_profile', {})}"
            )
            
            # ✅ STEP 2: Tạo enhanced query từ reasoning
            search_query = reasoning_profile.get("search_query", "").strip()
            summary = reasoning_profile.get("summary", "").strip()
            constraints_text = reasoning_profile.get("constraints_text", [])
            
            # Build enhanced query: search_query > summary > user_message
            if search_query:
                enhanced_query = search_query
                if summary and summary != search_query:
                    enhanced_query += f". {summary}"
            elif summary:
                enhanced_query = summary
            else:
                enhanced_query = user_message
            
            logger.debug(f"Restaurant search enhanced query: {enhanced_query[:100]}...")
            
            # ✅ STEP 3: Extract restaurant_id nếu có
            restaurant_id = entities.get("restaurant_id")
            
            # ✅ STEP 4: Detect collections cần query
            intent_result = {"intent": "restaurant_search"}
            collections = await self._detect_required_collections(
                user_message, intent_result
            )
            
            # ✅ STEP 5: Multi-collection search với enhanced query (thay vì raw user_message)
            search_results = await self._multi_collection_search(
                enhanced_query,  # ← Dùng enhanced query từ reasoning
                collections,
                distance_threshold=0.5,  # CHỈ lấy results gần
                limit_per_collection=20,
                restaurant_id=restaurant_id  # ✅ Pass restaurant_id để filter (None nếu generic search)
            )
            
            # ✅ STEP 6: Filter theo forbidden_tags (nếu có từ reasoning)
            forbidden_tags = self._extract_forbidden_tags(reasoning_profile)
            if forbidden_tags:
                logger.info(f"Filtering by forbidden_tags: {forbidden_tags}")
                search_results = self._filter_by_forbidden_tags(search_results, forbidden_tags)
            
            # ✅ STEP 7: Aggregate cross-collection data để liên kết món ăn/dịch vụ ↔ nhà hàng
            aggregated = await self._aggregate_search_results(search_results)
            restaurants_enriched = aggregated.get("restaurants", [])
            menus_enriched = aggregated.get("menus", [])
            services_enriched = aggregated.get("services", [])

            # ✅ STEP 8: Apply entity filters trên aggregated data
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

            # ✅ STEP 9: Nếu không có data → Không hallucinate
            if not restaurants_enriched and not menus_enriched and not services_enriched:
                return "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Bạn có thể thử tìm kiếm với từ khóa khác không?"

            # ✅ STEP 10: Format với AI - Đưa TẤT CẢ data vào
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
        """
        Menu inquiry với SEMANTIC REASONING
        
        ✅ MỚI: Dùng LLM reasoning để sinh structured profile
        ✅ MỚI: Dùng semantic_menu_search_with_reasoning() để search với reasoning
        """
        try:
            # ✅ STEP 1: LLM Reasoning - Sinh structured profile (thay vì keywords)
            reasoning_profile = await menu_reasoning_service.universal_query_reasoning(user_message)
            logger.info(
                f"Menu reasoning profile: summary='{reasoning_profile.get('summary', 'N/A')}', "
                f"constraints_text={reasoning_profile.get('constraints_text', [])}, "
                f"diet_profile={reasoning_profile.get('diet_profile', {})}, "
                f"search_query='{reasoning_profile.get('search_query', '')[:80]}...'"
            )
            
            # ✅ STEP 2: Find restaurant first to get restaurant_id
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
            
            # ✅ STEP 3: Semantic menu search với reasoning (PRIMARY METHOD)
            menus_enriched = await self.vector_service.semantic_menu_search_with_reasoning(
                user_message,
                reasoning_profile,
                restaurant_id=restaurant_id,
                limit=30,
                distance_threshold=0.6
            )
            
            # ✅ STEP 3.5: Filter theo forbidden_tags (nếu có từ reasoning)
            forbidden_tags = self._extract_forbidden_tags(reasoning_profile)
            if forbidden_tags:
                logger.info(f"Menu inquiry: Filtering by forbidden_tags: {forbidden_tags}")
                # Convert menus_enriched to search_results format để dùng filter
                filtered_search_results = {"menus": menus_enriched}
                filtered_search_results = self._filter_by_forbidden_tags(filtered_search_results, forbidden_tags)
                menus_enriched = filtered_search_results.get("menus", menus_enriched)
                logger.info(f"Menu inquiry: Filtered results: {len(menus_enriched)} menu items after filtering")
            
            # ✅ STEP 4: Get restaurant context nếu cần
            restaurants_enriched = []
            if restaurant_id:
                # Get restaurant details
                restaurants = await self.vector_service.get_restaurants_by_ids([restaurant_id])
                if restaurant_id in restaurants:
                    restaurants_enriched.append(restaurants[restaurant_id])
            elif menus_enriched:
                # Extract restaurant IDs từ menu results
                restaurant_ids = set()
                for menu in menus_enriched:
                    rid = menu.get("metadata", {}).get("restaurant_id")
                    if rid:
                        restaurant_ids.add(rid)
                
                if restaurant_ids:
                    restaurants = await self.vector_service.get_restaurants_by_ids(list(restaurant_ids))
                    restaurants_enriched = list(restaurants.values())
            
            # ✅ STEP 5: Format response với enriched data
            # ✅ FIX: Log để debug
            logger.info(f"Menu inquiry: Final menus_enriched count: {len(menus_enriched)}")
            
            if not menus_enriched:
                logger.warning(f"Menu inquiry: No menus found after reasoning and filtering")
                return "Xin lỗi, tôi không tìm thấy món ăn phù hợp với yêu cầu của bạn. Bạn có thể thử tìm kiếm với tiêu chí khác không?"

            # ✅ FIX: Ưu tiên dùng kết quả reasoning (menus_enriched đã có 16 món theo log)
            try:
                return await self._format_multi_data_with_ai(
                    user_message,
                    restaurants=restaurants_enriched,
                    menus=menus_enriched,  # ← Dùng kết quả reasoning (đã có 16 món)
                    services=[],  # Services không được search trong reasoning mode (có thể thêm sau)
                    user_id=user_id
                )
            except Exception as format_error:
                logger.error(f"Error formatting response with reasoning results: {format_error}", exc_info=True)
                # ✅ FIX: Nếu format fail nhưng vẫn có menus_enriched → thử format lại với fallback formatter
                return await self._format_multi_data_fallback(
                    restaurants=restaurants_enriched,
                    menus=menus_enriched,  # ← Vẫn dùng kết quả reasoning
                    services=[]
                )
            
        except Exception as e:
            logger.error(f"Error in _handle_menu_inquiry: {e}", exc_info=True)
            # ✅ FIX: Không fallback sang basic search nếu đã có kết quả reasoning
            # Chỉ fallback nếu thực sự không có gì
            # Fallback to old method nếu có lỗi nghiêm trọng (không phải lỗi format)
            try:
                # Fallback: Regular search không có reasoning (chỉ khi thực sự cần)
                restaurant_id = entities.get("restaurant_id")
                if restaurant_id:
                    logger.warning(f"Menu inquiry: Falling back to basic search for restaurant_id={restaurant_id}")
                    menus_enriched = await self.vector_service.search_menus(
                        user_message,
                        restaurant_id=restaurant_id,
                        limit=30,
                        distance_threshold=0.6
                    )
                    if menus_enriched:
                        logger.info(f"Menu inquiry: Fallback search found {len(menus_enriched)} menus")
                        return await self._format_multi_data_with_ai(
                            user_message,
                            restaurants=[],
                            menus=menus_enriched,
                            services=[],
                            user_id=user_id
                        )
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
            
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
        
        # ✅ FIX: Log để debug
        logger.info(f"_format_multi_data_with_ai: restaurants={len(restaurants)}, menus={len(menus)}, services={len(services)}")
        
        # Build STRICT data context cho TẤT CẢ loại data
        data_context_parts = []
        
        if restaurants:
            # Không limit vì đã filter theo distance - chỉ giới hạn token (max 20 restaurants)
            restaurants_to_show = restaurants[:20] if len(restaurants) > 20 else restaurants
            restaurant_lines: List[str] = []
            for index, r in enumerate(restaurants_to_show, 1):
                # ✅ FIX: Normalize restaurant item trước khi format
                r = self._normalize_restaurant_item(r)
                
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
                # ✅ FIX: Normalize menu item (extract metadata nếu có)
                dish = self._normalize_menu_item(dish)
                
                # Extract dish info với nhiều fallback options
                dish_name = (
                    dish.get("name") 
                    or dish.get("dishName") 
                    or dish.get("dish_name")
                    or "N/A"
                )
                
                price = dish.get("price")
                description = dish.get("description")
                
                # Restaurant name từ enriched fields hoặc metadata
                restaurant_name = (
                    dish.get("_restaurantName")  # Enriched từ aggregation
                    or dish.get("restaurantName")
                    or dish.get("restaurant_name")
                )
                
                # Category/type nếu có
                category = dish.get("category") or dish.get("dishCategory")
                
                # Tags nếu có (để hiển thị semantic context)
                tags = dish.get("tags", [])
                if isinstance(tags, str):
                    try:
                        import ast
                        tags = ast.literal_eval(tags) if tags.startswith("[") else [tags]
                    except:
                        tags = [tags] if tags else []
                
                line = f"- MÓN - {dish_name}"
                
                # Add restaurant context
                if restaurant_name:
                    line += f" (Nhà hàng: {restaurant_name})"
                
                # Add category nếu có
                if category:
                    line += f" [Loại: {category}]"
                
                # Add price
                if price in (None, "", "N/A"):
                    line += ": N/A"
                else:
                    price_str = str(price)
                    if "vnđ" not in price_str.lower() and "đ" not in price_str.lower() and price_str.strip():
                        price_str += " VNĐ"
                    line += f": {price_str}"
                
                # Add description
                if description:
                    trimmed_desc = description.strip()
                    if len(trimmed_desc) > 80:
                        trimmed_desc = trimmed_desc[:80].rstrip() + "..."
                    line += f" - {trimmed_desc}"
                
                # Add tags context nếu có (semantic boost info)
                if isinstance(tags, list) and tags:
                    tag_labels = []
                    tag_map = {
                        "high_protein": "Giàu protein",
                        "low_fat": "Ít béo",
                        "light_meal": "Món nhẹ",
                        "good_when_sick": "Tốt khi ốm",
                        "vegetarian": "Chay",
                        "spicy": "Cay"
                    }
                    for tag in tags[:3]:  # Chỉ hiển thị 3 tags đầu
                        if tag in tag_map:
                            tag_labels.append(tag_map[tag])
                    if tag_labels:
                        line += f" ({', '.join(tag_labels)})"
                
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
        
        # ✅ FIX: Log để debug
        logger.info(f"_format_multi_data_with_ai: data_context length={len(data_context)}, parts={len(data_context_parts)}")
        
        if not data_context:
            logger.warning(f"_format_multi_data_with_ai: No data_context generated (restaurants={len(restaurants)}, menus={len(menus)}, services={len(services)})")
            return "Không tìm thấy thông tin phù hợp."
        
        # STRICT System Prompt
        strict_prompt = f"""{self.strict_system_prompt}

DỮ LIỆU THỰC TẾ (CHỈ ĐƯỢC ĐỀ CẬP ĐẾN CÁC THÔNG TIN NÀY):
{data_context}

QUAN TRỌNG:
- CHỈ được đề cập đến thông tin trong danh sách trên
- KHÔNG được tự tạo tên nhà hàng, món ăn, dịch vụ, hoặc thông tin nào khác
- ✅ CÓ DỮ LIỆU TRONG DANH SÁCH → BẮT BUỘC phải đề cập đến ít nhất một số món/nhà hàng
- KHÔNG được trả "Không tìm thấy" nếu đã có dữ liệu trong danh sách
- Nếu user hỏi về thông tin không có trong danh sách → Nói "Trong danh sách hiện tại, tôi có thể gợi ý..." (KHÔNG nói "Không tìm thấy")
- Format response tự nhiên và tổng hợp các loại thông tin một cách hợp lý"""
        
        messages = [
            {"role": "system", "content": strict_prompt},
            {"role": "user", "content": user_message}
        ]
        
        history = self.conversations.get(user_id, [])
        if history:
            messages = [messages[0]] + history[-4:] + [messages[1]]
        
        response = await self._call_openai(messages)
        
        # ✅ FIX: Log response để debug
        logger.info(f"_format_multi_data_with_ai: LLM response length={len(response) if response else 0}")
        if response:
            logger.debug(f"_format_multi_data_with_ai: LLM response preview={response[:200]}...")
        
        # Fallback nếu AI không hoạt động
        if not response:
            logger.warning(f"_format_multi_data_with_ai: LLM returned None, using fallback formatter")
            return self._format_multi_data_fallback(restaurants, menus, services)
        
        # ✅ FIX: Nếu LLM trả "Không tìm thấy" nhưng có data → dùng fallback thay vì tin LLM
        if "không tìm thấy" in response.lower() and (restaurants or menus or services):
            logger.warning(
                f"_format_multi_data_with_ai: LLM returned 'Không tìm thấy' but have data "
                f"(restaurants={len(restaurants)}, menus={len(menus)}, services={len(services)}), "
                f"using fallback formatter"
            )
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
                # ✅ FIX: Normalize menu item (extract metadata nếu có)
                dish = self._normalize_menu_item(dish)
                
                dish_name = (
                    dish.get("name") 
                    or dish.get("dishName") 
                    or dish.get("dish_name")
                    or "N/A"
                )
                price = dish.get("price")
                restaurant_name = (
                    dish.get("_restaurantName")
                    or dish.get("restaurantName")
                    or dish.get("restaurant_name")
                )
                description = dish.get("description")
                
                line = f"• **{dish_name}**"
                if restaurant_name:
                    line += f" (Nhà hàng: {restaurant_name})"
                if price in (None, "", "N/A"):
                    line += " - N/A"
                else:
                    price_str = str(price)
                    if "vnđ" not in price_str.lower() and "đ" not in price_str.lower() and price_str.strip():
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
                response = completion.choices[0].message.content.strip()
                # ✅ FIX: Log response để debug
                logger.info(f"_call_openai: Response length={len(response)}, preview={response[:150]}...")
                return response
            logger.warning("_call_openai: No choices in completion")
            return None

        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
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
