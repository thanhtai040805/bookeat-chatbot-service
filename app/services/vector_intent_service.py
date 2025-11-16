import re
import logging
import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from app.services.vector_service import vector_service
from app.core.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

class VectorIntentService:
    """Auto Intent Recognition Service với Intent Embedding Collection + LLM + Verification"""
    
    def __init__(self):
        self.vector_service = vector_service
        self.openai_client = None
        self.intent_feedback_dataset = []  # Store feedback để học sau
        
        # Intent definitions với API functions
        # CHỈ CẦN description - LLM sẽ tự classify, không cần hardcode patterns
        self.intent_definitions = {
            "restaurant_search": {
                "api_function": "search_restaurants",
                "confidence": 0.9,
                "description": "User tìm nhà hàng. Ví dụ: 'nhà hàng nào', 'tìm quán ăn', 'recommend restaurant', 'có quán ăn không'"
            },
            "menu_inquiry": {
                "api_function": "get_restaurant_menu",
                "confidence": 0.9,
                "description": "User hỏi về menu, thực đơn, món ăn. Ví dụ: 'menu là gì', 'có món gì', 'thực đơn', 'dish'"
            },
            "table_inquiry": {
                "api_function": "get_tables",
                "confidence": 0.85,
                "description": "User hỏi về bàn, sơ đồ bàn, layout. Ví dụ: 'bàn nào', 'sơ đồ bàn', 'loại bàn', 'sức chứa', 'layout'"
            },
            "voucher_inquiry": {
                "api_function": "get_demo_vouchers",
                "confidence": 0.8,
                "description": "User hỏi về voucher, khuyến mãi, mã giảm giá. Ví dụ: 'voucher', 'khuyến mãi', 'giảm giá', 'discount', 'mã'"
            },
            "general_inquiry": {
                "api_function": None,
                "confidence": 0.5,
                "description": "Truy vấn chung, không rõ ý định"
            }
        }
        
        # Fallback patterns - CHỈ dùng khi LLM không available (fallback)
        # KHÔNG CẦN UPDATE THỦ CÔNG - LLM sẽ tự handle
        self.fallback_patterns = {
            "restaurant_search": [
                r"tìm nhà hàng", r"nhà hàng", r"restaurant", r"địa điểm ăn", r"chỗ ăn", 
                r"quán ăn", r"muốn ăn", r"đi ăn", r"ăn"
            ],
            "menu_inquiry": [
                r"thực đơn", r"menu", r"món ăn", r"có gì ăn", r"món"
            ],
        }
        
        # General fallback patterns
        self.general_fallback_patterns = [
            r"xin chào", r"hello", r"hi", r"chào", r"help",
            r"giá cả", r"pricing", r"thông tin", r"info"
        ]
        
        # Intent embeddings sẽ được init khi gọi initialize_intent_embeddings()
    
    async def recognize_intent_with_context(self, user_message: str, user_id: str = None) -> Dict[str, Any]:
        """
        Auto Intent Recognition với Intent Embedding Collection FIRST + LLM + Vector + Pattern
        
        Strategy (NEW - Optimized):
        1. Intent Embedding Collection (FASTEST - 1-2ms, không tốn token)
        2. LLM Classification (nếu intent collection không chắc chắn)
        3. Vector Search (fallback)
        4. Pattern Matching (last resort)
        
        Args:
            user_message: Tin nhắn từ user
            user_id: ID của user (optional)
            
        Returns:
            Dict với intent, confidence, api_function, context
        """
        try:
            # 1. Get conversation context - ✅ TỐI ƯU: Lấy recent conversations theo timestamp thay vì semantic search
            context = ""
            if user_id:
                # ✅ TỐI ƯU: Lấy recent conversations theo timestamp (nhanh hơn, không cần semantic search)
                recent_conversations = await self.vector_service.get_user_conversations_recent(
                    user_id=user_id,
                    limit=5  # Lấy 5 conversations gần nhất
                )
                
                if recent_conversations:
                    context = "Previous conversations:\n" + "\n".join([
                        conv['document'] for conv in recent_conversations[:3]  # Chỉ lấy 3 gần nhất cho context
                    ])
                    logger.info(f"Loaded {len(recent_conversations)} recent conversations for context")
            
            # 2. Intent Embedding Collection Classification (FASTEST - ưu tiên đầu tiên)
            intent_collection_result = await self._vector_intent_classify(user_message)
            
            # 3. Nếu intent collection có confidence cao → dùng luôn
            if intent_collection_result.get("confidence", 0) >= 0.8:
                logger.info(f"Using Intent Collection: {intent_collection_result['intent']} (confidence: {intent_collection_result['confidence']})")
                final_intent = intent_collection_result
            else:
                # 4. LLM-based classification (nếu intent collection không chắc)
                llm_intent = await self._llm_based_classification(user_message, context)
                
                # 5. Vector-based recognition (fallback)
                vector_intent = await self._vector_based_recognition(user_message, context, user_id)  # ✅ FIX: Thêm user_id
                
                # 6. Pattern-based recognition (last resort)
            pattern_intent = self._pattern_based_recognition(user_message)
            
            # 7. Combine results - Ưu tiên Intent Collection > LLM > Vector > Pattern
            final_intent = self._combine_intent_results_priority(
                intent_collection_result, llm_intent, vector_intent, pattern_intent
            )
            
            # 7.1. Early return nếu LLM đã chắc chắn - tránh pattern override
            if (final_intent.get("method") == "llm_classification" and 
                final_intent.get("confidence", 0) >= 0.6):
                logger.info(f"[LOCKED] Using LLM classification only: {final_intent['intent']} (confidence: {final_intent['confidence']})")
                final_intent["context"] = context
                final_intent["enhanced_message"] = f"{context}\nCurrent message: {user_message}"
                logger.info(f"Auto Intent Recognition: {final_intent['intent']} (confidence: {final_intent['confidence']}, method: {final_intent.get('method', 'unknown')})")
                return final_intent
            
            # 8. Intent Verification (ReAct-like) - Kiểm tra intent có phù hợp với data thực tế không
            verification_result = await self._verify_intent_with_data(
                final_intent, user_message, user_id
            )
            
            # 9. Nếu verification fail → adjust intent
            if not verification_result.get("intent_valid", True):
                logger.warning(f"Intent verification failed: {final_intent['intent']} -> {verification_result.get('suggest_intent')}")
                if verification_result.get("suggest_intent"):
                    final_intent["intent"] = verification_result["suggest_intent"]
                    final_intent["confidence"] = verification_result.get("suggest_confidence", 0.5)
                    final_intent["method"] = "verified_adjusted"
            
            # 10. Add context to result
            final_intent["context"] = context
            final_intent["enhanced_message"] = f"{context}\nCurrent message: {user_message}"
            
            logger.info(f"Auto Intent Recognition: {final_intent['intent']} (confidence: {final_intent['confidence']}, method: {final_intent.get('method', 'unknown')})")
            return final_intent
            
        except Exception as e:
            logger.error(f"Error in intent recognition: {e}", exc_info=True)
            # Fallback to pattern-based recognition
            fallback_result = self._pattern_based_recognition(user_message)
            fallback_result["context"] = ""
            fallback_result["enhanced_message"] = user_message
            fallback_result["method"] = "pattern_fallback_error"
            return fallback_result
    
    async def _vector_intent_classify(self, user_message: str) -> Dict[str, Any]:
        """
        Intent classification sử dụng Intent Embedding Collection
        
        Ưu điểm: Nhanh (1-2ms), không tốn token, có thể retrain dễ dàng
        
        Args:
            user_message: Tin nhắn từ user
            
        Returns:
            Dict với intent, confidence, api_function
        """
        try:
            results = await self.vector_service.search_intents(
                user_message, limit=1, distance_threshold=0.4
            )
            
            if results and results[0]['distance'] < 0.4:
                metadata = results[0]['metadata']
                distance = results[0]['distance']
                
                # Convert distance to confidence
                confidence = 1 - distance  # distance càng nhỏ → confidence càng cao
                confidence = min(confidence * 0.95, 0.98)  # Cap at 0.98
                
                intent_name = metadata.get('intent', 'general_inquiry')
                intent_def = self.intent_definitions.get(intent_name, {})
                
                return {
                    "intent": intent_name,
                    "confidence": confidence,
                    "api_function": metadata.get('api_function') or intent_def.get("api_function"),
                    "method": "intent_collection",
                    "distance": distance
                }
            
            # Không tìm thấy trong intent collection
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "api_function": None,
                "method": "intent_collection_no_match"
            }
            
        except Exception as e:
            logger.error(f"Error in vector intent classification: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "api_function": None,
                "method": "intent_collection_error"
            }
    
    async def _verify_intent_with_data(
        self, intent_result: Dict[str, Any], user_message: str, user_id: str = None
    ) -> Dict[str, Any]:
        """
        Intent Verification Stage (ReAct-like)
        
        Kiểm tra intent có phù hợp với data thực tế không.
        Ví dụ: Nếu intent là restaurant_search nhưng không có restaurant nào trong DB
        → Trả về intent_valid=False, suggest_intent="general_inquiry"
        
        Args:
            intent_result: Intent đã được recognize
            user_message: User message
            user_id: User ID
            
        Returns:
            Dict với intent_valid, suggest_intent (nếu cần)
        """
        try:
            intent = intent_result.get("intent")
            entities = intent_result.get("entities", {})
            user_message_lower = user_message.lower()
            
            # Semantic verification - Kiểm tra logic ngữ nghĩa
            if intent == "restaurant_search":
                # Nếu user hỏi về "menu" → suggest menu_inquiry
                if any(kw in user_message_lower for kw in ['menu', 'thực đơn', 'món', 'có gì ăn']):
                    return {
                        "intent_valid": False,
                        "suggest_intent": "menu_inquiry",
                        "suggest_confidence": 0.8,
                        "reason": "User asking about menu, not restaurant search"
                    }
                
                # Check xem có restaurant nào trong DB không
                restaurant_results = await self.vector_service.search_restaurants(
                    user_message, limit=1, distance_threshold=0.6
                )
                
                if not restaurant_results:
                    return {
                        "intent_valid": False,
                        "suggest_intent": "general_inquiry",
                        "suggest_confidence": 0.5,
                        "reason": "No restaurants found in database"
                    }
            
            elif intent == "menu_inquiry":
                # Nếu user hỏi về "nhà hàng" mà không có menu keywords → suggest restaurant_search
                if (any(kw in user_message_lower for kw in ['nhà hàng', 'restaurant', 'quán']) and
                    not any(kw in user_message_lower for kw in ['menu', 'thực đơn', 'món', 'có gì ăn'])):
                    return {
                        "intent_valid": False,
                        "suggest_intent": "restaurant_search",
                        "suggest_confidence": 0.7,
                        "reason": "User asking about restaurant, not menu"
                    }
                
                # Check xem có menu nào không
                menu_results = await self.vector_service.search_menus(
                    user_message, limit=1, distance_threshold=0.6
                )
                
                if not menu_results:
                    return {
                        "intent_valid": False,
                        "suggest_intent": "general_inquiry",
                        "suggest_confidence": 0.5,
                        "reason": "No menu items found in database"
                    }
            
            # Intent hợp lệ
            return {
                "intent_valid": True,
                "reason": "Intent verified with actual data"
            }
            
        except Exception as e:
            logger.error(f"Error in intent verification: {e}")
            return {
                "intent_valid": True,  # Default to valid nếu có lỗi
                "reason": "Verification error"
            }
    
    async def _initialize_intents_async(self):
        """Initialize intent embeddings trong Vector DB"""
        try:
            await self.vector_service.initialize_intent_embeddings(self.intent_definitions)
        except Exception as e:
            logger.error(f"Error initializing intent embeddings: {e}")
    
    async def initialize_intent_embeddings(self):
        """Public method để init intent embeddings"""
        await self._initialize_intents_async()
    
    async def _llm_based_classification(self, user_message: str, context: str = "") -> Dict[str, Any]:
        """
        LLM-based intent classification - TỰ ĐỘNG, không cần hardcode patterns
        
        Args:
            user_message: Tin nhắn từ user
            context: Conversation context
            
        Returns:
            Dict với intent, confidence, api_function
        """
        try:
            # Initialize OpenAI client nếu chưa có
            if not self.openai_client:
                if settings.OPENAI_API_KEY:
                    self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                else:
                    logger.warning("OpenAI API key not available, skipping LLM classification")
                    return {
                        "intent": "general_inquiry",
                        "confidence": 0.0,
                        "api_function": None,
                        "method": "llm_unavailable"
                    }
            
            # Build intent descriptions for LLM
            intent_descriptions = "\n".join([
                f"- {intent_name}: {defn['description']}"
                for intent_name, defn in self.intent_definitions.items()
            ])
            
            # System prompt cho classification
            classification_prompt = f"""Bạn là một hệ thống phân loại intent cho chatbot nhà hàng.

Nhiệm vụ: Phân loại user message vào một trong các intent sau:

{intent_descriptions}

Hãy phân tích user message và trả về JSON với format:
{{
    "intent": "tên_intent",
    "confidence": 0.0-1.0,
    "reasoning": "lý do tại sao chọn intent này"
}}

Lưu ý:
- Chọn intent phù hợp nhất với ý định của user
- Confidence phải từ 0.0 đến 1.0 (0.9+ nếu chắc chắn, 0.7-0.8 nếu khá chắc, 0.5-0.6 nếu không chắc)
- Nếu không chắc → chọn "general_inquiry" với confidence thấp"""
            
            messages = [
                {"role": "system", "content": classification_prompt}
            ]
            
            # Add context nếu có
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Context từ previous conversations:\n{context}"
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Call OpenAI
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.1,  # Low temperature để classification chính xác
                max_tokens=200,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            if response.choices:
                content = response.choices[0].message.content.strip()
                try:
                    result = json.loads(content)
                    intent_name = result.get("intent", "general_inquiry")
                    confidence = float(result.get("confidence", 0.5))
                    
                    # Validate intent
                    if intent_name not in self.intent_definitions:
                        logger.warning(f"LLM returned unknown intent: {intent_name}, using general_inquiry")
                        intent_name = "general_inquiry"
                    
                    intent_def = self.intent_definitions[intent_name]
                    
                    return {
                        "intent": intent_name,
                        "confidence": confidence,
                        "api_function": intent_def["api_function"],
                        "method": "llm_classification",
                        "reasoning": result.get("reasoning", "")
                    }
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {content}")
                    return {
                        "intent": "general_inquiry",
                        "confidence": 0.0,
                        "api_function": None,
                        "method": "llm_parse_error"
                    }
            
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "api_function": None,
                "method": "llm_no_response"
            }
            
        except Exception as e:
            logger.error(f"Error in LLM-based classification: {e}", exc_info=True)
            return {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "api_function": None,
                "method": "llm_error"
            }
    
    def _combine_intent_results_priority(
        self, 
        intent_collection_result: Dict[str, Any],
        llm_intent: Dict[str, Any],
        vector_intent: Dict[str, Any],
        pattern_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine results với priority: Intent Collection > LLM > Vector > Pattern
        
        ✅ CẢI THIỆN: Ưu tiên pattern nếu LLM trả general_inquiry (tránh học sai)
        
        Strategy:
        1. Intent Collection (confidence >= 0.8) → ưu tiên cao nhất
        2. LLM classification (confidence >= 0.5) → NHƯNG nếu LLM trả general_inquiry mà pattern có giá trị → ưu tiên pattern
        3. Pattern matching (nếu LLM confidence < 0.7 và pattern có giá trị) → ưu tiên pattern
        4. Vector search
        5. Fallback theo priority
        """
        # Priority 1: Intent Collection (nếu confidence >= 0.8)
        if intent_collection_result.get("confidence", 0) >= 0.8:
            logger.info(f"Using Intent Collection: {intent_collection_result['intent']} (confidence: {intent_collection_result['confidence']})")
            return intent_collection_result
        
        llm_intent_name = llm_intent.get("intent", "general_inquiry")
        llm_conf = llm_intent.get("confidence", 0)
        pattern_intent_name = pattern_intent.get("intent", "general_inquiry")
        pattern_conf = pattern_intent.get("confidence", 0)
        
        # ✅ FIX: Nếu LLM trả general_inquiry mà pattern đã match một intent cụ thể → ưu tiên pattern
        if (llm_intent_name == "general_inquiry" and 
            pattern_intent_name != "general_inquiry" and 
            pattern_conf >= 0.5):
            logger.info(
                f"LLM returned general_inquiry (conf={llm_conf}), but pattern matched {pattern_intent_name} "
                f"(conf={pattern_conf}) → Using pattern"
            )
            # Tăng confidence của pattern để tránh bị override
            pattern_intent["confidence"] = max(pattern_conf, 0.7)
            pattern_intent["method"] = "pattern_override_llm_general"
            return pattern_intent
        
        # ✅ FIX: Nếu LLM confidence < 0.7 mà pattern có giá trị → ưu tiên pattern
        if (llm_conf < 0.7 and 
            pattern_intent_name != "general_inquiry" and 
            pattern_conf >= 0.5):
            logger.info(
                f"LLM confidence ({llm_conf}) < 0.7, but pattern matched {pattern_intent_name} "
                f"(conf={pattern_conf}) → Using pattern"
            )
            pattern_intent["confidence"] = max(pattern_conf, 0.7)
            pattern_intent["method"] = "pattern_override_llm_low_confidence"
            return pattern_intent
        
        # Priority 2: LLM classification (ưu tiên cao vì hiểu context - giảm threshold xuống 0.5)
        if llm_conf >= 0.5:
            logger.info(f"Using LLM classification: {llm_intent_name} (confidence: {llm_conf})")
            return llm_intent
        
        # Priority 3: Pattern matching (nếu có giá trị và LLM confidence thấp)
        if pattern_conf >= 0.5:
            logger.info(f"Using Pattern matching: {pattern_intent_name} (confidence: {pattern_conf})")
            return pattern_intent
        
        # Priority 4: Vector search (nếu confidence >= 0.6)
        if vector_intent.get("confidence", 0) >= 0.6:
            logger.info(f"Using Vector search: {vector_intent['intent']} (confidence: {vector_intent['confidence']})")
            return vector_intent
        
        # Priority 5: Intent Collection (nếu có, dù confidence thấp)
        if intent_collection_result.get("confidence", 0) > 0:
            return intent_collection_result
        
        # Priority 6: LLM (nếu có, dù confidence thấp)
        if llm_conf > 0:
            return llm_intent
        
        # Priority 7: Vector (nếu có, dù confidence thấp)
        if vector_intent.get("confidence", 0) > 0:
            return vector_intent
        
        # Fallback: Pattern matching
        return pattern_intent
    
    async def _vector_based_recognition(self, user_message: str, context: str = "", user_id: str = None) -> Dict[str, Any]:
        """
        Vector-based intent recognition sử dụng semantic search
        
        Args:
            user_message: Tin nhắn từ user
            context: Conversation context
            user_id: User ID để filter conversations (QUAN TRỌNG - privacy)
            
        Returns:
            Dict với intent và confidence
        """
        try:
            # 1. Search restaurants - Tăng limit và giảm distance threshold để catch nhiều hơn
            restaurant_results = await self.vector_service.search_restaurants(
                user_message, limit=5, distance_threshold=0.6  # Tăng threshold để catch nhiều hơn
            )
            
            # 2. Search menus - Tăng limit và giảm distance threshold
            menu_results = await self.vector_service.search_menus(
                user_message, limit=5, distance_threshold=0.6  # Tăng threshold để catch nhiều hơn
            )
            
            # 3. Search conversations - QUAN TRỌNG: Phải filter theo user_id để tránh leak data
            conversation_results = await self.vector_service.search_similar_conversations(
                user_message, user_id=user_id, limit=3  # ✅ FIX: Thêm user_id
            )
            
            # 4. Find best match
            best_match = self._find_best_vector_match(
                restaurant_results, menu_results, conversation_results, user_message
            )
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error in vector-based recognition: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.3,
                "api_function": None,
                "matched_pattern": "vector_error"
            }
    
    def _find_best_vector_match(self, restaurant_results: List, menu_results: List, 
                               conversation_results: List, user_message: str) -> Dict[str, Any]:
        """Find best intent match từ vector search results"""
        try:
            best_match = None
            best_score = 0
            
            # Check restaurant results - Giảm threshold để catch nhiều hơn
            if restaurant_results and restaurant_results[0]['distance'] < 0.6:  # Tăng từ 0.4 lên 0.6
                score = 1 - restaurant_results[0]['distance']  # Convert distance to score
                if score > best_score:
                    best_match = {
                        "intent": "restaurant_search",
                        "confidence": min(score * 0.9, 0.95),  # Cap confidence
                        "api_function": "search_restaurants",
                        "matched_pattern": "vector_search_restaurant",
                        "vector_data": restaurant_results[0]
                    }
                    best_score = score
            
            # Check menu results - Giảm threshold để catch nhiều hơn
            if menu_results and menu_results[0]['distance'] < 0.6:  # Tăng từ 0.4 lên 0.6
                score = 1 - menu_results[0]['distance']
                if score > best_score:
                    best_match = {
                        "intent": "menu_inquiry",
                        "confidence": min(score * 0.9, 0.95),
                        "api_function": "get_restaurant_menu",
                        "matched_pattern": "vector_search_menu",
                        "vector_data": menu_results[0]
                    }
                    best_score = score
            
            # Check conversation results for intent patterns
            if conversation_results:
                for conv in conversation_results:
                    if conv['distance'] < 0.3:
                        # Extract intent from conversation metadata
                        intent = conv['metadata'].get('intent')
                        if intent and intent in self.intent_definitions:
                            score = 1 - conv['distance']
                            if score > best_score:
                                intent_def = self.intent_definitions[intent]
                                best_match = {
                                    "intent": intent,
                                    "confidence": min(score * 0.8, 0.9),
                                    "api_function": intent_def["api_function"],
                                    "matched_pattern": "vector_conversation_pattern",
                                    "vector_data": conv
                                }
                                best_score = score
            
            # Check for restaurant-related semantic patterns (QUAN TRỌNG - phải check trước)
            if not best_match or best_score < 0.5:
                restaurant_score = self._semantic_restaurant_detection(user_message)
                if restaurant_score > best_score:
                    best_match = {
                        "intent": "restaurant_search",
                        "confidence": restaurant_score,
                        "api_function": "search_restaurants",
                        "matched_pattern": "semantic_restaurant_detection"
                    }
                    best_score = restaurant_score
            
            
            
            return best_match or {
                "intent": "general_inquiry",
                "confidence": 0.3,
                "api_function": None,
                "matched_pattern": "vector_no_match"
            }
            
        except Exception as e:
            logger.error(f"Error finding best vector match: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.3,
                "api_function": None,
                "matched_pattern": "vector_error"
            }
    
    def _semantic_restaurant_detection(self, user_message: str) -> float:
        """Semantic detection cho restaurant search intent - QUAN TRỌNG"""
        restaurant_keywords = [
            # Từ khóa tìm kiếm
            "tìm", "find", "search", "look for",
            # Từ khóa ăn uống
            "ăn", "eat", "food", "restaurant", "nhà hàng", "quán",
            "đồ ăn", "món ăn", "ẩm thực", "cuisine",
            # Từ khóa muốn/đi
            "muốn ăn", "want to eat", "đi ăn", "go eat",
            "hôm nay", "today", "tối nay", "tonight",
            # Từ khóa loại ẩm thực
            "hàn", "korean", "việt", "vietnamese", "ý", "italian",
            "nhật", "japanese", "trung", "chinese", "thái", "thai",
            "châu á", "asian", "tây", "western", "european",
            # Từ khóa địa điểm
            "gần đây", "nearby", "ở đâu", "where",
            "chỗ ăn", "địa điểm ăn", "place to eat"
        ]
        
        message_lower = user_message.lower()
        
        # Check các keyword combinations
        matches = 0
        
        # Combination 1: muốn ăn + loại ẩm thực
        if any(kw in message_lower for kw in ["muốn ăn", "want to eat", "đi ăn", "go eat"]):
            matches += 2
            if any(kw in message_lower for kw in ["hàn", "korean", "việt", "vietnamese", "ý", "italian", 
                                                  "nhật", "japanese", "trung", "chinese", "thái", "thai",
                                                  "châu á", "asian"]):
                matches += 3  # Strong match
        
        # Combination 2: đồ ăn + loại ẩm thực
        if any(kw in message_lower for kw in ["đồ ăn", "food", "món ăn", "cuisine"]):
            matches += 1
            if any(kw in message_lower for kw in ["hàn", "korean", "việt", "vietnamese", "ý", "italian"]):
                matches += 2
        
        # Combination 3: tìm + nhà hàng/restaurant
        if any(kw in message_lower for kw in ["tìm", "find", "search"]):
            if any(kw in message_lower for kw in ["nhà hàng", "restaurant", "quán", "chỗ ăn"]):
                matches += 3
        
        # Combination 4: hôm nay/tối nay + muốn ăn
        if any(kw in message_lower for kw in ["hôm nay", "today", "tối nay", "tonight"]):
            if any(kw in message_lower for kw in ["muốn ăn", "đi ăn", "ăn"]):
                matches += 2
        
        # Single keyword matches
        single_matches = sum(1 for keyword in restaurant_keywords if keyword in message_lower)
        matches += single_matches * 0.5
        
        # Calculate score
        if matches >= 3:
            return min(0.75 + (matches - 3) * 0.05, 0.95)  # 0.75-0.95 for strong matches
        elif matches >= 2:
            return min(0.65 + (matches - 2) * 0.05, 0.8)   # 0.65-0.8 for medium matches
        elif matches >= 1:
            return 0.5 + (matches - 1) * 0.1               # 0.5-0.6 for weak matches
        
        return 0.0
    
    
    
    def _pattern_based_recognition(self, user_message: str) -> Dict[str, Any]:
        """Pattern-based intent recognition (Fallback - chỉ khi LLM không available)"""
        try:
            user_message_lower = user_message.lower()
            
            # Check fallback patterns (chỉ basic patterns, LLM sẽ handle phức tạp hơn)
            for intent_name, patterns in self.fallback_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, user_message_lower):
                        logger.info(f"Fallback pattern matched: {intent_name} with pattern: {pattern}")
                        intent_def = self.intent_definitions.get(intent_name, {})
                        return {
                            "intent": intent_name,
                            "confidence": intent_def.get("confidence", 0.7) * 0.7,  # Lower confidence for fallback
                            "api_function": intent_def.get("api_function"),
                            "matched_pattern": pattern,
                            "method": "pattern_fallback"
                        }
            
            # Check general fallback patterns
            for pattern in self.general_fallback_patterns:
                if re.search(pattern, user_message_lower):
                    logger.info(f"General fallback pattern matched: {pattern}")
                    return {
                        "intent": "general_inquiry",
                        "confidence": 0.6,
                        "api_function": None,
                        "matched_pattern": pattern,
                        "method": "pattern_fallback"
                    }
            
            # Default fallback
            logger.info("No pattern matched, using default fallback")
            return {
                "intent": "general_inquiry",
                "confidence": 0.3,
                "api_function": None,
                "matched_pattern": None,
                "method": "pattern_fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in pattern-based recognition: {e}")
            return {
                "intent": "general_inquiry",
                "confidence": 0.3,
                "api_function": None,
                "matched_pattern": "pattern_error",
                "method": "pattern_error"
            }
    
    def _combine_intent_results(self, vector_intent: Dict[str, Any], 
                               pattern_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Combine vector và pattern intent results"""
        try:
            # If vector intent has high confidence, use it
            if vector_intent["confidence"] >= 0.7:
                return vector_intent
            
            # If pattern intent has higher confidence, use it
            if pattern_intent["confidence"] > vector_intent["confidence"]:
                return pattern_intent
            
            # If both have similar confidence, prefer vector intent
            if abs(vector_intent["confidence"] - pattern_intent["confidence"]) < 0.1:
                return vector_intent
            
            # Otherwise use the one with higher confidence
            return vector_intent if vector_intent["confidence"] > pattern_intent["confidence"] else pattern_intent
            
        except Exception as e:
            logger.error(f"Error combining intent results: {e}")
            return pattern_intent  # Fallback to pattern intent
    
    async def extract_entities_with_context(self, user_message: str, intent: str, 
                                          context: str = "") -> Dict[str, Any]:
        """
        Enhanced entity extraction với Vector Database context
        
        Args:
            user_message: Tin nhắn từ user
            intent: Recognized intent
            context: Conversation context
            
        Returns:
            Dict chứa extracted entities
        """
        try:
            # 1. LLM-based entity extraction (Primary for complex intents)
            llm_entities = await self._llm_entity_extraction(user_message, intent, context)
            
            # 2. Vector-based entity extraction (Support)
            vector_entities = await self._vector_entity_extraction(user_message, intent)
            
            # 3. Pattern-based entity extraction (Support)
            pattern_entities = self._pattern_entity_extraction(user_message, intent)
            
            # 4. Context-based entity extraction
            context_entities = self._context_entity_extraction(context, intent)
            
            # 5. Combine all entities - LLM takes priority
            final_entities = {**pattern_entities, **context_entities, **vector_entities, **llm_entities}
            
            logger.info(f"Enhanced entities extracted for intent '{intent}': {final_entities} (LLM: {llm_entities})")
            return final_entities
            
        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            return self._pattern_entity_extraction(user_message, intent)
    
    async def _vector_entity_extraction(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Vector-based entity extraction"""
        try:
            entities = {}
            
            if intent == "restaurant_search":
                # Search restaurants để extract entities
                restaurant_results = await self.vector_service.search_restaurants(user_message, limit=1)
                
                if restaurant_results and restaurant_results[0]['distance'] < 0.4:
                    restaurant_metadata = restaurant_results[0]['metadata']
                    
                    # Extract cuisine type
                    if restaurant_metadata.get('cuisineType'):
                        entities['cuisine_type'] = restaurant_metadata['cuisineType'].lower()
                    
                    # Extract location
                    if restaurant_metadata.get('address'):
                        entities['location'] = restaurant_metadata['address']
                    
                    # Extract restaurant ID
                    if restaurant_metadata.get('id'):
                        entities['restaurant_id'] = restaurant_metadata['id']
            
            elif intent == "menu_inquiry":
                # Search menus để extract entities
                menu_results = await self.vector_service.search_menus(user_message, limit=1)
                
                if menu_results and menu_results[0]['distance'] < 0.4:
                    menu_metadata = menu_results[0]['metadata']
                    
                    # Extract restaurant ID
                    if menu_metadata.get('restaurant_id'):
                        entities['restaurant_id'] = menu_metadata['restaurant_id']
                    
                    # Extract dish category
                    if menu_metadata.get('category'):
                        entities['dish_category'] = menu_metadata['category']
            
            elif intent == "table_inquiry":
                # Search tables để extract entities
                table_results = await self.vector_service.search_tables(user_message, limit=1)
                
                if table_results and table_results[0]['distance'] < 0.4:
                    table_metadata = table_results[0]['metadata']
                    
                    # Extract restaurant ID
                    if table_metadata.get('restaurant_id'):
                        entities['restaurant_id'] = table_metadata['restaurant_id']
                    
                    # Extract table type
                    if table_metadata.get('tableType'):
                        entities['table_type'] = table_metadata['tableType']
            
            elif intent == "voucher_inquiry":
                # Search vouchers để extract entities
                voucher_results = await self.vector_service.search_demo_vouchers(user_message, limit=1)
                
                if voucher_results and voucher_results[0]['distance'] < 0.4:
                    voucher_metadata = voucher_results[0]['metadata']
                    
                    # Extract voucher code
                    if voucher_metadata.get('code'):
                        entities['voucher_code'] = voucher_metadata['code']
            
            
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in vector entity extraction: {e}")
            return {}
    
    def _pattern_entity_extraction(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Pattern-based entity extraction"""
        try:
            entities = {}
            user_message_lower = user_message.lower()
            
            if intent == "restaurant_search":
                # Extract cuisine type
                cuisine_patterns = {
                    "vietnamese": r"việt nam|vietnamese|phở|bún|gỏi",
                    "japanese": r"nhật|japanese|sushi|sashimi|ramen",
                    "korean": r"hàn|korean|kimchi|bbq",
                    "chinese": r"trung|chinese|dim sum|wok",
                    "italian": r"ý|italian|pizza|pasta",
                    "thai": r"thái|thai|tom yum|pad thai"
                }
                
                for cuisine, pattern in cuisine_patterns.items():
                    if re.search(pattern, user_message_lower):
                        entities["cuisine_type"] = cuisine
                        break
                
                # Extract location - Fix regex để tránh bắt "nh" từ "gần nhất"
                location_pattern = r"(?:ở|tại|gần)\s+([a-zA-ZÀ-ỹ\s]{3,})"
                location_match = re.search(location_pattern, user_message_lower)
                if location_match:
                    location = location_match.group(1).strip()
                    # Filter out common short words
                    if len(location) >= 3 and location.lower() not in ['nh', 'nhà', 'có', 'là', 'nào']:
                        entities["location"] = location
            
            elif intent in ["table_inquiry"]:
                # Extract restaurant name patterns
                restaurant_patterns = [
                    r"nhà hàng\s+([A-Za-z\s]+?)(?:\s|có|tại|ở|nào)",
                    r"([A-Za-z\s]+?)\s+có\s+bàn",
                    r"([A-Za-z\s]+?)\s+availability",
                    r"([A-Za-z\s]+?)\s+đặt\s+bàn"
                ]
                
                for pattern in restaurant_patterns:
                    match = re.search(pattern, user_message, re.IGNORECASE)
                    if match:
                        restaurant_name = match.group(1).strip()
                        if len(restaurant_name) > 2:
                            entities['restaurant_name'] = restaurant_name
                            break
                
                # Extract time patterns
                time_patterns = [
                    r"(\d{1,2}):(\d{2})",  # 19:30
                    r"(\d{1,2})\s+giờ",    # 7 giờ
                    r"(\d{1,2})\s*giờ\s*(\d{1,2})?\s*(trưa|chiều|tối|sáng)?",  # 12 giờ trưa
                    r"tối", r"sáng", r"trưa", r"chiều"
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, user_message_lower)
                    if match:
                        entities["time"] = match.group()
                        break
                
                # Extract date patterns
                date_patterns = [
                    r"ngày\s+mai",
                    r"hôm\s+nay", 
                    r"ngày\s+(\d{1,2})",
                    r"thứ\s+(\d+)"
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, user_message_lower)
                    if match:
                        entities["date"] = match.group()
                        break
                
                # Extract guest count
                guest_pattern = r"(\d+)\s*(?:người|khách|person)"
                guest_match = re.search(guest_pattern, user_message_lower)
                if guest_match:
                    entities["guest_count"] = int(guest_match.group(1))
            
            elif intent == "menu_inquiry":
                # Extract restaurant ID
                restaurant_pattern = r"nhà hàng\s+(\d+)|restaurant\s+(\d+)"
                restaurant_match = re.search(restaurant_pattern, user_message_lower)
                if restaurant_match:
                    entities["restaurant_id"] = int(restaurant_match.group(1) or restaurant_match.group(2))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in pattern entity extraction: {e}")
            return {}
    
    async def _llm_entity_extraction(self, user_message: str, intent: str, context: str = "") -> Dict[str, Any]:
        """LLM-based entity extraction using OpenAI"""
        try:
            if not self.openai_client:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Build dynamic prompt based on intent
            if intent == "table_inquiry":
                from datetime import datetime, timedelta
                current_year = datetime.now().year
                current_date = datetime.now().strftime("%Y-%m-%d")
                tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                
                prompt = f"""
                Extract entities from this Vietnamese message: "{user_message}"
                
                Extract the following entities:
                - restaurant_name: Name of the restaurant mentioned (e.g., "Seoul BBQ Premium", "Phở Bò ABC")
                - booking_time: Time mentioned (convert to format like "{current_year}-01-01 12:00" for tomorrow 12:00)
                - guest_count: Number of guests (default to 2 if not mentioned)
                - date: Date mentioned (e.g., "ngày mai", "hôm nay")
                
                IMPORTANT: Use current year {current_year} for dates. "ngày mai" = {tomorrow_date}, "hôm nay" = {current_date}
                
                Return JSON format only:
                {{
                    "restaurant_name": "extracted_name_or_null",
                    "booking_time": "{current_year}-01-01 12:00_or_null", 
                    "guest_count": 2_or_extracted_number,
                    "date": "extracted_date_or_null"
                }}
                """
            
            elif intent == "voucher_inquiry":
                prompt = f"""
                Extract entities from this Vietnamese message: "{user_message}"
                
                Extract the following entities:
                - voucher_code: The code of the voucher mentioned (e.g., "VOUCHER123", "PROMO2023")
                
                Return JSON format only:
                {{
                    "voucher_code": "extracted_code_or_null"
                }}
                """
            
            else:
                return {}
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an entity extraction system. Extract entities from Vietnamese text and return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Filter out null values
            filtered_result = {k: v for k, v in result.items() if v is not None and v != "null"}
            
            logger.info(f"LLM extracted entities for {intent}: {filtered_result}")
            return filtered_result
            
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return {}
    
    def _context_entity_extraction(self, context: str, intent: str) -> Dict[str, Any]:
        """Extract entities từ conversation context"""
        try:
            entities = {}
            
            if not context:
                return entities
            
            if intent == "restaurant_search":
                # Extract restaurant info từ context
                restaurant_patterns = [
                    r"restaurant\s+(\d+)",
                    r"nhà hàng\s+(\d+)",
                    r"restaurant\s+(\w+)"
                ]
                
                for pattern in restaurant_patterns:
                    match = re.search(pattern, context.lower())
                    if match:
                        entities["restaurant_id"] = int(match.group(1)) if match.group(1).isdigit() else match.group(1)
                        break
            
            elif intent in ["table_inquiry"]:
                # Extract booking info từ context
                guest_pattern = r"(\d+)\s*(?:người|khách|person)"
                guest_match = re.search(guest_pattern, context.lower())
                if guest_match:
                    entities["guest_count"] = int(guest_match.group(1))
                
                time_pattern = r"(\d{1,2}):(\d{2})|(\d{1,2})\s+giờ"
                time_match = re.search(time_pattern, context.lower())
                if time_match:
                    entities["time"] = time_match.group()
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in context entity extraction: {e}")
            return {}
    
    async def learn_from_interaction(
        self, user_id: str, user_message: str, intent: str, 
        entities: Dict[str, Any], response_success: bool
    ):
        """
        Learn từ user interactions để improve future responses
        + Feedback Learning Loop để tự động improve
        """
        try:
            # 1. Store interaction as preference
            interaction_data = {
                'intent': intent,
                'entities': entities,
                'success': response_success,
                'timestamp': int(time.time())
            }
            
            await self.vector_service.store_user_preference(
                user_id, 
                f"{intent}_interaction", 
                interaction_data
            )
            
            # 2. Store successful patterns for future recognition
            # ✅ FIX: Chỉ store pattern nếu intent không phải general_inquiry (tránh học sai)
            if response_success and intent != "general_inquiry":
                pattern_data = {
                    'pattern': user_message,
                    'intent': intent,
                    'entities': entities,
                    'timestamp': int(time.time())
                }
                
                await self.vector_service.store_user_preference(
                    user_id,
                    "successful_patterns",
                    pattern_data
                )
            
            # 3. FEEDBACK LEARNING LOOP - Store feedback để retrain sau
            # ✅ FIX: Không store feedback cho general_inquiry khi response_success=True (tránh học sai)
            # Chỉ store feedback cho failed cases hoặc general_inquiry + failed
            if not response_success or (intent == "general_inquiry" and not response_success):
                # Failed → Store để học
                feedback_entry = {
                    'user_message': user_message,
                    'predicted_intent': intent,
                    'entities': entities,
                    'success': response_success,
                    'timestamp': int(time.time()),
                    'user_id': user_id
                }
                
                # Store trong memory (có thể export để fine-tune sau)
                self.intent_feedback_dataset.append(feedback_entry)
                
                # Limit dataset size để tránh memory overflow
                if len(self.intent_feedback_dataset) > 1000:
                    self.intent_feedback_dataset = self.intent_feedback_dataset[-1000:]
                
                logger.info(f"Stored feedback entry for learning: {len(self.intent_feedback_dataset)} entries")
            
            # 4. Nếu có nhiều feedback và thành công → Update intent embedding
            if response_success and len(self.intent_feedback_dataset) > 0:
                # Có thể tự động update intent embedding với successful patterns
                await self._update_intent_embedding_from_feedback(intent, user_message)
            
            logger.info(f"Learned from interaction for user {user_id}: {intent} (success: {response_success})")
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    async def _update_intent_embedding_from_feedback(self, intent: str, successful_message: str):
        """
        Tự động update intent embedding từ successful feedback
        
        Args:
            intent: Intent name
            successful_message: Message đã được classify đúng
        """
        try:
            # Lấy current intent embedding
            results = await self.vector_service.search_intents(
                successful_message, limit=1, distance_threshold=1.0
            )
            
            if results:
                current_metadata = results[0]['metadata']
                current_examples = current_metadata.get('examples', [])
                
                # Thêm successful message vào examples nếu chưa có
                if successful_message not in current_examples:
                    new_examples = current_examples + [successful_message]
                    
                    # Update embedding
                    intent_def = self.intent_definitions.get(intent, {})
                    await self.vector_service.store_intent_embedding(
                        intent,
                        new_examples,
                        intent_def.get("api_function")
                    )
                    
                    logger.info(f"Updated intent embedding for {intent} with new successful example")
            
        except Exception as e:
            logger.error(f"Error updating intent embedding from feedback: {e}")
    
    def export_feedback_dataset(self, file_path: str = "intent_feedback_dataset.json"):
        """
        Export feedback dataset để fine-tune model sau
        
        Format:
        {
            "training_data": [
                {"input": "user message", "output": "correct_intent"},
                ...
            ]
        }
        """
        try:
            training_data = []
            
            for entry in self.intent_feedback_dataset:
                # Chỉ export các entries có thông tin đầy đủ
                if entry.get('user_message') and entry.get('predicted_intent'):
                    training_data.append({
                        "input": entry['user_message'],
                        "output": entry['predicted_intent'],
                        "success": entry.get('success', False),
                        "timestamp": entry.get('timestamp')
                    })
            
            export_data = {
                "total_entries": len(training_data),
                "training_data": training_data,
                "exported_at": int(time.time())
            }
            
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Exported {len(training_data)} feedback entries to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting feedback dataset: {e}")
            return None
    
    def store_intent_feedback(
        self, user_message: str, predicted_intent: str, correct_intent: str = None, 
        user_feedback: str = None
    ):
        """
        Store explicit feedback từ user về intent classification
        
        Args:
            user_message: Original user message
            predicted_intent: Intent đã được predict
            correct_intent: Intent đúng (nếu user correct)
            user_feedback: User feedback text (optional)
        """
        try:
            feedback_entry = {
                'user_message': user_message,
                'predicted_intent': predicted_intent,
                'correct_intent': correct_intent,
                'user_feedback': user_feedback,
                'success': correct_intent == predicted_intent if correct_intent else None,
                'timestamp': int(time.time())
            }
            
            self.intent_feedback_dataset.append(feedback_entry)
            
            # Limit dataset size
            if len(self.intent_feedback_dataset) > 1000:
                self.intent_feedback_dataset = self.intent_feedback_dataset[-1000:]
            
            logger.info(f"Stored explicit feedback: predicted={predicted_intent}, correct={correct_intent}")
            
            # Nếu có correct intent → Update intent embedding ngay
            if correct_intent and correct_intent != predicted_intent:
                asyncio.create_task(
                    self._update_intent_embedding_from_feedback(correct_intent, user_message)
                )
            
        except Exception as e:
            logger.error(f"Error storing intent feedback: {e}")
    
    async def get_personalized_suggestions(self, user_id: str, current_intent: str) -> List[str]:
        """Get personalized suggestions dựa trên user history"""
        try:
            suggestions = []
            
            # Get user preferences
            preferences = await self.vector_service.get_user_preferences(user_id)
            
            if current_intent == "restaurant_search":
                # Suggest based on previous restaurant searches
                for pref in preferences:
                    if pref['metadata'].get('preference_type') == 'restaurant_search':
                        data = pref['metadata'].get('data', {})
                        if data.get('cuisine_type'):
                            suggestions.append(f"Tìm nhà hàng {data['cuisine_type']}")
                        if data.get('location'):
                            suggestions.append(f"Nhà hàng gần {data['location']}")
            
            elif current_intent == "menu_inquiry":
                # Suggest based on previous menu inquiries
                for pref in preferences:
                    if pref['metadata'].get('preference_type') == 'menu_inquiry':
                        data = pref['metadata'].get('data', {})
                        if data.get('restaurant_id'):
                            suggestions.append(f"Xem menu nhà hàng {data['restaurant_id']}")
            
            # Remove duplicates và limit
            suggestions = list(set(suggestions))[:3]
            
            logger.info(f"Generated {len(suggestions)} personalized suggestions for user {user_id}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting personalized suggestions: {e}")
            return []


# Global instance
vector_intent_service = VectorIntentService()
