import json
import logging
import os
import time
import uuid
from typing import Optional, Dict, List, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorService:
    """Vector Database Service sử dụng Qdrant embedded và Sentence Transformers."""

    CONVERSATIONS_COLLECTION = "conversations"
    RESTAURANTS_COLLECTION = "restaurants"
    MENUS_COLLECTION = "menus"
    USER_PREFERENCES_COLLECTION = "user_preferences"
    INTENTS_COLLECTION = "intents"  # NEW: Intent Embedding Collection
    IMAGE_URL_COLLECTION = "image_url"

    def __init__(self):
        try:
            persist_dir = os.getenv("QDRANT_DB_PATH", "storage/qdrant")
            os.makedirs(persist_dir, exist_ok=True)
            self.client = QdrantClient(path=persist_dir)
            logger.info("Qdrant embedded client initialised at %s", persist_dir)

            self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info("Sentence Transformer model loaded successfully (dim=%s)", self.vector_size)

            self._ensure_collections()
        except Exception as e:
            logger.error(f"Error initializing VectorService: {e}")
            raise

    def _ensure_collections(self):
        """Tạo các collection cần thiết trong Qdrant (nếu chưa tồn tại)."""
        try:
            existing = {
                collection.name
                for collection in (self.client.get_collections().collections or [])
            }
        except Exception:
            existing = set()

        for name in (
            self.CONVERSATIONS_COLLECTION,
            self.RESTAURANTS_COLLECTION,
            self.MENUS_COLLECTION,
            self.USER_PREFERENCES_COLLECTION,
            self.INTENTS_COLLECTION,  # NEW: Intent collection
            self.IMAGE_URL_COLLECTION,
        ):
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )

        logger.info("All Qdrant collections ready")

    def encode_text(self, text: str) -> List[float]:
        """Encode text to vector embedding."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

    def _build_filter(self, field_pairs: Dict[str, Optional[str]]) -> Optional[Filter]:
        conditions = []
        for key, value in field_pairs.items():
            if value is not None:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if not conditions:
            return None
        return Filter(must=conditions)

    def _format_results(self, points) -> List[Dict]:
        formatted = []
        for point in points or []:
            payload = point.payload or {}
            score = point.score or 0.0
            formatted.append(
                {
                    "document": payload.get("document", ""),
                    "distance": max(0.0, 1.0 - score),
                    "metadata": payload,
                    "id": point.id,
                }
            )
        return formatted

    def _make_point_id(self, collection: str, *parts: Any, allow_int: bool = False):
        """Generate a Qdrant-compatible point ID for given parts."""
        valid_parts = [part for part in parts if part is not None]
        if not valid_parts:
            raise ValueError("Point ID requires at least one non-null part")

        if allow_int and len(valid_parts) == 1:
            value = valid_parts[0]
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)

        base = ":".join(str(part) for part in valid_parts)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{base}"))

    def _extract_id(self, data: Dict[str, Any], *preferred_keys: str) -> Optional[Any]:
        """Extract identifier from data supporting multiple naming styles."""
        if not isinstance(data, dict):
            return None

        for key in preferred_keys:
            if key in data:
                return data[key]

        # Case-insensitive lookup
        lowered_map = {key.lower(): key for key in data.keys()}
        for key in preferred_keys:
            lowered_key = key.lower()
            if lowered_key in lowered_map:
                return data[lowered_map[lowered_key]]

        # Generic fallback: pick the first key that ends with 'id'
        for key, value in data.items():
            if isinstance(key, str) and key.lower().endswith("id"):
                return value

        return None

    async def store_conversation(self, user_id: str, message: str, response: str, intent: str = None):
        """
        Store conversation for context và học hỏi.
        
        ⚠️ PRIVACY: Conversations chứa dữ liệu nhạy cảm, phải luôn có user_id.
        
        Args:
            user_id: User ID (REQUIRED - không được None)
            message: User message
            response: AI response
            intent: Intent được recognize
        """
        try:
            # ⚠️ SECURITY: Validate user_id để đảm bảo privacy
            if not user_id or user_id.strip() == "":
                logger.error("Cannot store conversation without user_id - privacy violation")
                return
            
            user_id = user_id.strip()  # Sanitize
            
            conversation_text = f"User: {message}\nAssistant: {response}"
            vector = self.encode_text(conversation_text)

            if not vector:
                logger.warning("Failed to encode conversation text")
                return

            # Point ID bao gồm user_id để dễ query và delete sau này
            conversation_id = self._make_point_id(
                self.CONVERSATIONS_COLLECTION, user_id, int(time.time())
            )
            
            payload = {
                "user_id": user_id,  # ✅ REQUIRED - Luôn có trong payload
                "timestamp": str(int(time.time())),
                "intent": intent or "unknown",
                "message_length": len(message),
                "response_length": len(response),
                "document": conversation_text,
                "point_id": str(conversation_id),
            }

            self.client.upsert(
                collection_name=self.CONVERSATIONS_COLLECTION,
                points=[PointStruct(id=conversation_id, vector=vector, payload=payload)],
            )

            logger.info(f"Stored conversation for user {user_id} (intent: {intent})")

        except Exception as e:
            logger.error(f"Error storing conversation: {e}")

    async def get_user_conversations_recent(
        self, user_id: str, limit: int = 10
    ) -> List[Dict]:
        """
        Lấy recent conversations của user theo timestamp (KHÔNG semantic search)
        
        ✅ TỐI ƯU: Với conversations, chỉ cần lấy theo user_id và timestamp DESC
        Không cần semantic search vì mỗi user chỉ cần conversations của họ.
        
        Args:
            user_id: User ID (REQUIRED)
            limit: Số lượng conversations gần nhất
            
        Returns:
            List of conversations (sorted by timestamp DESC - mới nhất trước)
        """
        try:
            if not user_id or user_id.strip() == "":
                return []
            
            user_id = user_id.strip()
            
            # ✅ FIX: Scroll không dùng filter (Qdrant local mode không hỗ trợ)
            # Scroll để lấy tất cả conversations
            all_points = []
            offset = None
            
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.CONVERSATIONS_COLLECTION,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Không cần vectors vì không search
                )
                
                # ✅ FIX: Manual filter theo user_id
                user_points = [
                    point for point in points
                    if point.payload and point.payload.get("user_id") == user_id
                ]
                all_points.extend(user_points)
                
                if offset is None or len(all_points) >= limit * 2:  # Lấy nhiều hơn để sort
                    break
            
            # Sort theo timestamp DESC (mới nhất trước)
            def get_timestamp(point):
                try:
                    return int(point.payload.get("timestamp", 0))
                except:
                    return 0
            
            all_points.sort(key=get_timestamp, reverse=True)
            
            # Format và limit
            formatted_results = []
            for point in all_points[:limit]:
                payload = point.payload or {}
                formatted_results.append({
                    "id": point.id,
                    "document": payload.get("document", ""),
                    "metadata": payload,
                    "distance": 0.0,  # Không có distance vì không phải semantic search
                })
            
            logger.info(f"Retrieved {len(formatted_results)} recent conversations for user {user_id} (by timestamp)")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting user conversations recent: {e}")
            return []
    
    async def search_similar_conversations(
        self, query: str, user_id: str = None, limit: int = 3
    ) -> List[Dict]:
        """
        Search for similar conversations để lấy context.
        
        ⚠️ QUAN TRỌNG: Luôn filter theo user_id để tránh leak conversations của users khác.
        
        ⚠️ NOTE: Nếu chỉ cần recent conversations, dùng get_user_conversations_recent() thay vì method này
        (Method này dùng semantic search, có thể chậm hơn và không cần thiết cho conversations)
        
        Args:
            query: Query text
            user_id: User ID để filter (REQUIRED cho privacy)
            limit: Số lượng results
            
        Returns:
            List of similar conversations
        """
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []

            # ⚠️ SECURITY: Nếu không có user_id, không search conversations để tránh leak data
            if not user_id:
                logger.warning("search_similar_conversations called without user_id - skipping for privacy")
                return []

            # ✅ FIX: Search không dùng filter (Qdrant local mode không hỗ trợ)
            # Tăng limit để có đủ kết quả sau khi filter
            results = self.client.search(
                collection_name=self.CONVERSATIONS_COLLECTION,
                query_vector=query_vector,
                limit=limit * 5,  # ✅ Tăng để có đủ kết quả sau khi filter
            )
            formatted_results = self._format_results(results)
            
            # ✅ FIX: Manual filter theo user_id (PRIVACY CRITICAL)
            verified_results = [
                r for r in formatted_results 
                if r.get("metadata", {}).get("user_id") == user_id
            ]
            
            # Limit sau khi filter
            verified_results = verified_results[:limit]
            
            logger.info(
                "Found %s similar conversations for user %s (filtered from %s)", 
                len(verified_results), user_id, len(formatted_results)
            )
            return verified_results

        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    async def delete_user_conversations(self, user_id: str) -> int:
        """
        Delete tất cả conversations của một user (GDPR compliance).
        
        ⚠️ PRIVACY: Method này cho phép user xóa dữ liệu riêng tư của họ.
        
        Args:
            user_id: User ID cần xóa conversations
            
        Returns:
            Số lượng conversations đã xóa
        """
        try:
            if not user_id or user_id.strip() == "":
                logger.error("Cannot delete conversations without user_id")
                return 0
            
            user_id = user_id.strip()
            
            # Tìm tất cả points của user này
            filter_obj = self._build_filter({"user_id": user_id})
            point_ids = []
            
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.CONVERSATIONS_COLLECTION,
                    limit=100,
                    filter=filter_obj,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                
                for point in points:
                    point_ids.append(point.id)
                
                if offset is None:
                    break
            
            if not point_ids:
                logger.info(f"No conversations found for user {user_id} to delete")
                return 0
            
            # Delete tất cả points
            self.client.delete(
                collection_name=self.CONVERSATIONS_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )
            
            logger.info(f"Deleted {len(point_ids)} conversations for user {user_id}")
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Error deleting user conversations: {e}")
            return 0
    
    async def get_user_conversations_count(self, user_id: str) -> int:
        """
        Đếm số lượng conversations của một user.
        
        Args:
            user_id: User ID
            
        Returns:
            Số lượng conversations
        """
        try:
            if not user_id or user_id.strip() == "":
                return 0
            
            user_id = user_id.strip()
            filter_obj = self._build_filter({"user_id": user_id})
            
            count = self.client.count(
                collection_name=self.CONVERSATIONS_COLLECTION,
                filter=filter_obj,
            )
            
            return count.count if count else 0
            
        except Exception as e:
            logger.error(f"Error counting user conversations: {e}")
            return 0
    
    async def store_restaurant_data(self, restaurant_data: List[Dict]):
        """Store restaurant information cho semantic search."""
        try:
            points = []
            for restaurant in restaurant_data:
                searchable_text = self._create_restaurant_searchable_text(restaurant)
                vector = self.encode_text(searchable_text)
                if not vector:
                    logger.warning("Failed to encode restaurant %s", restaurant.get("id"))
                    continue
                raw_id = self._extract_id(restaurant, "id", "restaurantId", "restaurantID")
                if raw_id is None:
                    logger.warning("Restaurant data missing id: %s", restaurant)
                    continue

                point_id = self._make_point_id(
                    self.RESTAURANTS_COLLECTION, raw_id, allow_int=True
                )
                payload = {
                    **restaurant,
                    "stored_at": str(int(time.time())),
                    "document": searchable_text,
                    "point_id": str(point_id),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if points:
                self.client.upsert(collection_name=self.RESTAURANTS_COLLECTION, points=points)
                logger.info("Stored %s restaurants", len(points))

        except Exception as e:
            logger.error(f"Error storing restaurant data: {e}")

    async def upsert_restaurant(self, restaurant: Dict):
        """Upsert single restaurant entry."""
        await self.store_restaurant_data([restaurant])

    async def delete_restaurant(self, restaurant_id: int):
        """Remove restaurant khỏi vector store."""
        try:
            point_id = self._make_point_id(
                self.RESTAURANTS_COLLECTION, restaurant_id, allow_int=True
            )
            self.client.delete(
                collection_name=self.RESTAURANTS_COLLECTION,
                points_selector=PointIdsList(points=[point_id]),
            )
            logger.info("Deleted restaurant %s from vector store", restaurant_id)
        except Exception as e:
            logger.error(f"Error deleting restaurant {restaurant_id}: {e}")

    async def search_restaurants(
        self, query: str, limit: int = 5, distance_threshold: float = 0.5
    ) -> List[Dict]:
        """Semantic search cho restaurants với distance filtering."""
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []

            # Search nhiều hơn để có thể filter sau
            results = self.client.search(
                collection_name=self.RESTAURANTS_COLLECTION,
                query_vector=query_vector,
                limit=limit * 3,  # Search nhiều hơn để filter
            )
            formatted_results = self._format_results(results)
            
            # FILTER theo distance threshold - CHỈ lấy results "gần gần"
            filtered_results = [
                r for r in formatted_results 
                if r["distance"] < distance_threshold
            ]
            
            # Giới hạn lại số lượng sau khi filter
            filtered_results = filtered_results[:limit]
            
            logger.info(
                "Found %s restaurants (filtered from %s, threshold=%.2f) for query: %s", 
                len(filtered_results), 
                len(formatted_results),
                distance_threshold,
                query
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Error searching restaurants: {e}")
            return []

    async def get_restaurants_by_ids(self, restaurant_ids: List[Any]) -> Dict[Any, Dict]:
        """Retrieve restaurant payloads theo danh sách restaurant_id."""
        try:
            if not restaurant_ids:
                return {}

            unique_ids: List[Any] = []
            seen_keys: set[str] = set()
            for raw_id in restaurant_ids:
                if raw_id in (None, ""):
                    continue
                key = str(raw_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_ids.append(raw_id)

            if not unique_ids:
                return {}

            point_ids: List[Any] = []
            point_id_lookup: Dict[str, Any] = {}
            for rid in unique_ids:
                try:
                    point_id = self._make_point_id(
                        self.RESTAURANTS_COLLECTION, rid, allow_int=True
                    )
                except Exception:
                    point_id = self._make_point_id(
                        self.RESTAURANTS_COLLECTION, str(rid)
                    )
                point_ids.append(point_id)
                point_id_lookup[str(point_id)] = rid

            points = self.client.retrieve(
                collection_name=self.RESTAURANTS_COLLECTION,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
            )

            results: Dict[Any, Dict] = {}
            for point in points or []:
                payload = point.payload or {}
                restaurant_id = (
                    payload.get("id")
                    or payload.get("restaurantId")
                    or payload.get("restaurant_id")
                    or point_id_lookup.get(str(point.id))
                )
                if restaurant_id is None:
                    continue
                results[restaurant_id] = payload

            return results

        except Exception as e:
            logger.error(f"Error fetching restaurants by ids: {e}")
            return {}

    async def cross_collection_search(self, query: str, limit: int = 5, distance_threshold: float = 0.5) -> List[Dict]:
        """
        Cross-collection search: Menu → Restaurant join
        
        Tìm món ăn trước, sau đó lấy thông tin nhà hàng tương ứng.
        Giải quyết vấn đề "lẩu kim châm" không có trong restaurant collection.
        """
        try:
            # 1. Search menus trước
            menu_results = await self.search_menus(
                query, limit=limit * 2, distance_threshold=distance_threshold
            )
            
            if not menu_results:
                return []
            
            # 2. Extract restaurant IDs từ menu results
            restaurant_ids = []
            for menu_result in menu_results:
                restaurant_id = menu_result.get('metadata', {}).get('restaurant_id')
                if restaurant_id is not None:
                    restaurant_ids.append(restaurant_id)
            
            if not restaurant_ids:
                return []
            
            # 3. Get restaurant details
            restaurants = await self.get_restaurants_by_ids(restaurant_ids)
            
            # 4. Format results với restaurant info
            formatted_results = []
            for menu_result in menu_results:
                restaurant_id = menu_result.get('metadata', {}).get('restaurant_id')
                if restaurant_id and restaurant_id in restaurants:
                    restaurant_info = restaurants[restaurant_id]
                    # Combine menu + restaurant info
                    combined_result = {
                        'distance': menu_result['distance'],
                        'metadata': {
                            **menu_result['metadata'],
                            'restaurant_name': restaurant_info.get('restaurantName') or restaurant_info.get('name'),
                            'restaurant_address': restaurant_info.get('address'),
                            'restaurant_cuisine': restaurant_info.get('cuisineType'),
                            'restaurant_rating': restaurant_info.get('rating'),
                            'search_type': 'menu_to_restaurant'
                        }
                    }
                    formatted_results.append(combined_result)
            
            # 5. Sort by distance và limit
            formatted_results.sort(key=lambda x: x['distance'])
            formatted_results = formatted_results[:limit]
            
            logger.info(f"Cross-collection search found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in cross-collection search: {e}")
            return []

    async def store_menu_data(self, restaurant_id: int, menu_data: List[Dict]):
        """Store menu information cho semantic search."""
        try:
            logger.debug(
                "store_menu_data -> restaurant %s received %d items",
                restaurant_id,
                len(menu_data),
            )
            points = []
            for dish in menu_data:
                searchable_text = self._create_menu_searchable_text(dish, restaurant_id)
                vector = self.encode_text(searchable_text)
                if not vector:
                    logger.warning("Failed to encode dish %s", dish.get("id"))
                    continue
                dish_raw_id = self._extract_id(
                    dish,
                    "id",
                    "dishId",
                    "dishID",
                    "menuId",
                    "menuID",
                    "dish_id",
                )
                if dish_raw_id is None:
                    logger.warning(
                        "Menu item missing id for restaurant %s: %s", restaurant_id, dish
                    )
                    continue

                point_id = self._make_point_id(
                    self.MENUS_COLLECTION, restaurant_id, dish_raw_id
                )
                payload = {
                    **dish,
                    "restaurant_id": restaurant_id,
                    "stored_at": str(int(time.time())),
                    "document": searchable_text,
                    "point_id": str(point_id),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if points:
                self.client.upsert(collection_name=self.MENUS_COLLECTION, points=points)
                logger.info(
                    "Stored %s menu items for restaurant %s", len(points), restaurant_id
                )
            else:
                logger.debug(
                    "store_menu_data -> no valid menu items stored for restaurant %s",
                    restaurant_id,
                )

        except Exception as e:
            logger.error(f"Error storing menu data: {e}")

    async def store_services_data(self, restaurant_id: int, services_data: List[Dict]):
        """Store restaurant services information cho semantic search."""
        try:
            logger.debug(
                "store_services_data -> restaurant %s received %d services",
                restaurant_id,
                len(services_data),
            )
            points = []
            for service in services_data:
                searchable_text = self._create_service_searchable_text(service, restaurant_id)
                vector = self.encode_text(searchable_text)
                if not vector:
                    logger.warning("Failed to encode service %s", service.get("id"))
                    continue
                service_raw_id = self._extract_id(
                    service,
                    "id",
                    "serviceId",
                    "serviceID",
                    "service_id",
                    "code",
                )
                if service_raw_id is None:
                    logger.warning(
                        "Service item missing id for restaurant %s: %s",
                        restaurant_id,
                        service,
                    )
                    continue

                point_id = self._make_point_id(
                    self.MENUS_COLLECTION, "service", restaurant_id, service_raw_id
                )
                payload = {
                    **service,
                    "restaurant_id": restaurant_id,
                    "stored_at": str(int(time.time())),
                    "document": searchable_text,
                    "point_id": str(point_id),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if points:
                self.client.upsert(collection_name=self.MENUS_COLLECTION, points=points)  # Reuse menus collection
                logger.info(
                    "Stored %s services for restaurant %s", len(points), restaurant_id
                )
            else:
                logger.debug(
                    "store_services_data -> no valid services stored for restaurant %s",
                    restaurant_id,
                )

        except Exception as e:
            logger.error(f"Error storing services data: {e}")

    async def store_tables_data(self, restaurant_id: int, tables_data: List[Dict]):
        """Store restaurant table information cho semantic search."""
        try:
            points = []
            for table in tables_data:
                searchable_text = self._create_table_searchable_text(table, restaurant_id)
                vector = self.encode_text(searchable_text)
                if not vector:
                    logger.warning("Failed to encode table %s", table)
                    continue

                table_raw_id = self._extract_id(
                    table,
                    "id",
                    "tableId",
                    "tableID",
                    "table_id",
                    "code",
                )
                if table_raw_id is None:
                    logger.warning(
                        "Table item missing id for restaurant %s: %s",
                        restaurant_id,
                        table,
                    )
                    continue

                point_id = self._make_point_id(
                    self.MENUS_COLLECTION, "table", restaurant_id, table_raw_id
                )
                payload = {
                    **table,
                    "restaurant_id": restaurant_id,
                    "stored_at": str(int(time.time())),
                    "document": searchable_text,
                    "point_id": str(point_id),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            if points:
                self.client.upsert(collection_name=self.MENUS_COLLECTION, points=points)
                logger.info(
                    "Stored %s tables for restaurant %s", len(points), restaurant_id
                )

        except Exception as e:
            logger.error(f"Error storing tables data: {e}")

    async def store_table_layouts_data(self, restaurant_id: int, table_layouts_data: List[Dict]):
        """Store table layouts and media info in IMAGE_URL_COLLECTION, always include restaurant_id."""
        try:
            points = []
            for layout in table_layouts_data:
                # Table layout có thể là ảnh/phòng ...
                searchable_text = self._create_table_layout_searchable_text(layout, restaurant_id)
                vector = self.encode_text(searchable_text)
                if not vector:
                    logger.warning("Failed to encode table layout for image_url: %s", layout.get("id"))
                    continue
                mediaId = (
                    layout.get("mediaId")
                    or layout.get("id")
                    or layout.get("layoutId")
                )
                # Lưu từng ảnh/media là 1 point
                point_id = self._make_point_id(
                    self.IMAGE_URL_COLLECTION,
                    "table_layout",
                    restaurant_id,
                    mediaId or uuid.uuid4().hex,
                )
                payload = {
                    **layout,
                    "restaurant_id": restaurant_id,
                    "stored_at": str(int(time.time())),
                    "type": layout.get("type") or "table_layout",
                    "point_id": str(point_id),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            if points:
                self.client.upsert(collection_name=self.IMAGE_URL_COLLECTION, points=points)
                logger.info(
                    "Stored %s table layouts/images for restaurant %s in image_url collection", len(points), restaurant_id
                )
        except Exception as e:
            logger.error(f"Error storing table layouts/image data: {e}")

    async def upsert_menu(self, restaurant_id: int, dish: Dict):
        """Upsert single menu item."""
        await self.store_menu_data(restaurant_id, [dish])

    async def delete_menu(self, restaurant_id: int, dish_id: int):
        """Remove menu item khỏi vector store."""
        try:
            point_id = self._make_point_id(
                self.MENUS_COLLECTION, restaurant_id, dish_id
            )
            self.client.delete(
                collection_name=self.MENUS_COLLECTION,
                points_selector=PointIdsList(points=[point_id]),
            )
            logger.info(
                "Deleted menu item %s for restaurant %s from vector store",
                dish_id,
                restaurant_id,
            )
        except Exception as e:
            logger.error(f"Error deleting menu {dish_id} of restaurant {restaurant_id}: {e}")

    async def search_menus(
        self, query: str, restaurant_id: int = None, limit: int = 5, distance_threshold: float = 0.5
    ) -> List[Dict]:
        """Semantic search cho menus với distance filtering."""
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []

            # ✅ FIX: Search không dùng filter (Qdrant local mode không hỗ trợ)
            # Tăng limit để có đủ kết quả sau khi filter
            results = self.client.search(
                collection_name=self.MENUS_COLLECTION,
                query_vector=query_vector,
                limit=limit * 5,  # ✅ Tăng từ limit * 3 lên limit * 5
            )
            formatted_results = self._format_results(results)
            
            # ✅ FIX: Manual filter theo restaurant_id nếu có
            if restaurant_id is not None:
                formatted_results = [
                    r for r in formatted_results
                    if r.get("metadata", {}).get("restaurant_id") == restaurant_id
                ]
            
            # FILTER theo distance threshold - CHỈ lấy results "gần gần"
            filtered_results = [
                r for r in formatted_results 
                if r["distance"] < distance_threshold
            ]
            
            # Giới hạn lại số lượng sau khi filter
            filtered_results = filtered_results[:limit]
            
            logger.info(
                "Found %s menu items (filtered from %s, threshold=%.2f) for query: %s",
                len(filtered_results),
                len(formatted_results),
                distance_threshold,
                query
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Error searching menus: {e}")
            return []

    async def semantic_menu_search_with_reasoning(
        self,
        user_message: str,
        reasoning_profile: Dict[str, Any],
        restaurant_id: int = None,
        limit: int = 10,
        distance_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Semantic search kết hợp reasoning profile
        
        Strategy:
        1. Tạo embedding từ user_message + reasoning summary (enhanced query)
        2. Vector search để lấy candidates
        3. Filter & boost theo tags/metadata match với reasoning profile
        
        Args:
            user_message: User query về món ăn
            reasoning_profile: Profile từ LLM reasoning (diet_profile, occasion, temperature, ...)
            restaurant_id: Filter theo restaurant (optional)
            limit: Số lượng results
            distance_threshold: Distance threshold cho vector search
            
        Returns:
            List of menu results với distance/score đã được boost
        """
        try:
            # 1. Tạo enhanced query text với search_query + summary (HYBRID APPROACH)
            # Priority: search_query (LLM-generated) > summary > user_message
            search_query = reasoning_profile.get("search_query", "").strip()
            summary = reasoning_profile.get("summary", "").strip()
            
            if search_query:
                # Nếu có search_query từ LLM → dùng search_query (đã được tối ưu)
                query_text = search_query
                if summary and summary != search_query:
                    # Thêm summary nếu khác với search_query (thêm context)
                    query_text += f". {summary}"
            elif summary:
                # Fallback: dùng summary nếu không có search_query
                query_text = summary
            else:
                # Fallback cuối: dùng user_message
                query_text = user_message
            
            logger.debug(f"Semantic search query: {query_text[:100]}...")
            
            # 2. Vector search với enhanced query (lấy nhiều hơn để filter)
            results = await self.search_menus(
                query_text,
                restaurant_id=restaurant_id,
                limit=limit * 3,  # Lấy nhiều hơn để filter & boost
                distance_threshold=distance_threshold * 1.2  # Threshold cao hơn để có đủ candidates
            )
            
            if not results:
                return []
            
            # 3. Convert distance to score và boost theo reasoning profile
            filtered_results = []
            diet_profile = reasoning_profile.get("diet_profile", {})
            occasion = reasoning_profile.get("occasion", "any")
            temperature = reasoning_profile.get("temperature", "any")
            spice_level = reasoning_profile.get("spice_level", "any")
            constraints = reasoning_profile.get("constraints", [])
            
            for result in results:
                metadata = result.get("metadata", {})
                
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
                
                # Start với base score (convert distance to similarity score)
                base_distance = result.get("distance", 1.0)
                base_score = max(0.0, 1.0 - base_distance)  # Convert distance to similarity (0..1)
                
                boost_score = 0.0  # Boost điểm cho matching tags
                
                # Boost theo diet_profile tags
                if diet_profile.get("high_protein") and "high_protein" in tags:
                    boost_score += 0.15
                if diet_profile.get("low_carb") and "low_carb" in tags:
                    boost_score += 0.12
                if diet_profile.get("low_fat") and "low_fat" in tags:
                    boost_score += 0.12
                if diet_profile.get("light_meal") and "light_meal" in tags:
                    boost_score += 0.10
                
                # Boost theo occasion
                if occasion == "gym" and "high_protein" in tags:
                    boost_score += 0.10
                if occasion == "sick" and "good_when_sick" in tags:
                    boost_score += 0.15
                if occasion == "comfort" and "comfort_food" in tags:
                    boost_score += 0.10
                if occasion == "celebration" and "celebration" in tags:
                    boost_score += 0.08
                
                # Boost theo temperature
                if temperature == "hot":
                    # Check nếu món nóng (từ category, name, description)
                    category = (metadata.get("category") or "").lower()
                    name = (metadata.get("name") or "").lower()
                    desc = (metadata.get("description") or "").lower()
                    hot_items = ["cháo", "soup", "canh", "lẩu", "phở", "bún", "miến", "nước dùng", "hot pot"]
                    if any(item in category or item in name or item in desc for item in hot_items):
                        boost_score += 0.12
                
                # Boost theo spice_level
                if spice_level == "spicy" and (metadata.get("is_spicy") or "spicy" in tags):
                    boost_score += 0.08
                elif spice_level in ["mild", "medium"] and ("non_spicy" in tags or metadata.get("is_non_spicy")):
                    boost_score += 0.06
                
                # Boost theo constraints_text (backward compatibility: cũng check "constraints")
                constraints_text = reasoning_profile.get("constraints_text", constraints)
                if not constraints_text:
                    constraints_text = constraints  # Fallback
                
                constraint_text = f"{metadata.get('name', '')} {metadata.get('description', '')}".lower()
                for constraint in constraints_text:
                    constraint_lower = str(constraint).lower()
                    if "chay" in constraint_lower and ("vegetarian" in tags or metadata.get("is_vegetarian")):
                        boost_score += 0.10
                    if "ít dầu" in constraint_lower or "ít béo" in constraint_lower:
                        if "low_fat" in tags:
                            boost_score += 0.08
                    if "không cay" in constraint_lower:
                        if "non_spicy" in tags or metadata.get("is_non_spicy"):
                            boost_score += 0.08
                
                # Boost theo cuisine (nếu có)
                cuisine_list = reasoning_profile.get("cuisine", [])
                if isinstance(cuisine_list, list) and cuisine_list:
                    # Check trong metadata (restaurant cuisine, dish category...)
                    restaurant_cuisine = (metadata.get("restaurant_cuisine") or "").lower()
                    dish_category = (metadata.get("category") or "").lower()
                    for cuisine in cuisine_list:
                        cuisine_lower = str(cuisine).lower()
                        if cuisine_lower in restaurant_cuisine or cuisine_lower in dish_category:
                            boost_score += 0.08
                
                # Boost theo is_local_specialty
                if reasoning_profile.get("is_local_specialty") and metadata.get("is_local_specialty"):
                    boost_score += 0.10
                
                # Final score = base_score + boost (cap at 1.0)
                final_score = min(1.0, base_score + boost_score)
                # Convert back to distance for consistency
                final_distance = max(0.0, 1.0 - final_score)
                
                result["distance"] = final_distance
                result["score"] = final_score  # Also store score for sorting
                result["boost_score"] = boost_score  # Debug info
                
                filtered_results.append(result)
            
            # 4. Sort by score (descending) hoặc distance (ascending)
            filtered_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            # 5. Filter theo final distance threshold và limit
            final_results = [
                r for r in filtered_results
                if r.get("distance", 1.0) < distance_threshold
            ][:limit]
            
            logger.info(
                "Semantic menu search with reasoning: found %s items (from %s candidates) for query: %s",
                len(final_results),
                len(results),
                user_message[:50]
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in semantic_menu_search_with_reasoning: {e}", exc_info=True)
            # Fallback to regular search
            return await self.search_menus(
                user_message,
                restaurant_id=restaurant_id,
                limit=limit,
                distance_threshold=distance_threshold
            )

    async def store_user_preference(self, user_id: str, preference_type: str, data: Dict):
        """Store user preferences để cá nhân hóa."""
        try:
            # ✅ FIX: Khai báo preference_id TRƯỚC khi dùng
            preference_user = user_id or "anonymous"
            preference_id = self._make_point_id(
                self.USER_PREFERENCES_COLLECTION,
                preference_user,
                preference_type,
                int(time.time()),
            )
            
            preference_text = (
                f"User {user_id} preference for {preference_type}: {json.dumps(data, ensure_ascii=False)}"
            )
            vector = self.encode_text(preference_text)
            if not vector:
                logger.warning("Failed to encode user preference")
                return

            payload = {
                "user_id": user_id,
                "preference_type": preference_type,
                "timestamp": str(int(time.time())),
                "data": data,
                "document": preference_text,
                "point_id": str(preference_id),  # ✅ Giờ đã có preference_id
            }

            self.client.upsert(
                collection_name=self.USER_PREFERENCES_COLLECTION,
                points=[PointStruct(id=preference_id, vector=vector, payload=payload)],
            )

            logger.info("Stored preference for user %s: %s", user_id, preference_type)

        except Exception as e:
            logger.error(f"Error storing user preference: {e}")

    async def get_user_preferences(
        self, user_id: str, preference_type: str = None
    ) -> List[Dict]:
        """Get user preferences cho personalization."""
        try:
            conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            if preference_type:
                conditions.append(
                    FieldCondition(key="preference_type", match=MatchValue(value=preference_type))
                )

            filter_obj = Filter(must=conditions)

            all_results: List[Dict] = []
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.USER_PREFERENCES_COLLECTION,
                    limit=64,
                    filter=filter_obj,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    all_results.append(
                        {
                            "metadata": payload,
                            "document": payload.get("document"),
                            "id": point.id,
                        }
                    )

                if offset is None:
                    break

            logger.info("Found %s preferences for user %s", len(all_results), user_id)
            return all_results

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return []

    async def get_context_for_query(self, query: str, user_id: str = None, distance_threshold: float = 0.5) -> str:
        """Get relevant context cho user query với distance filtering."""
        try:
            context_parts: List[str] = []

            if user_id:
                # ✅ TỐI ƯU: Lấy recent conversations theo timestamp thay vì semantic search
                conversations = await self.get_user_conversations_recent(user_id, limit=3)
                if conversations:
                    context_parts.append("Previous conversations:")
                    for conv in conversations:
                        context_parts.append(f"- {conv['document']}")

            restaurants = await self.search_restaurants(query, limit=2, distance_threshold=distance_threshold)
            if restaurants:
                context_parts.append("Relevant restaurant information:")
                for restaurant in restaurants:
                    if restaurant["distance"] < distance_threshold:
                        context_parts.append(f"- {restaurant['document']}")

            menus = await self.search_menus(query, limit=2, distance_threshold=distance_threshold)
            if menus:
                context_parts.append("Relevant menu information:")
                for menu in menus:
                    if menu["distance"] < distance_threshold:
                        context_parts.append(f"- {menu['document']}")

            context = "\n".join(context_parts)
            logger.info("Generated context for query: %s...", query[:50])
            return context

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics về các collection."""
        stats: Dict[str, int] = {}
        try:
            stats["conversations"] = self.client.count(
                collection_name=self.CONVERSATIONS_COLLECTION
            ).count
            stats["restaurants"] = self.client.count(
                collection_name=self.RESTAURANTS_COLLECTION
            ).count
            stats["menus"] = self.client.count(
                collection_name=self.MENUS_COLLECTION
            ).count
            stats["user_preferences"] = self.client.count(
                collection_name=self.USER_PREFERENCES_COLLECTION
            ).count
            stats["intents"] = self.client.count(
                collection_name=self.INTENTS_COLLECTION
            ).count
            stats["image_url"] = self.client.count(
                collection_name=self.IMAGE_URL_COLLECTION
            ).count
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
        return stats

    def _create_restaurant_searchable_text(self, restaurant: Dict) -> str:
        """Tạo đoạn text có thể tìm kiếm từ dữ liệu nhà hàng."""
        try:
            text_parts = []
            if restaurant.get("name"):
                text_parts.append(f"Tên nhà hàng: {restaurant['name']}")
            if restaurant.get("address"):
                text_parts.append(f"Địa chỉ: {restaurant['address']}")
            if restaurant.get("cuisineType"):
                text_parts.append(f"Loại ẩm thực: {restaurant['cuisineType']}")
            if restaurant.get("description"):
                text_parts.append(f"Mô tả: {restaurant['description']}")
            if restaurant.get("rating"):
                text_parts.append(f"Đánh giá: {restaurant['rating']}/5")
            if restaurant.get("priceRange"):
                text_parts.append(f"Khoảng giá: {restaurant['priceRange']}")

            searchable_terms = []
            if restaurant.get("cuisineType"):
                searchable_terms.extend(
                    [
                        f"nhà hàng {restaurant['cuisineType']}",
                        f"restaurant {restaurant['cuisineType']}",
                        f"ẩm thực {restaurant['cuisineType']}",
                    ]
                )
            if restaurant.get("name"):
                searchable_terms.append(f"quán {restaurant['name']}")
                searchable_terms.append(f"restaurant {restaurant['name']}")

            if searchable_terms:
                text_parts.append(f"Từ khóa tìm kiếm: {', '.join(searchable_terms)}")

            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error creating restaurant searchable text: {e}")
            return json.dumps(restaurant, ensure_ascii=False)

    def _create_service_searchable_text(self, service: Dict, restaurant_id: int) -> str:
        """Tạo đoạn text có thể tìm kiếm từ dữ liệu dịch vụ nhà hàng."""
        try:
            text_parts = []
            if service.get("name"):
                text_parts.append(f"Dịch vụ: {service['name']}")
            if service.get("description"):
                text_parts.append(f"Mô tả: {service['description']}")
            if service.get("price"):
                text_parts.append(f"Giá: {service['price']} VNĐ")
            if service.get("duration"):
                text_parts.append(f"Thời gian: {service['duration']}")

            searchable_terms = []
            if service.get("name"):
                searchable_terms.extend([
                    f"dịch vụ {service['name']}",
                    f"service {service['name']}",
                    f"tiện ích {service['name']}"
                ])
            if service.get("category"):
                searchable_terms.extend([
                    f"dịch vụ {service['category']}",
                    f"service {service['category']}"
                ])

            return " ".join(text_parts + searchable_terms)
        except Exception as e:
            logger.error(f"Error creating service searchable text: {e}")
            return f"Service: {service.get('name', 'Unknown')}"

    def _create_table_layout_searchable_text(self, layout: Dict, restaurant_id: int) -> str:
        """Tạo đoạn text có thể tìm kiếm từ dữ liệu sơ đồ bàn."""
        try:
            text_parts = []
            if layout.get("tableType"):
                text_parts.append(f"Loại bàn: {layout['tableType']}")
            if layout.get("capacity"):
                text_parts.append(f"Sức chứa: {layout['capacity']} người")
            if layout.get("description"):
                text_parts.append(f"Mô tả: {layout['description']}")
            if layout.get("location"):
                text_parts.append(f"Vị trí: {layout['location']}")

            searchable_terms = []
            if layout.get("tableType"):
                searchable_terms.extend([
                    f"bàn {layout['tableType']}",
                    f"table {layout['tableType']}",
                    f"loại bàn {layout['tableType']}"
                ])
            if layout.get("capacity"):
                searchable_terms.extend([
                    f"bàn {layout['capacity']} người",
                    f"table for {layout['capacity']}",
                    f"sức chứa {layout['capacity']}"
                ])

            return " ".join(text_parts + searchable_terms)
        except Exception as e:
            logger.error(f"Error creating table layout searchable text: {e}")
            return f"Table Layout: {layout.get('tableType', 'Unknown')}"

    def _create_table_searchable_text(self, table: Dict, restaurant_id: int) -> str:
        """Tạo đoạn text có thể tìm kiếm từ dữ liệu bàn."""
        try:
            text_parts = []
            if table.get("tableName"):
                text_parts.append(f"Tên bàn: {table['tableName']}")
            if table.get("capacity"):
                text_parts.append(f"Sức chứa: {table['capacity']} người")
            if table.get("status"):
                text_parts.append(f"Trạng thái: {table['status']}")
            if table.get("depositAmount"):
                text_parts.append(f"Đặt cọc: {table['depositAmount']}")

            searchable_terms = []
            if table.get("tableName"):
                searchable_terms.extend(
                    [
                        f"bàn {table['tableName']}",
                        f"table {table['tableName']}",
                    ]
                )
            if table.get("capacity"):
                searchable_terms.extend(
                    [
                        f"bàn {table['capacity']} người",
                        f"table for {table['capacity']}",
                    ]
                )

            base_text = " \n".join(text_parts)
            if searchable_terms:
                base_text += "\n" + ", ".join(searchable_terms)

            if not base_text.strip():
                return json.dumps(table, ensure_ascii=False)
            return base_text

        except Exception as e:
            logger.error(f"Error creating table searchable text: {e}")
            return json.dumps(table, ensure_ascii=False)

    def _create_menu_searchable_text(self, dish: Dict, restaurant_id: int) -> str:
        """
        Tạo rich semantic text cho menu item với:
        - Nutritional info, health attributes
        - Dietary restrictions
        - Tags từ LLM (nếu có)
        - Contextual descriptions (nóng, lạnh, cay, chay...)
        """
        try:
            text_parts = []
            
            # Basic info
            if dish.get("name"):
                text_parts.append(f"Tên món: {dish['name']}")
            if dish.get("description"):
                text_parts.append(f"Mô tả: {dish['description']}")
            if dish.get("category"):
                text_parts.append(f"Loại: {dish['category']}")
            
            # Restaurant context (semantic boost)
            restaurant_name = dish.get("_restaurantName") or dish.get("restaurantName") or dish.get("restaurant_name")
            if restaurant_name:
                text_parts.append(f"Thuộc nhà hàng: {restaurant_name}")
            
            cuisine_type = dish.get("cuisineType") or dish.get("cuisine_type")
            if cuisine_type:
                text_parts.append(f"Ẩm thực: {cuisine_type}")
            
            # Tags-based semantic context (CHÍNH LÀ CÁI QUAN TRỌNG)
            tags = dish.get("tags", [])
            # Đảm bảo tags là list, không phải string
            if isinstance(tags, str):
                try:
                    import ast
                    tags = ast.literal_eval(tags) if tags.startswith("[") else [tags]
                except:
                    tags = [tags] if tags else []
            
            # ✅ Get ingredient_tags (MỚI - dùng cho dị ứng/kiêng khem)
            ingredient_tags = dish.get("ingredient_tags", [])
            if isinstance(ingredient_tags, str):
                try:
                    import ast
                    ingredient_tags = ast.literal_eval(ingredient_tags) if ingredient_tags.startswith("[") else [ingredient_tags]
                except:
                    ingredient_tags = [ingredient_tags] if ingredient_tags else []
            if not isinstance(ingredient_tags, list):
                ingredient_tags = []
            
            if isinstance(tags, list) and tags:
                tag_contexts = []
                
                # Health & nutritional tags
                if "high_protein" in tags:
                    tag_contexts.append("Giàu protein, phù hợp cho người tập gym, cần bổ sung đạm")
                if "low_fat" in tags:
                    tag_contexts.append("Ít dầu mỡ, ít béo, phù hợp ăn kiêng")
                if "low_carb" in tags:
                    tag_contexts.append("Ít tinh bột, low carb")
                if "light_meal" in tags:
                    tag_contexts.append("Món nhẹ, dễ tiêu, không quá no")
                if "good_when_sick" in tags:
                    tag_contexts.append("Phù hợp khi ốm, món nóng dễ tiêu, dễ nuốt")
                
                # Dietary restrictions
                if "vegetarian" in tags or dish.get("is_vegetarian"):
                    tag_contexts.append("Món chay, không có thịt")
                if "vegan" in tags:
                    tag_contexts.append("Món thuần chay, không sản phẩm động vật")
                if dish.get("is_spicy") or "spicy" in tags:
                    tag_contexts.append("Món cay, đậm đà")
                if "non_spicy" in tags or dish.get("is_non_spicy"):
                    tag_contexts.append("Món không cay, nhẹ nhàng")
                
                # Occasion tags
                if "comfort_food" in tags:
                    tag_contexts.append("Comfort food, món dễ chịu, thoải mái")
                if "celebration" in tags:
                    tag_contexts.append("Phù hợp cho dịp đặc biệt, tiệc tùng")
                
                if tag_contexts:
                    text_parts.append("Đặc điểm: " + ", ".join(tag_contexts))
            
            # ✅ Ingredient context (MỚI - cho semantic search về dị ứng/kiêng khem)
            if ingredient_tags:
                ingredient_map = {
                    "beef": "có thịt bò",
                    "pork": "có thịt heo",
                    "chicken": "có thịt gà",
                    "seafood": "có hải sản",
                    "shrimp": "có tôm",
                    "crab": "có cua",
                    "squid": "có mực",
                    "clam": "có nghêu/sò",
                    "fish": "có cá",
                    "egg": "có trứng",
                    "milk": "có sữa",
                    "peanut": "có đậu phộng",
                    "soy": "có đậu nành/đậu phụ"
                }
                ingredient_labels = [
                    ingredient_map.get(tag.lower(), tag) 
                    for tag in ingredient_tags 
                    if tag.lower() in ingredient_map
                ]
                if ingredient_labels:
                    text_parts.append(f"Nguyên liệu: {', '.join(ingredient_labels)}")
            
            # Temperature context (semantic cho query kiểu "trời lạnh ăn gì nóng")
            category = dish.get("category", "").lower()
            name_lower = (dish.get("name") or "").lower()
            desc_lower = (dish.get("description") or "").lower()
            
            hot_items = ["cháo", "soup", "canh", "lẩu", "phở", "bún", "miến", "nước dùng", "hot pot"]
            if any(item in category or item in name_lower or item in desc_lower for item in hot_items):
                text_parts.append("Món nóng, ấm bụng, phù hợp trời lạnh")
            
            # Price context (nếu có)
            if dish.get("price"):
                price = dish.get("price")
                # Có thể thêm context về giá nếu cần
                # text_parts.append(f"Giá: {price}")
            
            # Restaurant ID (metadata)
            text_parts.append(f"Nhà hàng ID: {restaurant_id}")
            
            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error creating menu searchable text: {e}")
            return json.dumps(dish, ensure_ascii=False)
    
    # ==================== INTENT EMBEDDING COLLECTION ====================
    
    async def delete_intent_embedding(self, intent_name: str) -> bool:
        """Xóa intent embedding khỏi Vector DB"""
        try:
            # Tìm point ID của intent
            point_id = self._make_point_id(self.INTENTS_COLLECTION, intent_name)
            
            # Xóa point
            self.client.delete(
                collection_name=self.INTENTS_COLLECTION,
                points_selector=PointIdsList(points=[point_id])
            )
            
            logger.info(f"Deleted intent embedding: {intent_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting intent embedding {intent_name}: {e}")
            return False
    
    async def clear_all_intent_embeddings(self) -> bool:
        """Xóa tất cả intent embeddings (để reset)"""
        try:
            # Lấy tất cả points trong collection
            points, _ = self.client.scroll(
                collection_name=self.INTENTS_COLLECTION,
                limit=1000  # Lấy tất cả
            )
            
            if points:
                point_ids = [point.id for point in points]
                self.client.delete(
                    collection_name=self.INTENTS_COLLECTION,
                    points_selector=PointIdsList(points=point_ids)
                )
                logger.info(f"Cleared {len(point_ids)} intent embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing intent embeddings: {e}")
            return False
    
    async def store_intent_embedding(
        self, intent_name: str, examples: List[str], api_function: str = None
    ):
        """
        Store intent embedding với multiple examples
        
        Args:
            intent_name: Tên intent (e.g., "restaurant_search")
            examples: List các example messages cho intent này
            api_function: API function tương ứng (optional)
        """
        try:
            # Combine tất cả examples thành một text
            combined_text = ", ".join(examples)
            
            # Create embedding
            vector = self.encode_text(combined_text)
            if not vector:
                logger.warning(f"Failed to encode intent: {intent_name}")
                return
            
            # Create point ID từ intent name
            point_id = self._make_point_id(self.INTENTS_COLLECTION, intent_name)
            
            payload = {
                "intent": intent_name,
                "api_function": api_function,
                "examples": examples,
                "document": combined_text,
                "stored_at": str(int(time.time())),
            }
            
            self.client.upsert(
                collection_name=self.INTENTS_COLLECTION,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            
            logger.info(f"Stored intent embedding: {intent_name} with {len(examples)} examples")
            
        except Exception as e:
            logger.error(f"Error storing intent embedding: {e}")
    
    async def search_intents(
        self, query: str, limit: int = 3, distance_threshold: float = 0.4
    ) -> List[Dict]:
        """
        Search intent embeddings
        
        Args:
            query: User message
            limit: Số lượng results
            distance_threshold: Distance threshold
            
        Returns:
            List of intent results với distance và metadata
        """
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []
            
            # Search với higher limit để filter sau
            results = self.client.search(
                collection_name=self.INTENTS_COLLECTION,
                query_vector=query_vector,
                limit=limit * 2,
            )
            
            formatted_results = self._format_results(results)
            
            # Filter theo distance threshold
            filtered_results = [
                r for r in formatted_results 
                if r["distance"] < distance_threshold
            ]
            
            filtered_results = filtered_results[:limit]
            
            logger.info(
                "Found %s intents (filtered from %s, threshold=%.2f) for query: %s",
                len(filtered_results),
                len(formatted_results),
                distance_threshold,
                query[:50]
            )
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching intents: {e}")
            return []
    
    async def initialize_intent_embeddings(self, intent_definitions: Dict[str, Dict]):
        """
        Initialize intent embeddings từ intent definitions
        
        Args:
            intent_definitions: Dict với intent definitions
        """
        try:
            logger.info("Initializing intent embeddings...")
            
            # Intent examples cho mỗi intent type
            intent_examples = {
                "restaurant_search": [
                    "tìm nhà hàng", "nhà hàng gần đây", "restaurant near me",
                    "địa điểm ăn", "chỗ ăn", "quán ăn", "tìm chỗ ăn",
                    "muốn ăn", "ăn gì", "ăn ở đâu", "đi ăn",
                    "đồ ăn", "ẩm thực", "cuisine", "loại ẩm thực",
                    "hôm nay muốn ăn", "tối nay đi ăn", "ăn đồ Hàn",
                    "nhà hàng Hàn Quốc", "quán Việt Nam", "restaurant Ý",
                    "ăn món Nhật", "châu á", "asian food", "Korean restaurant"
                ],
                "menu_inquiry": [
                    "thực đơn", "menu", "món ăn", "có gì ăn",
                    "món nào ngon", "specialty", "đặc sản", "món gì",
                    "xem menu", "danh sách món", "món ăn của nhà hàng"
                ],
                "table_inquiry": [
                    "bàn nào", "sơ đồ bàn", "loại bàn", "sức chứa",
                    "layout", "bàn", "table", "chỗ ngồi",
                    "phòng riêng", "không gian", "vị trí bàn"
                ],
                "voucher_inquiry": [
                    "voucher", "mã giảm giá", "discount", "promotion",
                    "khuyến mãi", "ưu đãi", "giảm giá", "coupon"
                ],
                "general_inquiry": [
                    "xin chào", "hello", "hi", "chào", "help",
                    "giá cả", "pricing", "thông tin", "info"
                ]
            }
            
            # Store embeddings cho mỗi intent
            for intent_name, intent_def in intent_definitions.items():
                examples = intent_examples.get(intent_name, [])
                if examples:
                    await self.store_intent_embedding(
                        intent_name,
                        examples,
                        intent_def.get("api_function")
                    )
            
            logger.info("Intent embeddings initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing intent embeddings: {e}")

    async def search_tables(
        self, query: str, restaurant_id: int = None, limit: int = 5, distance_threshold: float = 0.5
    ) -> List[Dict]:
        """Semantic search cho tables (menus collection, type=table)."""
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []
            
            # ✅ FIX: Bỏ filter parameter (Qdrant local mode không hỗ trợ)
            # Search nhiều hơn để có đủ kết quả sau khi filter
            results = self.client.search(
                collection_name=self.MENUS_COLLECTION,
                query_vector=query_vector,
                limit=limit * 5,
            )
            
            # Format results
            tables = [
                dict(result.payload, distance=result.score, id=result.id)
                for result in results
            ]
            
            # ✅ FIX: Manual filter theo type=table
            tables = [
                t for t in tables
                if t.get("type") == "table" or t.get("metadata", {}).get("type") == "table"
            ]
            
            # ✅ FIX: Manual filter theo restaurant_id nếu có
            if restaurant_id is not None:
                tables = [
                    t for t in tables
                    if str(t.get("restaurant_id")) == str(restaurant_id) or 
                       str(t.get("metadata", {}).get("restaurant_id")) == str(restaurant_id)
                ]
            
            # FILTER theo distance threshold - CHỈ lấy results "gần gần"
            filtered_tables = [
                t for t in tables
                if t.get("distance", 999) < distance_threshold
            ]
            
            # Giới hạn lại số lượng sau khi filter
            return filtered_tables[:limit]
        except Exception as e:
            logger.error(f"Error searching tables: {e}")
            return []
    async def search_table_layouts(
        self, query: str, restaurant_id: int = None, limit: int = 5, distance_threshold: float = 0.5
    ) -> List[Dict]:
        """Semantic search cho table_layout (image_url collection), luôn filter theo type=table_layout."""
        try:
            query_vector = self.encode_text(query)
            if not query_vector:
                return []
            
            # ✅ FIX: Bỏ filter parameter (Qdrant local mode không hỗ trợ)
            # Search nhiều hơn để có đủ kết quả sau khi filter
            results = self.client.search(
                collection_name=self.IMAGE_URL_COLLECTION,
                query_vector=query_vector,
                limit=limit * 5,
            )
            
            # Format results
            layouts = [
                dict(result.payload, distance=result.score, id=result.id)
                for result in results
            ]
            
            # ✅ FIX: Manual filter theo type=table_layout
            layouts = [
                l for l in layouts
                if l.get("type") == "table_layout" or l.get("metadata", {}).get("type") == "table_layout"
            ]
            
            # ✅ FIX: Manual filter theo restaurant_id nếu có
            if restaurant_id is not None:
                layouts = [
                    l for l in layouts
                    if str(l.get("restaurant_id")) == str(restaurant_id) or
                       str(l.get("metadata", {}).get("restaurant_id")) == str(restaurant_id)
                ]
            
            # FILTER theo distance threshold - CHỈ lấy results "gần gần"
            filtered_layouts = [
                l for l in layouts
                if l.get("distance", 999) < distance_threshold
            ]
            
            # Giới hạn lại số lượng sau khi filter
            return filtered_layouts[:limit]
        except Exception as e:
            logger.error(f"Error searching table_layouts/images: {e}")
            return []


# Global instance
vector_service = VectorService()
