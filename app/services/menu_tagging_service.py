"""
Menu Tagging Service - Offline background job để tag menu items với LLM
Gán tags: high_protein, low_fat, light_meal, good_when_sick, etc.
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from openai import OpenAI
from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.spring_api_client import spring_api_client

logger = logging.getLogger(__name__)


class MenuTaggingService:
    """Service để tag menu items offline với LLM"""
    
    def __init__(self):
        self.openai_client = None
    
    def _get_openai_client(self) -> Optional[OpenAI]:
        """Get OpenAI client (lazy init)"""
        if not self.openai_client:
            if settings.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                logger.warning("OpenAI API key not available for menu tagging")
        return self.openai_client
    
    async def tag_menu_item(self, dish: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag một menu item với LLM
        
        Args:
            dish: Dish data (name, description, category, price, ...)
            
        Returns:
            Dict với tags: ["high_protein", "low_fat", "light_meal", ...]
        """
        try:
            client = self._get_openai_client()
            if not client:
                logger.warning("OpenAI client not available for tagging")
                return dish  # Return unchanged
            
            # Build dish description for LLM
            dish_info = f"""
Tên món: {dish.get('name', 'N/A')}
Loại: {dish.get('category', 'N/A')}
Mô tả: {dish.get('description', 'N/A')}
Giá: {dish.get('price', 'N/A')}
"""
            
            system_prompt = """Bạn là hệ thống phân tích và tag món ăn.

Nhiệm vụ: Phân tích món ăn và trả về tags phù hợp.

Health / lifestyle tags:
- "high_protein": Món giàu đạm (thịt, cá, trứng, đậu...)
- "low_fat": Món ít dầu mỡ, ít béo
- "low_carb": Món ít tinh bột
- "light_meal": Món nhẹ, dễ tiêu, không quá no (cháo, soup, salad...)
- "good_when_sick": Phù hợp khi ốm (cháo, soup, món nóng dễ nuốt...)
- "comfort_food": Comfort food, món dễ chịu
- "celebration": Phù hợp dịp đặc biệt, tiệc tùng
- "vegetarian": Món chay (không có thịt)
- "vegan": Món thuần chay (không sản phẩm động vật)
- "spicy": Món cay
- "non_spicy": Món không cay

Ingredient tags (dùng cho dị ứng / kiêng khem):
- "beef": có thịt bò
- "pork": có thịt heo
- "chicken": có thịt gà
- "seafood": có hải sản nói chung (tôm, cua, mực, nghêu, sò, ốc...)
- "shrimp": có tôm
- "crab": có cua
- "squid": có mực
- "clam": nghêu/sò/chem chép
- "fish": cá
- "egg": trứng
- "milk": sữa/sản phẩm từ sữa
- "peanut": đậu phộng
- "soy": đậu nành/đậu phụ

Trả về JSON format:
{
    "tags": ["tag1", "tag2", ...],          // health / lifestyle tags
    "ingredient_tags": ["beef", "shrimp"],  // các nguyên liệu chính quan trọng
    "is_spicy": boolean,
    "is_vegetarian": boolean,
    "is_vegan": boolean,
    "reasoning": "Lý do tại sao tag như vậy"
}

Lưu ý:
- Chỉ tag những gì thực sự phù hợp (không tag quá nhiều)
- Nếu món có tên/chứa thịt, cá, trứng → có thể "high_protein"
- Nếu món là cháo, soup, canh → có thể "light_meal", "good_when_sick"
- Nếu món chiên, xào nhiều dầu → KHÔNG tag "low_fat"
- Ingredient tags:
  * Chỉ gắn khi chắc chắn từ tên/mô tả (vd: "bò", "beef" → "beef")
  * Nếu không chắc → bỏ qua, không đoán mò
  * Ưu tiên các nguyên liệu chính, quan trọng (không cần tag mọi thứ nhỏ nhặt)
- Reasoning để debug
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Phân tích món ăn này và trả về tags:\n{dish_info}"}
            ]
            
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.2,  # Low temperature để tagging chính xác
                max_tokens=400,  # Tăng lên vì có thêm ingredient_tags
                response_format={"type": "json_object"}
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                try:
                    result = json.loads(content)
                    tags = result.get("tags", [])
                    ingredient_tags = result.get("ingredient_tags", []) or []
                    
                    if isinstance(tags, list):
                        # Merge tags vào dish
                        dish["tags"] = tags
                        dish["ingredient_tags"] = ingredient_tags if isinstance(ingredient_tags, list) else []
                        dish["is_spicy"] = result.get("is_spicy", False)
                        dish["is_vegetarian"] = result.get("is_vegetarian", False)
                        dish["is_vegan"] = result.get("is_vegan", False)
                        
                        logger.debug(
                            f"Tagged dish {dish.get('name')}: tags={tags}, ingredient_tags={ingredient_tags}"
                        )
                        return dish
                    else:
                        logger.warning(f"Tags not a list: {tags}")
                        return dish
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tagging JSON: {content}, error: {e}")
                    return dish
            
            return dish
            
        except Exception as e:
            logger.error(f"Error tagging menu item: {e}", exc_info=True)
            return dish
    
    async def tag_all_menus_for_restaurant(self, restaurant_id: int, from_vector_db: bool = False) -> int:
        """
        Tag tất cả menu items của một restaurant
        
        Args:
            restaurant_id: Restaurant ID
            from_vector_db: Nếu True, lấy menu từ Vector DB thay vì Spring API
            
        Returns:
            Số lượng items đã tag
        """
        try:
            menu = []
            
            if from_vector_db:
                # Get menu from Vector DB
                logger.info(f"Getting menu from Vector DB for restaurant {restaurant_id}...")
                # Search all menus for this restaurant (use broad query)
                results = await vector_service.search_menus(
                    "*",  # Broad query để lấy tất cả
                    restaurant_id=restaurant_id,
                    limit=1000,  # Lấy nhiều items
                    distance_threshold=1.0  # High threshold để lấy tất cả
                )
                
                # Extract dishes from results
                menu = [result.get("metadata", {}) for result in results]
                
                if not menu:
                    logger.warning(f"No menu found in Vector DB for restaurant {restaurant_id}")
                    return 0
                
                logger.info(f"Found {len(menu)} items in Vector DB for restaurant {restaurant_id}")
            else:
                # Get menu from Spring API
                try:
                    menu = await spring_api_client.get_restaurant_menu(restaurant_id)
                    if not menu:
                        logger.warning(f"No menu found from Spring API for restaurant {restaurant_id}")
                        return 0
                except Exception as e:
                    logger.error(f"Error getting menu from Spring API: {e}")
                    logger.info(f"Trying to get menu from Vector DB instead...")
                    # Fallback to Vector DB
                    return await self.tag_all_menus_for_restaurant(restaurant_id, from_vector_db=True)
            
            tagged_count = 0
            
            # Tag từng dish (có thể parallel nếu cần)
            for dish in menu:
                try:
                    # Skip nếu đã có tags và ingredient_tags (tránh re-tag)
                    existing_tags = dish.get("tags", [])
                    existing_ingredient_tags = dish.get("ingredient_tags", [])
                    has_tags = existing_tags and isinstance(existing_tags, list) and len(existing_tags) > 0
                    has_ingredient_tags = existing_ingredient_tags and isinstance(existing_ingredient_tags, list) and len(existing_ingredient_tags) > 0
                    
                    # Chỉ skip nếu đã có cả tags VÀ ingredient_tags (để đảm bảo đã tag đầy đủ)
                    if has_tags and has_ingredient_tags:
                        logger.debug(f"Dish {dish.get('name', 'unknown')} already has tags and ingredient_tags, skipping...")
                        continue
                    
                    tagged_dish = await self.tag_menu_item(dish)
                    
                    # Update trong vector DB (re-index với tags mới)
                    await vector_service.store_menu_data(restaurant_id, [tagged_dish])
                    
                    tagged_count += 1
                    
                    # Log progress
                    if tagged_count % 10 == 0:
                        logger.info(f"Tagged {tagged_count}/{len(menu)} items for restaurant {restaurant_id}")
                    
                    # Rate limiting (tránh spam API)
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error tagging dish {dish.get('name', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Completed tagging {tagged_count}/{len(menu)} items for restaurant {restaurant_id}")
            return tagged_count
            
        except Exception as e:
            logger.error(f"Error tagging all menus for restaurant {restaurant_id}: {e}", exc_info=True)
            return 0
    
    async def tag_all_menus(self, from_vector_db: bool = False) -> Dict[str, int]:
        """
        Tag tất cả menu items của tất cả restaurants (background job)
        
        Args:
            from_vector_db: Nếu True, lấy restaurants/menus từ Vector DB thay vì Spring API
        
        Returns:
            Dict với stats: {"total_restaurants": X, "total_tagged": Y}
        """
        try:
            restaurants = []
            
            if from_vector_db:
                # Get restaurants from Vector DB
                logger.info("Getting restaurants from Vector DB...")
                # Search all restaurants (use broad query)
                results = await vector_service.search_restaurants(
                    "*",  # Broad query để lấy tất cả
                    limit=1000,  # Lấy nhiều restaurants
                    distance_threshold=1.0  # High threshold để lấy tất cả
                )
                
                # Extract restaurant IDs from results
                seen_ids = set()
                for result in results:
                    metadata = result.get("metadata", {})
                    rid = (
                        metadata.get("id")
                        or metadata.get("restaurantId")
                        or metadata.get("restaurantID")
                    )
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        restaurants.append({"id": rid})  # Minimal structure
                
                if not restaurants:
                    logger.warning("No restaurants found in Vector DB")
                    return {"total_restaurants": 0, "total_tagged": 0}
                
                logger.info(f"Found {len(restaurants)} restaurants in Vector DB")
            else:
                # Get all restaurants from Spring API
                try:
                    restaurants = await spring_api_client.get_all_restaurants()
                    if not restaurants:
                        logger.warning("No restaurants found from Spring API")
                        # Fallback to Vector DB
                        logger.info("Trying to get restaurants from Vector DB instead...")
                        return await self.tag_all_menus(from_vector_db=True)
                except Exception as e:
                    logger.error(f"Error getting restaurants from Spring API: {e}")
                    logger.info("Trying to get restaurants from Vector DB instead...")
                    # Fallback to Vector DB
                    return await self.tag_all_menus(from_vector_db=True)
            
            total_restaurants = len(restaurants)
            total_tagged = 0
            
            logger.info(f"Starting tagging job for {total_restaurants} restaurants...")
            
            for idx, restaurant in enumerate(restaurants, 1):
                restaurant_id = (
                    restaurant.get('id')
                    or restaurant.get('restaurantId')
                    or restaurant.get('restaurantID')
                )
                
                if not restaurant_id:
                    logger.warning(f"Restaurant missing ID: {restaurant}")
                    continue
                
                try:
                    tagged_count = await self.tag_all_menus_for_restaurant(restaurant_id, from_vector_db=from_vector_db)
                    total_tagged += tagged_count
                    
                    logger.info(f"Progress: {idx}/{total_restaurants} restaurants, {total_tagged} items tagged")
                    
                except Exception as e:
                    logger.error(f"Error tagging restaurant {restaurant_id}: {e}")
                    continue
            
            logger.info(f"Tagging job completed: {total_tagged} items tagged across {total_restaurants} restaurants")
            return {
                "total_restaurants": total_restaurants,
                "total_tagged": total_tagged
            }
            
        except Exception as e:
            logger.error(f"Error in tag_all_menus: {e}", exc_info=True)
            return {"total_restaurants": 0, "total_tagged": 0}


# Global instance
menu_tagging_service = MenuTaggingService()

