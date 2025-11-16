"""
Menu Reasoning Service - LLM-based reasoning để sinh structured profile
Thay vì dùng keyword list, dùng ontology nhỏ (high_protein, low_fat, occasion, temperature...)
"""
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class MenuReasoningService:
    """Service để LLM reasoning và sinh structured profile từ user query"""
    
    def __init__(self):
        self.openai_client = None
        self._default_profile = {
            # Structured fields (để boost/filter cho cases phổ biến)
            "diet_profile": {
                "high_protein": False,
                "low_carb": False,
                "low_fat": False,
                "light_meal": False
            },
            "occasion": "any",  # "gym" | "sick" | "comfort" | "celebration" | "any"
            "temperature": "any",  # "hot" | "cold" | "any"
            "spice_level": "any",  # "mild" | "medium" | "spicy" | "any"
            "cuisine": [],  # ["vietnamese", "japanese", "korean", ...]
            "is_local_specialty": False,  # Đặc sản địa phương
            
            # Text fields (để semantic search không bao giờ bị thiếu)
            "goals": [],  # ["thử đặc sản địa phương", "ăn healthy", "không bị ngán", ...]
            "constraints_text": [],  # ["không cay", "không hải sản", "ít dầu mỡ", ...]
            "search_query": "",  # Câu query tối ưu cho semantic search
            "summary": ""  # Tóm tắt ngắn gọn nhu cầu
        }
    
    def _get_openai_client(self) -> Optional[OpenAI]:
        """Get OpenAI client (lazy init)"""
        if not self.openai_client:
            if settings.OPENAI_API_KEY:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                logger.warning("OpenAI API key not available for menu reasoning")
        return self.openai_client
    
    async def universal_query_reasoning(self, user_message: str) -> Dict[str, Any]:
        """
        LLM reasoning để sinh hybrid profile: structured fields + text fields
        
        Strategy:
        - Structured fields (boolean/enum) → boost/filter cho cases phổ biến
        - Text fields (goals, constraints_text, search_query, summary) → semantic search không bao giờ bị thiếu
        
        Args:
            user_message: User query về món ăn
            
        Returns:
            {
                # Structured fields (để boost/filter)
                "diet_profile": {...},
                "occasion": "...",
                "temperature": "...",
                "spice_level": "...",
                "cuisine": [...],
                "is_local_specialty": bool,
                
                # Text fields (để semantic search)
                "goals": [...],
                "constraints_text": [...],
                "search_query": "...",
                "summary": "..."
            }
        """
        try:
            client = self._get_openai_client()
            if not client:
                logger.warning("OpenAI client not available, returning default profile")
                return {**self._default_profile, "summary": user_message}
            
            # System prompt - Hybrid approach: structured + free-text
            system_prompt = """Bạn là hệ thống phân tích yêu cầu món ăn của user.

Nhiệm vụ: Phân tích câu hỏi của user và trả về profile HYBRID (structured + free-text) về nhu cầu món ăn.

NGUYÊN TẮC THIẾT KẾ:
- Structured fields (boolean/enum) → chỉ cho những chiều quan trọng để tối ưu boost/filter
- Text fields (goals, constraints_text, search_query) → capture mọi thứ không có trong structured fields
→ Không bao giờ bị "thiếu field thì bot ngu", vì semantic search dựa trên text tự do

Format JSON trả về:
{
    // ========== STRUCTURED FIELDS (để boost/filter cho cases phổ biến) ==========
    "diet_profile": {
        "high_protein": boolean,  // Món giàu đạm (cho người tập gym, cần protein)
        "low_carb": boolean,      // Ít tinh bột (low carb diet)
        "low_fat": boolean,       // Ít dầu mỡ, ít béo
        "light_meal": boolean     // Món nhẹ, dễ tiêu, không quá no
    },
    "occasion": string,  // "gym" | "sick" | "comfort" | "celebration" | "any"
    "temperature": string,  // "hot" | "cold" | "any" (món nóng, lạnh, hoặc không quan trọng)
    "spice_level": string,  // "mild" | "medium" | "spicy" | "any" (độ cay)
    "cuisine": [string],  // ["vietnamese", "japanese", "korean", "chinese", "western", ...] hoặc []
    "is_local_specialty": boolean,  // Có phải đặc sản địa phương không
    
    // ========== TEXT FIELDS (để semantic search không bao giờ bị thiếu) ==========
    "goals": [string],  // List mục tiêu/nhu cầu: ["thử đặc sản địa phương", "ăn healthy", "không bị ngán", "phù hợp khách du lịch", ...]
    "constraints_text": [string],  // List ràng buộc: ["không cay", "không hải sản", "ít dầu mỡ", "không có thịt", ...]
    "search_query": string,  // Câu query tối ưu cho semantic search (diễn đạt rõ ràng, đầy đủ nhu cầu)
    "summary": string  // Tóm tắt ngắn gọn nhu cầu: "User tập gym, cần món giàu protein, ít tinh bột"
}

LƯU Ý QUAN TRỌNG:
1. Structured fields (diet_profile, occasion, temperature, spice_level):
   - Chỉ set true/giá trị cụ thể nếu user thực sự yêu cầu/mention rõ ràng
   - Nếu không rõ → dùng giá trị mặc định ("any" hoặc false)
   - Occasion: "gym" = đang tập gym, "sick" = đang ốm/bệnh/cảm/cần món dễ tiêu, "comfort" = món dễ chịu, "celebration" = dịp đặc biệt
   - Temperature: "hot" = món nóng (trời lạnh, ốm, cần ấm bụng), "cold" = món lạnh (trời nóng), "any" = không quan trọng

2. Text fields (goals, constraints_text, search_query, summary):
   - QUAN TRỌNG: Capture MỌI THỨ user muốn, kể cả những gì không có trong structured fields
   - goals: Các mục tiêu/nhu cầu của user (dạng text tự do)
   - constraints_text: Các ràng buộc/yêu cầu cụ thể (dạng text tự do)
   - search_query: Câu query đầy đủ, tối ưu để semantic search hiểu được toàn bộ nhu cầu
   - summary: Tóm tắt ngắn gọn, bằng tiếng Việt

3. Xử lý edge cases:
   - Nếu user nói về tình trạng sức khỏe (ốm, bệnh, cảm, không khỏe, đau...): set occasion="sick", light_meal=true, temperature="hot" nếu không có yêu cầu khác
   
   - ✅ QUAN TRỌNG: Nếu user nói về vấn đề về da (sẹo, mụn, thâm, dị ứng da...) HOẶC vết mổ, mới phẫu thuật:
     * goals: ["ăn ngon", "không làm tình trạng da xấu đi", "ăn nhẹ nhàng, không quá kích ứng"]
     * diet_profile: low_fat=true, light_meal=true
     * constraints_text: [
         "không quá nhiều dầu mỡ",
         "hạn chế đồ chiên xào",
         "không quá cay",
         "không quá nhiều đường",
         "thanh đạm",
         "tránh thịt bò",  # ← THEO VĂN HÓA VIỆT NAM: Sẹo/vết mổ kiêng bò
         "tránh hải sản (tôm, cua, mực, nghêu, sò, ốc, sò điệp...)"  # ← THEO VĂN HÓA VIỆT NAM: Kiêng hải sản
       ]
     * search_query: "món ăn nhẹ, ít dầu mỡ, ít cay, tốt cho da, thanh đạm, tránh thịt bò và hải sản, nhưng vẫn ngon"
     * summary: "User bị sẹo/vết mổ/vấn đề da, muốn ăn ngon nhưng tránh đồ dầu mỡ, cay, bò, hải sản"
     * KHÔNG cần field "scar" hay "da" trong schema, chỉ cần map sang constraints về đồ ăn "đỡ hại da"
   
   - ✅ FALLBACK QUAN TRỌNG: Nếu user nhắc đến bệnh/tình trạng sức khỏe mà KHÔNG khớp với các rule cụ thể ở trên:
     * Áp dụng rule chung "health_generic_safe":
       * diet_profile: low_fat=true, light_meal=true
       * spice_level: "mild" (không cay)
       * constraints_text: [
           "không quá nhiều dầu mỡ",
           "hạn chế đồ chiên xào",
           "không quá cay",
           "thanh đạm"
         ]
       * search_query: "món ăn nhẹ, ít dầu mỡ, không cay, thanh đạm, tốt cho sức khỏe"
       * summary: "User có vấn đề sức khỏe, cần món nhẹ, ít dầu mỡ, không cay, thanh đạm"
     * → Đảm bảo BẤT KỲ health condition nào cũng được xử lý "tạm an toàn", không cần phải liệt kê hết các bệnh trên đời
   
   - Nếu có typo hoặc cách nói không rõ ràng: suy luận theo context và diễn đạt rõ trong text fields
   - Nếu có yêu cầu không có trong structured fields → đưa vào goals/constraints_text/search_query

QUAN TRỌNG - Các rule văn hóa Việt Nam về ăn uống:
- "Sẹo", "vết mổ", "mới phẫu thuật" → tránh thịt bò và hải sản (theo kinh nghiệm dân gian Việt Nam)
- "Đau dạ dày", "trào ngược", "bao tử" → tránh cay, dầu mỡ, đồ nóng, đồ chua, ít acid
- "Cảm", "ốm", "sốt" → món nóng, dễ tiêu, dễ nuốt (cháo, soup, canh)
- "Tập gym", "cần protein" → giàu protein, ít tinh bột, ít béo
- "Dị ứng tôm/cua/hải sản" → tránh hải sản (tôm, cua, mực, nghêu, sò, ốc...)
- "Tiểu đường", "đường huyết cao" → ít đường, ít tinh bột, tránh đồ ngọt
- "Huyết áp cao", "tăng huyết áp" → ít muối, ít mặn, ít cholesterol
- "Đau khớp", "gout" → tránh thực phẩm giàu purin (thịt đỏ, hải sản), ít acid
- "Viêm gan", "gan yếu" → tránh rượu, ít dầu mỡ, ít cay, thanh đạm
- "Sau phẫu thuật", "phục hồi" → món nhẹ, dễ tiêu, giàu protein, thanh đạm

QUAN TRỌNG - Rule chung cho health conditions (fallback khi không khớp rule cụ thể):
- BẤT KỲ vấn đề sức khỏe nào chưa có rule riêng → áp dụng "health_generic_safe":
  * diet_profile: low_fat=true, light_meal=true
  * spice_level: "mild"
  * constraints_text: ["không quá nhiều dầu mỡ", "hạn chế đồ chiên xào", "không quá cay", "thanh đạm"]
  * → Đảm bảo luôn an toàn, không gây hại thêm

Ví dụ chi tiết:

1. Query: "Tôi tập gym mà lười ăn, gợi ý món gì dễ ăn, đỡ ngán"
   → {
        "diet_profile": {"high_protein": true, "light_meal": true, "low_carb": false, "low_fat": false},
        "occasion": "gym",
        "temperature": "any",
        "spice_level": "any",
        "cuisine": [],
        "is_local_specialty": false,
        "goals": ["dễ ăn", "không ngán", "phù hợp người tập gym"],
        "constraints_text": [],
        "search_query": "món giàu protein, dễ ăn, không ngán, phù hợp người tập gym",
        "summary": "User tập gym, cần món giàu protein, dễ ăn, không ngán"
      }

2. Query: "Đặc sản Việt Nam dễ ăn, không quá lạ, phù hợp khách du lịch"
   → {
        "diet_profile": {"high_protein": false, "low_carb": false, "low_fat": false, "light_meal": true},
        "occasion": "any",
        "temperature": "any",
        "spice_level": "mild",
        "cuisine": ["vietnamese"],
        "is_local_specialty": true,
        "goals": ["thử đặc sản địa phương", "dễ ăn", "không quá lạ", "phù hợp khách du lịch"],
        "constraints_text": ["không cay quá", "dễ ăn cho người nước ngoài"],
        "search_query": "đặc sản Việt Nam dễ ăn, không quá lạ, phù hợp khách du lịch, không cay",
        "summary": "User muốn đặc sản Việt Nam dễ ăn, không quá lạ, phù hợp khách du lịch"
      }

3. Query: "Tôi đang bị sẹo, món ăn nào là phù hợp" (nếu context là về món ăn cho người không khỏe)
   → {
        "diet_profile": {"high_protein": false, "low_carb": false, "low_fat": false, "light_meal": true},
        "occasion": "sick",
        "temperature": "hot",
        "spice_level": "mild",
        "cuisine": [],
        "is_local_specialty": false,
        "goals": ["món dễ tiêu", "phù hợp khi ốm/bệnh"],
        "constraints_text": ["dễ nuốt", "không cay"],
        "search_query": "món ăn phù hợp khi ốm, dễ tiêu, nóng, dễ nuốt, không cay",
        "summary": "User đang ốm/bệnh, cần món dễ tiêu, nóng, dễ nuốt"
      }

5. Query: "Tôi bị sẹo hãy giới thiệu cho tôi chỗ ăn ngon mà không bị sao" (vấn đề về da)
   → {
        "diet_profile": {"high_protein": false, "low_carb": false, "low_fat": true, "light_meal": true},
        "occasion": "any",
        "temperature": "any",
        "spice_level": "mild",
        "cuisine": [],
        "is_local_specialty": false,
        "goals": ["ăn ngon", "không làm tình trạng da xấu đi", "ăn nhẹ nhàng, không quá kích ứng"],
        "constraints_text": [
          "không quá nhiều dầu mỡ",
          "hạn chế đồ chiên xào",
          "không quá cay",
          "không quá nhiều đường",
          "thanh đạm",
          "tránh thịt bò",
          "tránh hải sản (tôm, cua, mực, nghêu, sò, ốc, sò điệp...)"
        ],
        "search_query": "món ăn nhẹ, ít dầu mỡ, ít cay, tốt cho da, thanh đạm, tránh thịt bò và hải sản, nhưng vẫn ngon",
        "summary": "User bị sẹo/vấn đề da, muốn ăn ngon nhưng tránh đồ dầu mỡ, cay, bò, hải sản"
      }

6. Query: "Tôi mới mổ, còn sẹo, muốn ăn gì cho an toàn?"
   → {
        "diet_profile": {"high_protein": false, "low_carb": false, "low_fat": true, "light_meal": true},
        "occasion": "any",
        "temperature": "any",
        "spice_level": "mild",
        "cuisine": [],
        "is_local_specialty": false,
        "goals": ["ăn an toàn", "không làm vết mổ/sẹo xấu đi", "ăn nhẹ nhàng"],
        "constraints_text": [
          "tránh thịt bò",
          "tránh hải sản (tôm, cua, mực, nghêu, sò, ốc, sò điệp...)",
          "không quá nhiều dầu mỡ",
          "không quá cay",
          "thanh đạm"
        ],
        "search_query": "món ăn an toàn sau phẫu thuật, nhẹ, ít dầu mỡ, ít cay, tránh thịt bò và hải sản, thanh đạm",
        "summary": "User mới mổ, còn sẹo, cần món an toàn, tránh bò và hải sản"
      }

4. Query: "Món hợp người lớn tuổi huyết áp cao, ít cholesterol nhưng vẫn ngon miệng"
   → {
        "diet_profile": {"high_protein": false, "low_carb": false, "low_fat": true, "light_meal": true},
        "occasion": "any",
        "temperature": "any",
        "spice_level": "mild",
        "cuisine": [],
        "is_local_specialty": false,
        "goals": ["phù hợp người lớn tuổi", "tốt cho sức khỏe", "ngon miệng"],
        "constraints_text": ["ít cholesterol", "không quá mặn", "không quá cay", "tốt cho huyết áp cao"],
        "search_query": "món ăn phù hợp người lớn tuổi huyết áp cao, ít cholesterol, nhẹ nhàng, không quá mặn, vẫn ngon miệng",
        "summary": "User cần món phù hợp người lớn tuổi huyết áp cao, ít cholesterol, vẫn ngon miệng"
      }

LƯU Ý CUỐI: 
- QUAN TRỌNG NHẤT: search_query phải diễn đạt ĐẦY ĐỦ và RÕ RÀNG toàn bộ nhu cầu, kể cả những gì không có trong structured fields
- goals và constraints_text giúp capture thêm context
- Structured fields chỉ là "shortcut" để boost/filter, không phải giới hạn của hệ thống
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Call OpenAI với JSON response format
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.2,  # Low temperature để reasoning chính xác
                max_tokens=500,  # Tăng lên vì có thêm text fields
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                try:
                    result = json.loads(content)
                    
                    # Validate và normalize result
                    validated_result = self._validate_profile(result)
                    
                    # Fallback: nếu search_query không có, tạo từ user_message + summary
                    if not validated_result.get("search_query"):
                        summary = validated_result.get("summary", "")
                        if summary:
                            validated_result["search_query"] = summary
                        else:
                            validated_result["search_query"] = user_message
                    
                    logger.info(f"Menu reasoning result: summary='{validated_result['summary']}', search_query='{validated_result.get('search_query', '')[:50]}...'")
                    return validated_result
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM reasoning JSON: {content}, error: {e}")
                    fallback = {**self._default_profile, "summary": user_message, "search_query": user_message}
                    return fallback
            
            # Fallback: return default với search_query
            fallback = {**self._default_profile, "summary": user_message, "search_query": user_message}
            return fallback
            
        except Exception as e:
            logger.error(f"Error in universal_query_reasoning: {e}", exc_info=True)
            fallback = {**self._default_profile, "summary": user_message, "search_query": user_message}
            return fallback
    
    def _validate_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate và normalize profile từ LLM (hybrid approach)"""
        validated = {**self._default_profile}
        
        # ========== Validate structured fields ==========
        
        # Validate diet_profile
        if isinstance(profile.get("diet_profile"), dict):
            diet_profile = profile.get("diet_profile")
            validated["diet_profile"] = {
                "high_protein": bool(diet_profile.get("high_protein", False)),
                "low_carb": bool(diet_profile.get("low_carb", False)),
                "low_fat": bool(diet_profile.get("low_fat", False)),
                "light_meal": bool(diet_profile.get("light_meal", False))
            }
        
        # Validate occasion
        occasion = profile.get("occasion", "any")
        valid_occasions = ["gym", "sick", "comfort", "celebration", "any"]
        validated["occasion"] = occasion if occasion in valid_occasions else "any"
        
        # Validate temperature
        temperature = profile.get("temperature", "any")
        valid_temperatures = ["hot", "cold", "any"]
        validated["temperature"] = temperature if temperature in valid_temperatures else "any"
        
        # Validate spice_level
        spice_level = profile.get("spice_level", "any")
        valid_spice_levels = ["mild", "medium", "spicy", "any"]
        validated["spice_level"] = spice_level if spice_level in valid_spice_levels else "any"
        
        # Validate cuisine
        cuisine = profile.get("cuisine", [])
        if isinstance(cuisine, list):
            validated["cuisine"] = [str(c) for c in cuisine if c]
        else:
            validated["cuisine"] = []
        
        # Validate is_local_specialty
        validated["is_local_specialty"] = bool(profile.get("is_local_specialty", False))
        
        # ========== Validate text fields ==========
        
        # Validate goals
        goals = profile.get("goals", [])
        if isinstance(goals, list):
            validated["goals"] = [str(g) for g in goals if g]
        else:
            validated["goals"] = []
        
        # Validate constraints_text (backward compatibility: cũng accept "constraints")
        constraints_text = profile.get("constraints_text", profile.get("constraints", []))
        if isinstance(constraints_text, list):
            validated["constraints_text"] = [str(c) for c in constraints_text if c]
        else:
            validated["constraints_text"] = []
        
        # Validate search_query
        search_query = profile.get("search_query", "").strip()
        if not search_query:
            # Fallback: tạo search_query từ user_message nếu LLM không generate
            # (sẽ được set sau trong code caller nếu cần)
            validated["search_query"] = ""
        else:
            validated["search_query"] = search_query
        
        # Validate summary
        validated["summary"] = str(profile.get("summary", "")).strip()
        
        return validated


# Global instance
menu_reasoning_service = MenuReasoningService()

