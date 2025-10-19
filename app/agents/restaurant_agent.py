import asyncio
import logging
import unicodedata
from typing import Dict, List, Optional
from uuid import uuid4

from openai import OpenAI

from app.core.config import settings
from app.models import MessageRequest, MessageResponse

LOGGER = logging.getLogger("restaurant_agent")
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = (
    "You are RestaurantBot, a friendly AI concierge for La Petite Ourson, "
    "a modern Vietnamese fusion restaurant. Greet guests warmly, keep answers "
    "concise, default to Vietnamese unless the user clearly uses another "
    "language, and offer help with reservations, menu questions, and venue info."
)
MAX_HISTORY_MESSAGES = 12

_conversations: Dict[str, List[Dict[str, str]]] = {}
_openai_client: Optional[OpenAI] = None


async def handle_message(payload: MessageRequest) -> MessageResponse:
    """
    Xử lý message từ Spring Boot theo API spec
    Nhận message, userId, timestamp và trả về AI response
    """
    conversation_id = f"{payload.userId}_{payload.timestamp}"
    messages = _build_messages(conversation_id, payload.message)
    reply = await _call_openai(messages)
    source = "openai"

    if not reply:
        reply = _fallback(payload.message)
        source = "fallback"

    _store(conversation_id, payload.message, reply)
    return MessageResponse(response=reply)


def _build_messages(conversation_id: str, user_message: str) -> List[Dict[str, str]]:
    history = _conversations.get(conversation_id, [])[-MAX_HISTORY_MESSAGES:]
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_message}]


async def _call_openai(messages: List[Dict[str, str]]) -> Optional[str]:
    client = _get_openai_client()
    if client is None:
        return None

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=settings.OPENAI_TEMPERATURE,
        )
    except Exception as exc:
        LOGGER.warning("OpenAI call failed: %s", exc)
        return None

    if not completion.choices:
        return None

    reply = completion.choices[0].message.content
    return reply.strip() if reply else None


def _get_openai_client() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        LOGGER.info("OPENAI_API_KEY not set; falling back to heuristic responses.")
        return None

    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _fallback(user_message: str) -> str:
    normalized = unicodedata.normalize("NFD", user_message).encode("ascii", "ignore").decode("ascii").lower()

    if any(keyword in normalized for keyword in ("gio", "open", "hour")):
        return "Nha hang mo cua tu 10:00 den 22:00 moi ngay. Ban muon dat ban khung gio nao?"
    if any(keyword in normalized for keyword in ("dat ban", "reservation", "book")):
        return "Toi co the ho tro dat ban. Cho toi biet so khach, ngay va gio mong muon nhe."
    if any(keyword in normalized for keyword in ("menu", "mon", "dish")):
        return "Thuc don noi bat gom pho bo wagyu, goi cuon tom cang va cocktail tra sen. Ban muon tim hieu them mon nao?"
    if any(keyword in normalized for keyword in ("dia chi", "o dau", "location", "park")):
        return "La Petite Ourson nam tai 45A Nguyen Thi Minh Khai, Quan 1, TP.HCM. Bai gui xe nam ngay sau nha hang."

    return "Toi la tro ly cua La Petite Ourson. Toi co the giup ban voi viec dat ban, thuc don hoac huong dan den nha hang. Ban can ho tro gi?"


def _store(conversation_id: str, user_message: str, reply: str) -> None:
    conversation = _conversations.setdefault(conversation_id, [])
    conversation.extend(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
    )
