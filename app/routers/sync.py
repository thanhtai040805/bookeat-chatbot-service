import logging
from typing import Any, Dict, Iterable, Optional

from fastapi import APIRouter, HTTPException, status

from app.models.sync_event import SyncAction, SyncEvent
from app.services.vector_service import vector_service

router = APIRouter(prefix="/sync", tags=["Sync"])
logger = logging.getLogger("sync_router")


def _normalise_resource(resource: str) -> str:
    return resource.strip().lower()


def _coerce_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Thiếu trường '{field_name}' trong payload đồng bộ.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Giá trị '{field_name}' không hợp lệ: {value}") from exc


def _find_first(container: Optional[Dict[str, Any]], keys: Iterable[str]) -> Optional[Any]:
    if not container:
        return None
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


@router.post("/event", status_code=status.HTTP_202_ACCEPTED)
async def receive_sync_event(event: SyncEvent):
    """Nhận sự kiện CRUD từ Spring backend và đồng bộ vào vector store."""
    resource = _normalise_resource(event.resourceType)
    logger.info(
        "Received sync event %s (%s %s)",
        event.eventId,
        resource,
        event.action.value,
    )

    try:
        if resource == "restaurant":
            restaurant_id = _coerce_int(
                _find_first(event.data, ("id", "restaurantId", "restaurant_id"))
                or _find_first(event.metadata, ("id", "restaurantId", "restaurant_id")),
                "restaurantId",
            )
            if event.action == SyncAction.DELETE:
                await vector_service.delete_restaurant(restaurant_id)
            else:
                payload = dict(event.data)
                payload.setdefault("id", restaurant_id)
                await vector_service.upsert_restaurant(payload)

        elif resource in {"menu", "dish"}:
            restaurant_id = _coerce_int(
                _find_first(event.data, ("restaurantId", "restaurant_id"))
                or _find_first(event.metadata, ("restaurantId", "restaurant_id")),
                "restaurantId",
            )
            dish_id = _coerce_int(
                _find_first(event.data, ("id", "dishId", "dish_id"))
                or _find_first(event.metadata, ("id", "dishId", "dish_id")),
                "menuId",
            )

            if event.action == SyncAction.DELETE:
                await vector_service.delete_menu(restaurant_id, dish_id)
            else:
                payload = dict(event.data)
                payload.setdefault("id", dish_id)
                await vector_service.upsert_menu(restaurant_id, payload)

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"resourceType '{event.resourceType}' chưa được hỗ trợ.",
            )

    except ValueError as exc:
        logger.warning("Sync event %s bị từ chối: %s", event.eventId, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - phòng thủ
        logger.exception("Sync event %s lỗi: %s", event.eventId, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Sync error") from exc

    return {"status": "accepted", "eventId": event.eventId}

