"""Pydantic models cho sự kiện đồng bộ dữ liệu từ Spring backend."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class SyncAction(str, Enum):
    """Các loại hành động CRUD từ backend gửi sang."""

    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class SyncEvent(BaseModel):
    """Payload chuẩn cho webhook đồng bộ dữ liệu."""

    eventId: str = Field(..., description="Định danh duy nhất của sự kiện.")
    resourceType: str = Field(..., description="Loại tài nguyên (restaurant, menu, ...).")
    action: SyncAction = Field(..., description="Hành động CRUD.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Dữ liệu mới nhất của tài nguyên.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Thông tin bổ sung (vd: restaurantId cho menu)."
    )
    timestamp: Optional[int] = Field(
        default=None, description="Unix timestamp thời điểm sự kiện diễn ra."
    )

