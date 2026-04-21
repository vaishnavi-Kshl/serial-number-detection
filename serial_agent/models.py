from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AssetRecord:
    asset_id: str
    document_id: Optional[str] = None
    asset_name: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    document_text: Optional[str] = None
    asset_location: Optional[str] = None
    serial_number_location: Optional[str] = None
    serial_number_series: Optional[str] = None
    serial_number_guide: Optional[str] = None
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    ai_attributes: Optional[str] = None
    source_pdf: Optional[str] = None
    source_page: Optional[int] = None
    raw_excerpt: Optional[str] = None
    record_set: Optional[str] = None
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["document_id"] = self.document_id
        payload["model_id"] = self.model_number
        payload["manufacturer_name"] = self.manufacturer
        payload["location"] = self.asset_location if self.record_set == "a" else self.serial_number_location
        payload["asset_location"] = self.asset_location
        payload["location_guide"] = self.serial_number_guide
        payload["document_text"] = self.document_text
        payload["set_name"] = self.record_set
        payload["model_number_normalized"] = _normalize_text(self.model_number)
        payload["manufacturer_normalized"] = _normalize_text(self.manufacturer)
        payload["category_normalized"] = _normalize_text(self.category)
        payload["asset_id_normalized"] = _normalize_text(self.asset_id)
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "AssetRecord":
        allowed = {
            "asset_id",
            "document_id",
            "asset_name",
            "model_number",
            "manufacturer",
            "category",
            "document_text",
            "asset_location",
            "serial_number_location",
            "location",
            "serial_number_series",
            "serial_number_guide",
            "location_guide",
            "image_id",
            "image_path",
            "ai_attributes",
            "source_pdf",
            "source_page",
            "raw_excerpt",
            "record_set",
            "confidence",
            "extra",
        }
        data = {
            "asset_id": payload.get("asset_id") or payload.get("image_id") or payload.get("asset_name") or "",
            "document_id": payload.get("document_id"),
            "asset_name": payload.get("asset_name"),
            "model_number": payload.get("model_number") or payload.get("model_id"),
            "manufacturer": payload.get("manufacturer") or payload.get("manufacturer_name"),
            "category": payload.get("category"),
            "document_text": payload.get("document_text"),
            "asset_location": _text_payload_value(payload.get("asset_location"))
            or (
                _text_payload_value(payload.get("location"))
                if (payload.get("record_set") or payload.get("set_name")) == "a"
                else None
            ),
            "serial_number_location": _text_payload_value(payload.get("serial_number_location"))
            or (
                _text_payload_value(payload.get("location"))
                if (payload.get("record_set") or payload.get("set_name")) != "a"
                else None
            ),
            "serial_number_series": payload.get("serial_number_series"),
            "serial_number_guide": _text_payload_value(payload.get("serial_number_guide"))
            or _text_payload_value(payload.get("location_guide")),
            "image_id": payload.get("image_id"),
            "image_path": payload.get("image_path"),
            "ai_attributes": payload.get("ai_attributes"),
            "source_pdf": payload.get("source_pdf"),
            "source_page": payload.get("source_page"),
            "raw_excerpt": payload.get("raw_excerpt"),
            "record_set": payload.get("record_set") or payload.get("set_name"),
            "confidence": payload.get("confidence", 0.0),
            "extra": {},
        }
        for key, value in payload.items():
            if key == "location" and not isinstance(value, str):
                data["extra"][key] = value
                continue
            if key == "asset_location" and not isinstance(value, str):
                data["extra"][key] = value
                continue
            if key == "location_guide" and not isinstance(value, str):
                data["extra"][key] = value
                continue
            if key not in {
                "asset_id",
                "document_id",
                "asset_name",
                "model_number",
                "model_id",
                "manufacturer",
                "manufacturer_name",
                "category",
                "document_text",
                "asset_location",
                "serial_number_location",
                "location",
                "serial_number_guide",
                "location_guide",
                "image_id",
                "image_path",
                "ai_attributes",
                "set_name",
                "confidence",
                "extra",
            }:
                data["extra"][key] = value
        return cls(**data)  # type: ignore[arg-type]


@dataclass
class AssetQuery:
    asset_id: Optional[str] = None
    image_id: Optional[str] = None
    asset_name: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    serial_number: Optional[str] = None


@dataclass
class LookupResult:
    found: bool
    message: str
    asset: Optional[AssetRecord] = None
    candidates: List[AssetRecord] = field(default_factory=list)
    verified: bool = False
    needs_image: bool = False
    serial_extracted: Optional[str] = None


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(value.strip().lower().split()) or None


def _text_payload_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None
