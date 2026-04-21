from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Optional

from .models import AssetQuery, AssetRecord, LookupResult
from .ocr import extract_serial_from_image
from .parsing import parse_pdf_pages
from .repository import InMemorySerialRepository, SerialRepository, score_candidate
from .verification import verify_serial_number


@dataclass
class SerialNumberAgent:
    repository: SerialRepository

    @classmethod
    def with_memory(cls) -> "SerialNumberAgent":
        return cls(repository=InMemorySerialRepository())

    def train_a(self, record: AssetRecord) -> AssetRecord:
        self.repository.upsert_set_a(record)
        return record

    def train_b1(self, record: AssetRecord) -> AssetRecord:
        self.repository.upsert_b1(record)
        return record

    def train_b2(self, record: AssetRecord) -> AssetRecord:
        self.repository.upsert_b2(record)
        return record

    def ingest_pdf(
        self,
        pdf_path: str,
        *,
        asset_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> list[AssetRecord]:
        records = parse_pdf_pages(pdf_path, document_id=asset_id, source_name=source_name)
        for record in records:
            self.repository.upsert(record)
        return records

    def guide(self, question: str, **kwargs) -> LookupResult:
        query = parse_question(question, **kwargs)
        asset_record = self._best_asset_record(query)
        location_record = self._best_document_record(
            asset_record.asset_id if asset_record else query.asset_id or query.image_id,
            query,
            set_name="b2",
        )
        if not location_record and not asset_record:
            return LookupResult(
                found=False,
                message="No matching asset found in Qdrant set b2. Add the asset location guide first.",
            )

        merged_asset = _merge_asset_context(asset_record, location_record)
        guide_text = build_location_guide(merged_asset or location_record or asset_record)
        if not location_record and asset_record:
            guide_text = (
                f"{guide_text}. "
                "No matching serial-number location guide was found in Qdrant set b2."
            )
        return LookupResult(
            found=bool(location_record),
            message=guide_text,
            asset=merged_asset or location_record or asset_record,
            needs_image=True,
        )

    def verify_image(self, image_path: str, question: str, **kwargs) -> LookupResult:
        query = parse_question(question, **kwargs)
        asset_record = self._best_asset_record(query)
        try:
            extracted_serial = extract_serial_from_image(image_path)
        except Exception:
            return LookupResult(
                found=False,
                message="No matching asset found in Qdrant set b1. Add the serial number series first.",
                asset=asset_record,
                serial_extracted=None,
                verified=False,
            )

        if not extracted_serial:
            return LookupResult(
                found=False,
                message="No matching asset found in Qdrant set b1. Add the serial number series first.",
                asset=asset_record,
                serial_extracted=None,
                verified=False,
            )

        location_record = self._best_document_record(
            asset_record.asset_id if asset_record else query.asset_id or query.image_id,
            query,
            set_name="b2",
        )
        series_record = self._best_document_record(
            asset_record.asset_id if asset_record else query.asset_id or query.image_id,
            query,
            set_name="b1",
        )
        if not series_record:
            merged_asset = _merge_asset_context(asset_record, location_record)
            guide = location_record.serial_number_guide if location_record else None
            location = location_record.serial_number_location if location_record else (asset_record.asset_location if asset_record else None)
            parts = [f"Extracted serial number: {extracted_serial}"]
            if location or guide:
                parts.append(f"Location guide: {guide or location}")
            parts.append("No matching asset found in Qdrant set b1. Add the serial number series first.")
            return LookupResult(
                found=False,
                message=". ".join(parts),
                asset=merged_asset or location_record or asset_record,
                serial_extracted=extracted_serial,
                verified=False,
            )

        merged_asset = _merge_asset_context(asset_record, series_record or location_record)
        verification = verify_serial_number(extracted_serial, series_record.serial_number_series)
        message = build_verification_message(
            asset=merged_asset or series_record,
            extracted_serial=extracted_serial,
            verified=verification.matched,
            verification_reason=verification.reason,
            location_record=location_record,
        )
        return LookupResult(
            found=verification.matched,
            message=message,
            asset=merged_asset or series_record,
            verified=verification.matched,
            serial_extracted=extracted_serial,
        )

    def _best_asset_record(self, query: AssetQuery) -> Optional[AssetRecord]:
        for identifier in (query.asset_id, query.image_id):
            if not identifier:
                continue
            exact = self.repository.get_set_a(identifier)
            if exact:
                return exact
            candidates = self.repository.find_set_a_by_document_id(identifier)
            best = _best_candidate(_filter_records_for_set(candidates, "a"), query)
            if best:
                return best

        candidates = self.repository.find_set_a_candidates(query)
        return _best_candidate(_filter_records_for_set(candidates, "a"), query)

    def _best_document_record(
        self,
        document_id: Optional[str],
        query: AssetQuery,
        *,
        set_name: str,
    ) -> Optional[AssetRecord]:
        if document_id:
            exact = self._get_record_by_id(set_name, document_id)
            if exact:
                return exact
        if document_id:
            if set_name == "b1":
                candidates = self.repository.find_b1_by_document_id(document_id)
            elif set_name == "b2":
                candidates = self.repository.find_b2_by_document_id(document_id)
            else:
                candidates = self.repository.find_set_a_by_document_id(document_id)
            best = _best_candidate(_filter_records_for_set(candidates, set_name), query)
            if best:
                return best
        if set_name == "b1":
            candidates = self.repository.find_b1_candidates(query)
        elif set_name == "b2":
            candidates = self.repository.find_b2_candidates(query)
        else:
            candidates = self.repository.find_set_a_candidates(query)
        best = _best_candidate(_filter_records_for_set(candidates, set_name), query)
        return best

    def _get_record_by_id(self, set_name: str, asset_id: str) -> Optional[AssetRecord]:
        if set_name == "a":
            return self.repository.get_set_a(asset_id)
        if set_name == "b1":
            return self.repository.get_b1(asset_id)
        if set_name == "b2":
            return self.repository.get_b2(asset_id)
        return None


def parse_question(
    question: str,
    *,
    asset_id: Optional[str] = None,
    image_id: Optional[str] = None,
    asset_name: Optional[str] = None,
    model_number: Optional[str] = None,
    manufacturer: Optional[str] = None,
    category: Optional[str] = None,
    serial_number: Optional[str] = None,
) -> AssetQuery:
    return AssetQuery(
        asset_id=asset_id or extract_label(question, ["asset id", "image id", "id"]),
        image_id=image_id or extract_label(question, ["image id", "asset image id"]),
        asset_name=asset_name or extract_label(question, ["asset name", "name"]),
        model_number=model_number or extract_label(question, ["model id", "model number", "model no", "model"]),
        manufacturer=manufacturer or extract_label(question, ["manufacturer", "manufacturer name", "make", "brand"]),
        category=category or extract_label(question, ["category", "asset category", "type"]),
        serial_number=serial_number or extract_serial_number(question),
    )


def build_location_guide(record: AssetRecord) -> str:
    guide = record.serial_number_guide or record.serial_number_location or "Location guide not available"
    asset_bits = [
        f"Asset: {record.asset_name}" if record.asset_name else None,
        f"Model: {record.model_number}" if record.model_number else None,
        f"Manufacturer: {record.manufacturer}" if record.manufacturer else None,
        f"Category: {record.category}" if record.category else None,
        f"Asset location: {record.asset_location}" if record.asset_location else None,
    ]
    asset_text = " | ".join(bit for bit in asset_bits if bit)
    if asset_text:
        return (
            f"{asset_text}. "
            f"Serial number location guide: {guide}. "
            "Please upload a serial number image so I can extract and verify it."
        )
    return (
        f"Serial number location guide: {guide}. "
        "Please upload a serial number image so I can extract and verify it."
    )


def build_verification_message(
    *,
    asset: AssetRecord,
    extracted_serial: str,
    verified: bool,
    verification_reason: str,
    location_record: Optional[AssetRecord] = None,
) -> str:
    guide = location_record.serial_number_guide if location_record else None
    location = (
        location_record.serial_number_location
        if location_record and location_record.serial_number_location
        else asset.serial_number_location or asset.asset_location
    )
    parts = [
        f"Extracted serial number: {extracted_serial}",
        f"Verification: {'matched' if verified else 'did not match'}",
        f"Reason: {verification_reason}",
    ]
    if location:
        parts.append(f"Location guide: {location}")
    if guide and guide != location:
        parts.append(f"Guide: {guide}")
    return ". ".join(parts)


def extract_label(text: str, labels: list[str]) -> Optional[str]:
    for label in labels:
        pattern = rf"\b{re.escape(label)}\b\s*[:#\-]?\s*([^\?\.,;]+)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        value = truncate_after_connectors(value)
        if label in {"serial number", "serial no", "serial #"} and not looks_like_serial(value):
            continue
        if value:
            return value
    return None


def truncate_after_connectors(value: str) -> str:
    for separator in (" by ", " for ", " from ", " with ", " in ", " on ", " at "):
        match = re.search(rf"\s{separator.strip()}\s", value, flags=re.IGNORECASE)
        if match:
            value = value[: match.start()].strip()
    return value.strip()


def looks_like_serial(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered.startswith(("number", "location", "series", "where", "for", "from", "the location")):
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))


def extract_serial_number(text: str) -> Optional[str]:
    patterns = [
        r"\bserial(?:\s+number|\s+no\.?|\s+#)\b\s*[:#\-]?\s*([A-Za-z0-9][A-Za-z0-9._/\-]{1,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = truncate_after_connectors(match.group(1).strip())
        if looks_like_serial(value):
            return value
    return None


def _best_candidate(candidates: list[AssetRecord], query: AssetQuery) -> Optional[AssetRecord]:
    scored = [(score_candidate(query, candidate), candidate) for candidate in candidates]
    scored = [(score, candidate) for score, candidate in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else None


def _filter_records_for_set(records: list[AssetRecord], set_name: str) -> list[AssetRecord]:
    if set_name == "a":
        return records
    if set_name == "b1":
        return [record for record in records if _has_series_text(record)]
    if set_name == "b2":
        return [record for record in records if _has_location_text(record)]
    return records


def _has_series_text(record: AssetRecord) -> bool:
    return bool(record.serial_number_series and record.serial_number_series.strip())


def _has_location_text(record: AssetRecord) -> bool:
    if record.serial_number_location and record.serial_number_location.strip():
        return True
    if record.serial_number_guide and record.serial_number_guide.strip():
        return True
    return False


def _merge_asset_context(asset_context: Optional[AssetRecord], serial_record: Optional[AssetRecord] = None) -> Optional[AssetRecord]:
    if asset_context is None and serial_record is None:
        return None

    base = serial_record or asset_context
    if base is None:
        return None

    merged = replace(base)
    if asset_context:
        for field_name in [
            "document_id",
            "asset_name",
            "model_number",
            "manufacturer",
            "category",
            "asset_location",
            "image_id",
            "image_path",
            "ai_attributes",
            "source_pdf",
            "source_page",
        ]:
            value = getattr(asset_context, field_name)
            if value and not getattr(merged, field_name):
                setattr(merged, field_name, value)
        if asset_context.extra:
            merged.extra = {**asset_context.extra, **merged.extra}

    if serial_record:
        for field_name in [
            "serial_number_location",
            "serial_number_series",
            "serial_number_guide",
        ]:
            value = getattr(serial_record, field_name)
            if value:
                setattr(merged, field_name, value)
        if serial_record.extra:
            merged.extra = {**merged.extra, **serial_record.extra}

        if serial_record.record_set:
            merged.record_set = serial_record.record_set
        merged.confidence = max(merged.confidence, serial_record.confidence)
    elif asset_context and asset_context.record_set:
        merged.record_set = asset_context.record_set

    return merged
