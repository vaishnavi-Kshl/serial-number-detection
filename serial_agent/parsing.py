from __future__ import annotations

import os
import re
import uuid
from typing import List, Optional, Sequence, Tuple

from .models import AssetRecord


FIELD_PATTERNS = {
    "asset_name": [
        r"(?:item\s+name|asset\s+name|name)\s*[:\-]\s*(.+)",
    ],
    "model_number": [
        r"(?:item\s+model)\s*[:\-]\s*(.+)",
        r"(?:model(?:\s+number)?|model\s+no\.?|model\s+#)\s*[:\-]\s*(.+)",
        r"\bmodel\s*[:\-]\s*(.+)",
    ],
    "manufacturer": [
        r"(?:item\s+manufacturer)\s*[:\-]\s*(.+)",
        r"(?:manufacturer|make|brand)\s*[:\-]\s*(.+)",
    ],
    "category": [
        r"(?:category|asset\s+category|type)\s*[:\-]\s*(.+)",
    ],
    "serial_number_location": [
        r"(?is)(?:special\s+instructions)\s*[:\-]\s*(.+?)(?:\n\s*\n|(?:overall\s+photo|qr\s+code\s+placement|serial\s+number\s+location)\b|\Z)",
        r"(?:serial(?:\s+number)?\s+location|location\s+of\s+serial(?:\s+number)?|where\s+is\s+the\s+serial(?:\s+number)?)\s*[:\-]\s*(.+)",
        r"(?:serial(?:\s+number)?\s+located\s+at|serial(?:\s+number)?\s+is\s+located\s+at)\s*[:\-]?\s*(.+)",
        r"(?:special\s+instructions)\s*[:\-]\s*(.+)",
    ],
    "serial_number_series": [
        r"(?:serial(?:\s+number)?\s+series|serial\s+range|series)\s*[:\-]\s*(.+)",
    ],
}


def extract_pdf_text_by_page(pdf_path: str) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages: List[Tuple[int, str]] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((index + 1, text))
    return pages


def parse_pdf_pages(
    pdf_path: str,
    *,
    document_id: Optional[str] = None,
    source_name: Optional[str] = None,
) -> List[AssetRecord]:
    pages = extract_pdf_text_by_page(pdf_path)
    resolved_document_id = document_id or derive_document_id(source_name or pdf_path)
    records: List[AssetRecord] = []
    for page_number, text in pages:
        record = parse_text(text, source_pdf=source_name or pdf_path, source_page=page_number)
        record.document_id = resolved_document_id
        record.asset_id = derive_page_asset_id(resolved_document_id, page_number, record)
        record.source_page = page_number
        records.append(record)
    return records


def parse_pdf(pdf_path: str, asset_id: Optional[str] = None) -> AssetRecord:
    pages = extract_pdf_text_by_page(pdf_path)
    best = AssetRecord(asset_id=asset_id or derive_asset_id(source_pdf=pdf_path))
    field_pages = {}
    document_text_parts = []
    for page_number, text in pages:
        if text.strip():
            document_text_parts.append(f"[page {page_number}]\n{text.strip()}")
        candidate = parse_text(text, source_pdf=pdf_path, source_page=page_number)
        for field_name in FIELD_PATTERNS:
            candidate_value = getattr(candidate, field_name)
            current_value = getattr(best, field_name)
            if candidate_value and not current_value:
                setattr(best, field_name, candidate_value)
                field_pages[field_name] = page_number
        if candidate.raw_excerpt:
            best.raw_excerpt = (
                f"{best.raw_excerpt} | {candidate.raw_excerpt}" if best.raw_excerpt else candidate.raw_excerpt
            )
        if candidate.confidence > best.confidence:
            best.confidence = candidate.confidence
    if not asset_id:
        best.asset_id = derive_asset_id(best, source_pdf=pdf_path)
    best.source_pdf = pdf_path
    best.document_text = "\n\n".join(document_text_parts) or None
    if best.serial_number_location and field_pages.get("serial_number_location"):
        best.source_page = field_pages["serial_number_location"]
    elif best.source_page is None and field_pages:
        best.source_page = next(iter(field_pages.values()))
    best.confidence = round(
        sum(1 for field_name in FIELD_PATTERNS if getattr(best, field_name)) / len(FIELD_PATTERNS),
        2,
    )
    return best


def parse_text(text: str, source_pdf: Optional[str] = None, source_page: Optional[int] = None) -> AssetRecord:
    lines = [clean_line(line) for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    combined = "\n".join(nonempty)

    values = {}
    excerpts = []
    for field_name, patterns in FIELD_PATTERNS.items():
        value, excerpt = extract_first_match(combined, nonempty, patterns, field_name)
        if value:
            values[field_name] = value
        if excerpt:
            excerpts.append(excerpt)

    raw_excerpt = " | ".join(excerpts) if excerpts else None
    confidence = round(len(values) / len(FIELD_PATTERNS), 2)
    asset_id = derive_asset_id(
        model_number=values.get("model_number"),
        manufacturer=values.get("manufacturer"),
        category=values.get("category"),
        source_pdf=source_pdf,
        source_page=source_page,
    )
    return AssetRecord(
        asset_id=asset_id,
        asset_name=values.get("asset_name"),
        model_number=values.get("model_number"),
        manufacturer=values.get("manufacturer"),
        category=values.get("category"),
        serial_number_location=values.get("serial_number_location"),
        serial_number_series=values.get("serial_number_series"),
        source_pdf=source_pdf,
        source_page=source_page,
        document_text=combined or None,
        raw_excerpt=raw_excerpt,
        confidence=confidence,
    )


def extract_first_match(
    combined_text: str,
    lines: Sequence[str],
    patterns: Sequence[str],
    field_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    for pattern in patterns:
        match = re.search(pattern, combined_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = normalize_value(match.group(1))
            if value:
                return value, match.group(0).strip()

    # Fallback: scan line by line for the keyword and keep the full line.
    keyword = field_name.replace("_", " ")
    for line in lines:
        lowered = line.lower()
        if "serial" in lowered and "location" in lowered and field_name == "serial_number_location":
            return normalize_value(line.split(":", 1)[-1]), line
        if keyword in lowered:
            parts = line.split(":", 1)
            if len(parts) == 2:
                value = normalize_value(parts[1])
                if value:
                    return value, line
    return None, None


def derive_asset_id(
    record: Optional[AssetRecord] = None,
    *,
    model_number: Optional[str] = None,
    manufacturer: Optional[str] = None,
    category: Optional[str] = None,
    source_pdf: Optional[str] = None,
    source_page: Optional[int] = None,
) -> str:
    if record:
        model_number = model_number or record.model_number
        manufacturer = manufacturer or record.manufacturer
        category = category or record.category
        source_pdf = source_pdf or record.source_pdf
        source_page = source_page or record.source_page

    seed = " | ".join(
        value
        for value in [
            normalize_value(model_number),
            normalize_value(manufacturer),
            normalize_value(category),
            os.path.basename(source_pdf) if source_pdf else None,
            str(source_page) if source_page else None,
        ]
        if value
    )
    if not seed:
        seed = "serial-ai-agent"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def derive_document_id(source_name: Optional[str] = None, *, record: Optional[AssetRecord] = None) -> str:
    if record is not None:
        source_name = source_name or record.source_pdf or record.asset_name or record.model_number
    seed = normalize_value(source_name) or "serial-ai-agent"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def derive_page_asset_id(document_id: str, page_number: int, record: Optional[AssetRecord] = None) -> str:
    seed_parts = [document_id, str(page_number)]
    if record:
        seed_parts.extend(
            value
            for value in [
                normalize_value(record.asset_name),
                normalize_value(record.model_number),
                normalize_value(record.manufacturer),
            ]
            if value
        )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, " | ".join(seed_parts)))


def clean_line(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None
