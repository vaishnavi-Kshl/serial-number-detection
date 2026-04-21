from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from typing import Optional
from urllib import error, request

try:  # pragma: no cover - optional fallback dependency
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

try:  # pragma: no cover - optional fallback dependencyk
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

OPENAI_SERIAL_PROMPT = (
    "Extract the serial number visible in this image. "
    "Return only the serial number value and nothing else. "
    "If you cannot clearly read a serial number, return null."
)


def extract_text_from_image(image_path: str) -> str:
    if Image is None or pytesseract is None:
        raise RuntimeError(
            "OCR dependencies are missing. Install pillow and pytesseract, and make sure the Tesseract binary is available."
        )
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except OSError as exc:
        raise RuntimeError(f"Unable to read image for OCR: {exc}") from exc


def extract_serial_from_image(image_path: str) -> Optional[str]:
    openai_serial = _extract_serial_with_openai(image_path)
    if openai_serial:
        return openai_serial

    try:
        text = extract_text_from_image(image_path)
    except RuntimeError as exc:
        logger.warning("Local OCR fallback unavailable: %s", exc)
        return None
    return extract_serial_from_text(text)


def _extract_serial_with_openai(image_path: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_SERIAL_MODEL") or os.getenv("MODEL_NAME") or "gpt-5"
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    except OSError as exc:
        logger.warning("Unable to read image for OpenAI extraction: %s", exc)
        return None

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OPENAI_SERIAL_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 32,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("OpenAI serial extraction failed: %s", detail)
        return None
    except OSError as exc:
        logger.warning("OpenAI serial extraction failed: %s", exc)
        return None

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        logger.warning("OpenAI serial extraction returned invalid JSON: %s", exc)
        return None

    choices = data.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        return None

    return _parse_serial_response(content)


def _parse_serial_response(content: str) -> Optional[str]:
    text = content.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"null", "none", "n/a", "not found", "no serial number"}:
        return None

    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("serial_number", "serial", "value", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate and candidate.lower() not in {"null", "none", "n/a"}:
                    return candidate
        return None

    serial = extract_serial_from_text(text)
    if serial:
        return serial

    cleaned = re.sub(
        r"^\s*(serial(?:\s+number)?|serial\s+no\.?|s/?n|sn)\s*[:#\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" \t\r\n\"'`.,;")
    if _looks_like_serial_candidate(cleaned):
        return cleaned
    return None


def extract_serial_from_text(text: str) -> Optional[str]:
    normalized = " ".join(text.split())
    patterns = [
        r"\b(?:serial(?:\s+number)?|serial\s+no\.?|s/?n|sn|imei)\b\s*[:#\-]?\s*([A-Z0-9][A-Z0-9._/\-]{3,})",
        r"\b([A-Z0-9][A-Z0-9._/\-]{5,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if _looks_like_serial(candidate):
                return candidate
    return None


def _looks_like_serial(value: str) -> bool:
    value = value.strip()
    if len(value) < 4:
        return False
    return bool(re.search(r"[A-Za-z]", value) and re.search(r"\d", value))


def _looks_like_serial_candidate(value: str) -> bool:
    value = value.strip()
    if len(value) < 3:
        return False
    if any(ch.isspace() for ch in value):
        return False
    return bool(re.search(r"[A-Za-z0-9]", value))
