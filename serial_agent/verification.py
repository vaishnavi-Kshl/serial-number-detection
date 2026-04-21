from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    matched: bool
    reason: str


def verify_serial_number(serial_number: Optional[str], serial_series: Optional[str]) -> VerificationResult:
    if not serial_series:
        return VerificationResult(matched=True, reason="No serial series stored for verification.")
    if not serial_number:
        return VerificationResult(matched=False, reason="No serial number was supplied.")

    serial_number = serial_number.strip()
    serial_series = serial_series.strip()

    range_result = _match_numeric_range(serial_number, serial_series)
    if range_result is not None:
        return range_result

    if serial_series.lower() in serial_number.lower():
        return VerificationResult(matched=True, reason="Serial number contains the stored series label.")

    token_pattern = re.escape(serial_series)
    if re.search(token_pattern, serial_number, flags=re.IGNORECASE):
        return VerificationResult(matched=True, reason="Serial number matches the stored series text.")

    return VerificationResult(matched=False, reason="Serial number does not match the stored series.")


def _match_numeric_range(serial_number: str, serial_series: str) -> Optional[VerificationResult]:
    normalized = serial_series.replace(" ", "")
    match = re.fullmatch(r"([A-Za-z_-]*)(\d+)-([A-Za-z_-]*)(\d+)", normalized)
    if not match:
        return None

    prefix_a, start_s, prefix_b, end_s = match.groups()
    if prefix_a != prefix_b:
        return VerificationResult(matched=False, reason="The stored series range uses different prefixes.")

    digits = re.search(r"(\d+)$", serial_number)
    if not digits:
        return VerificationResult(matched=False, reason="Serial number has no numeric suffix to compare.")

    number = int(digits.group(1))
    start = int(start_s)
    end = int(end_s)
    prefix = serial_number[: digits.start(1)]
    if prefix_a and prefix.lower() != prefix_a.lower():
        return VerificationResult(matched=False, reason="Serial prefix does not match the stored series prefix.")

    if start <= number <= end:
        return VerificationResult(matched=True, reason="Serial number falls inside the stored series range.")
    return VerificationResult(matched=False, reason="Serial number is outside the stored series range.")

