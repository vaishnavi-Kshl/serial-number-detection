from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_set_a_collection: str = "google_image_assets_beta"
    qdrant_b1_collection: str = "serial_b1"
    qdrant_b2_collection: str = "serial_b2"

    @classmethod
    def load(cls, env_path: Optional[str] = None) -> "AppConfig":
        _load_dotenv(env_path)
        config = cls(
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_set_a_collection=os.getenv("QDRANT_SET_A_COLLECTION", "google_image_assets_beta"),
            qdrant_b1_collection=os.getenv("QDRANT_B1_COLLECTION", "serial_b1"),
            qdrant_b2_collection=os.getenv("QDRANT_B2_COLLECTION", "serial_b2"),
        )
        if config.qdrant_set_a_collection in {config.qdrant_b1_collection, config.qdrant_b2_collection}:
            logger.warning(
                "QDRANT_SET_A_COLLECTION should be distinct from serial collections. "
                "Current value: %r",
                config.qdrant_set_a_collection,
            )
        if config.qdrant_b1_collection == config.qdrant_b2_collection:
            logger.warning(
                "QDRANT_B1_COLLECTION and QDRANT_B2_COLLECTION are both set to %r. "
                "Use two different collections if you want b1 and b2 separated.",
                config.qdrant_b1_collection,
            )
        return config


def _load_dotenv(env_path: Optional[str] = None) -> None:
    path = _resolve_env_path(env_path)
    if path is None or not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_env_path(env_path: Optional[str]) -> Optional[Path]:
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
