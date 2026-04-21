from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Protocol

from .models import AssetQuery, AssetRecord
from .models import _normalize_text


class SerialRepository(Protocol):
    def upsert_set_a(self, record: AssetRecord) -> None: ...

    def upsert_b1(self, record: AssetRecord) -> None: ...

    def upsert_b2(self, record: AssetRecord) -> None: ...

    def get_set_a(self, asset_id: str) -> Optional[AssetRecord]: ...

    def get_b1(self, asset_id: str) -> Optional[AssetRecord]: ...

    def get_b2(self, asset_id: str) -> Optional[AssetRecord]: ...

    def find_set_a_by_document_id(self, document_id: str) -> List[AssetRecord]: ...

    def find_b1_by_document_id(self, document_id: str) -> List[AssetRecord]: ...

    def find_b2_by_document_id(self, document_id: str) -> List[AssetRecord]: ...

    def find_set_a_candidates(self, query: AssetQuery) -> List[AssetRecord]: ...

    def find_b1_candidates(self, query: AssetQuery) -> List[AssetRecord]: ...

    def find_b2_candidates(self, query: AssetQuery) -> List[AssetRecord]: ...

    def best_set_a(self, query: AssetQuery) -> Optional[AssetRecord]: ...

    def best_b1(self, query: AssetQuery) -> Optional[AssetRecord]: ...

    def best_b2(self, query: AssetQuery) -> Optional[AssetRecord]: ...


@dataclass
class InMemorySerialRepository:
    a_records: Dict[str, AssetRecord] = field(default_factory=dict)
    b1_records: Dict[str, AssetRecord] = field(default_factory=dict)
    b2_records: Dict[str, AssetRecord] = field(default_factory=dict)

    def upsert_set_a(self, record: AssetRecord) -> None:
        prepared = _prepare_record_for_set(record, "a")
        if prepared is None:
            return
        self.a_records[prepared.asset_id] = prepared

    def upsert_b1(self, record: AssetRecord) -> None:
        prepared = _prepare_record_for_set(record, "b1")
        if prepared is None:
            return
        self.b1_records[prepared.asset_id] = prepared

    def upsert_b2(self, record: AssetRecord) -> None:
        prepared = _prepare_record_for_set(record, "b2")
        if prepared is None:
            return
        self.b2_records[prepared.asset_id] = prepared

    def upsert(self, record: AssetRecord) -> None:
        if record.record_set == "a":
            self.upsert_set_a(record)
        elif record.record_set == "b1":
            self.upsert_b1(record)
        elif record.record_set == "b2":
            self.upsert_b2(record)
        else:
            if _has_asset_text(record) and not _has_serial_text(record):
                self.upsert_set_a(record)
            if record.serial_number_series:
                self.upsert_b1(record)
            if record.serial_number_location or record.serial_number_guide:
                self.upsert_b2(record)

    def get_set_a(self, asset_id: str) -> Optional[AssetRecord]:
        return self.a_records.get(asset_id)

    def get_b1(self, asset_id: str) -> Optional[AssetRecord]:
        return self.b1_records.get(asset_id)

    def get_b2(self, asset_id: str) -> Optional[AssetRecord]:
        return self.b2_records.get(asset_id)

    def find_set_a_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return [record for record in self.a_records.values() if record.document_id == document_id]

    def find_b1_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return [record for record in self.b1_records.values() if record.document_id == document_id]

    def find_b2_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return [record for record in self.b2_records.values() if record.document_id == document_id]

    def find_set_a_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return _rank(self.a_records.values(), query)

    def find_b1_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return _rank(self.b1_records.values(), query)

    def find_b2_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return _rank(self.b2_records.values(), query)

    def best_set_a(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_set_a_candidates(query)
        return candidates[0] if candidates else None

    def best_b1(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_b1_candidates(query)
        return candidates[0] if candidates else None

    def best_b2(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_b2_candidates(query)
        return candidates[0] if candidates else None


class QdrantSerialRepository:
    DEFAULT_VECTOR_SIZE = 1408

    def __init__(
        self,
        set_a_collection: str = "asset_set_a",
        b1_collection: str = "serial_b1",
        b2_collection: str = "serial_b2",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        ensure_collection: bool = True,
    ) -> None:
        self.set_a_collection = set_a_collection
        self.b1_collection = b1_collection
        self.b2_collection = b2_collection
        self.url = url
        self.api_key = api_key
        self.ensure_collection = ensure_collection
        self._client = None
        self._vector_sizes: Dict[str, int] = {}

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, api_key=self.api_key, check_compatibility=False)
        return self._client

    def upsert_set_a(self, record: AssetRecord) -> None:
        self._upsert(self.set_a_collection, record, "a")

    def upsert_b1(self, record: AssetRecord) -> None:
        self._upsert(self.b1_collection, record, "b1")

    def upsert_b2(self, record: AssetRecord) -> None:
        self._upsert(self.b2_collection, record, "b2")

    def upsert(self, record: AssetRecord) -> None:
        if record.record_set == "a":
            self.upsert_set_a(record)
        elif record.record_set == "b1":
            self.upsert_b1(record)
        elif record.record_set == "b2":
            self.upsert_b2(record)
        else:
            if _has_asset_text(record) and not _has_serial_text(record):
                self.upsert_set_a(record)
            if record.serial_number_series:
                self.upsert_b1(record)
            if record.serial_number_location or record.serial_number_guide:
                self.upsert_b2(record)

    def get_set_a(self, asset_id: str) -> Optional[AssetRecord]:
        points = self.client.retrieve(
            collection_name=self.set_a_collection,
            ids=[asset_id],
            with_payload=True,
            with_vectors=False,
        )
        return AssetRecord.from_payload(points[0].payload or {}) if points else None

    def get_b1(self, asset_id: str) -> Optional[AssetRecord]:
        points = self.client.retrieve(
            collection_name=self.b1_collection,
            ids=[asset_id],
            with_payload=True,
            with_vectors=False,
        )
        return AssetRecord.from_payload(points[0].payload or {}) if points else None

    def get_b2(self, asset_id: str) -> Optional[AssetRecord]:
        points = self.client.retrieve(
            collection_name=self.b2_collection,
            ids=[asset_id],
            with_payload=True,
            with_vectors=False,
        )
        return AssetRecord.from_payload(points[0].payload or {}) if points else None

    def find_b1_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return self._find_document_candidates(self.b1_collection, document_id)

    def find_b2_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return self._find_document_candidates(self.b2_collection, document_id)

    def find_set_a_by_document_id(self, document_id: str) -> List[AssetRecord]:
        return self._find_document_candidates(self.set_a_collection, document_id)

    def find_set_a_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return self._find_candidates(self.set_a_collection, query)

    def find_b1_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return self._find_candidates(self.b1_collection, query)

    def find_b2_candidates(self, query: AssetQuery) -> List[AssetRecord]:
        return self._find_candidates(self.b2_collection, query)

    def best_b1(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_b1_candidates(query)
        return candidates[0] if candidates else None

    def best_set_a(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_set_a_candidates(query)
        return candidates[0] if candidates else None

    def best_b2(self, query: AssetQuery) -> Optional[AssetRecord]:
        candidates = self.find_b2_candidates(query)
        return candidates[0] if candidates else None

    def _upsert(self, collection_name: str, record: AssetRecord, record_set: str) -> None:
        self._ensure_collection(collection_name)
        from qdrant_client.models import PointStruct

        prepared = _prepare_record_for_set(record, record_set)
        if prepared is None:
            return
        vector_size = self._collection_vector_size(collection_name)
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=prepared.asset_id,
                    vector=_dummy_vector(vector_size),
                    payload=prepared.to_payload(),
                )
            ],
        )

    def _find_candidates(self, collection_name: str, query: AssetQuery) -> List[AssetRecord]:
        records = self._all_records(collection_name)
        return _rank(records, query)

    def _find_document_candidates(self, collection_name: str, document_id: str) -> List[AssetRecord]:
        return [record for record in self._all_records(collection_name) if record.document_id == document_id]

    def _all_records(self, collection_name: str) -> List[AssetRecord]:
        out: List[AssetRecord] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                out.append(AssetRecord.from_payload(payload))
            if offset is None:
                break
        return out

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.ensure_collection:
            return
        if self.client.collection_exists(collection_name):
            return
        from qdrant_client.models import Distance, VectorParams

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.DEFAULT_VECTOR_SIZE, distance=Distance.DOT),
        )

    def _collection_vector_size(self, collection_name: str) -> int:
        cached = self._vector_sizes.get(collection_name)
        if cached:
            return cached

        try:
            collection = self.client.get_collection(collection_name)
            vectors = collection.config.params.vectors
            size = _extract_vector_size(vectors)
            if size:
                self._vector_sizes[collection_name] = size
                return size
        except Exception:
            pass

        self._vector_sizes[collection_name] = self.DEFAULT_VECTOR_SIZE
        return self.DEFAULT_VECTOR_SIZE


def _rank(records, query: AssetQuery) -> List[AssetRecord]:
    scored = []
    for record in records:
        score = score_candidate(query, record)
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


def score_candidate(query: AssetQuery, record: AssetRecord) -> float:
    score = 0.0
    score += _score_field(query.asset_id, record.asset_id, exact_weight=6.0)
    score += _score_field(query.image_id, record.image_id, exact_weight=5.5)
    score += _score_field(query.asset_name, record.asset_name, exact_weight=5.0)
    score += _score_field(query.model_number, record.model_number, exact_weight=4.0)
    score += _score_field(query.manufacturer, record.manufacturer, exact_weight=3.0)
    score += _score_field(query.category, record.category, exact_weight=2.0)
    return score


def _score_field(query_value: Optional[str], record_value: Optional[str], exact_weight: float) -> float:
    if not query_value or not record_value:
        return 0.0
    q = _normalize_text(query_value) or ""
    r = _normalize_text(record_value) or ""
    if not q or not r:
        return 0.0
    if q == r:
        return exact_weight
    if q in r or r in q:
        return exact_weight * 0.6
    q_tokens = set(q.split())
    r_tokens = set(r.split())
    if not q_tokens or not r_tokens:
        return 0.0
    overlap = len(q_tokens & r_tokens) / max(len(q_tokens), len(r_tokens))
    return exact_weight * overlap * 0.5


def _prepare_record_for_set(record: AssetRecord, record_set: str) -> Optional[AssetRecord]:
    if record_set == "a":
        prepared = replace(
            record,
            record_set="a",
            serial_number_location=None,
            serial_number_series=None,
            serial_number_guide=None,
        )
        return prepared
    if record_set == "b1":
        if not record.serial_number_series:
            return None
        prepared = replace(
            record,
            record_set="b1",
            serial_number_location=None,
            serial_number_guide=None,
        )
        return prepared
    if record_set == "b2":
        if not record.serial_number_location and not record.serial_number_guide:
            return None
        prepared = replace(
            record,
            record_set="b2",
            serial_number_series=None,
        )
        return prepared
    raise ValueError(f"Unknown record set: {record_set}")


def _has_asset_text(record: AssetRecord) -> bool:
    return bool(
        record.asset_name
        or record.model_number
        or record.manufacturer
        or record.category
        or record.image_id
        or record.image_path
        or record.ai_attributes
        or record.asset_location
    )


def _has_serial_text(record: AssetRecord) -> bool:
    return bool(record.serial_number_series or record.serial_number_location or record.serial_number_guide)


def _dummy_vector(size: int) -> List[float]:
    if size <= 0:
        size = 1
    return [1.0] + [0.0] * (size - 1)


def _extract_vector_size(vectors: Any) -> Optional[int]:
    if vectors is None:
        return None
    if hasattr(vectors, "size"):
        size = getattr(vectors, "size", None)
        return int(size) if size else None
    if isinstance(vectors, dict):
        for value in vectors.values():
            size = _extract_vector_size(value)
            if size:
                return size
    return None


# Backward-compatible aliases.
AssetRepository = SerialRepository
InMemoryAssetRepository = InMemorySerialRepository
QdrantAssetRepository = QdrantSerialRepository
