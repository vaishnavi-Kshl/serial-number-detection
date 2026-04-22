from __future__ import annotations

import os
import tempfile
import uuid
import logging
from typing import Optional

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi import HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    File = Form = UploadFile = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from .agent import SerialNumberAgent
from .config import AppConfig
from .models import AssetRecord
from .repository import QdrantSerialRepository


logger = logging.getLogger(__name__)


class AssetBaseRequest(BaseModel):
    asset_id: Optional[str] = None
    image_id: Optional[str] = None
    asset_name: Optional[str] = None
    model_id: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer_name: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    image_path: Optional[str] = None
    document_text: Optional[str] = None
    ai_attributes: Optional[str] = None


class TrainB1Request(AssetBaseRequest):
    serial_number_location: str
    location_guide: Optional[str] = None


class TrainB2Request(AssetBaseRequest):
    serial_number_series: str


class GuideRequest(AssetBaseRequest):
    pass


class GuideResponse(BaseModel):
    found: bool
    needs_image: bool
    image_id: Optional[str] = None
    asset_id: Optional[str] = None
    document_id: Optional[str] = None
    asset_name: Optional[str] = None
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    category: Optional[str] = None
    asset_location: Optional[str] = None
    image_path: Optional[str] = None
    ai_attributes: Optional[str] = None
    message: str
    set_name: Optional[str] = None
    serial_number_location: Optional[str] = None


class GuideErrorResponse(BaseModel):
    detail: str


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed")

    config = AppConfig.load()
    app = FastAPI(title="Serial AI Agent")
    repository = QdrantSerialRepository(
        set_a_collection=config.qdrant_set_a_collection,
        b1_collection=config.qdrant_b1_collection,
        b2_collection=config.qdrant_b2_collection,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        ensure_collection=True,
    )
    agent = SerialNumberAgent(repository=repository)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/train/b1")
    def train_b1(req: TrainB1Request):
        record = _build_record(
            req,
            serial_number_location=req.serial_number_location,
            serial_number_guide=req.location_guide,
            record_set="b1",
        )
        saved = agent.train_b1(record)
        return _record_response(saved)

    @app.post("/train/a")
    def train_a(req: AssetBaseRequest):
        record = _build_record(req, asset_location=req.location, record_set="a")
        saved = agent.train_a(record)
        return _record_response(saved)

    @app.post("/train/b2")
    def train_b2(req: TrainB2Request):
        record = _build_record(
            req,
            serial_number_series=req.serial_number_series,
            record_set="b2",
        )
        saved = agent.train_b2(record)
        return _record_response(saved)

    @app.post("/ingest/pdf")
    async def ingest_pdf(
        file: UploadFile = File(...),
        qdrant_collection_name: Optional[str] = Form(None),
    ):
        suffix = os.path.splitext(file.filename or "asset.pdf")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        try:
            try:
                records = agent.ingest_pdf(
                    temp_path,
                    collection_name=qdrant_collection_name,
                    source_name=file.filename,
                )
                stored_in = []
                if any(record.serial_number_location or record.serial_number_guide for record in records):
                    stored_in.append("b1")
                if any(record.serial_number_series for record in records):
                    stored_in.append("b2")
                return {
                    "document_id": records[0].document_id if records else None,
                    "page_count": len(records),
                    "page_ids": [record.asset_id for record in records],
                    "stored_in": stored_in,
                    "pages": [
                        {
                            "page": record.source_page,
                            "asset_id": record.asset_id,
                            "asset_name": record.asset_name,
                            "model_number": record.model_number,
                            "manufacturer": record.manufacturer,
                            "serial_number_location": record.serial_number_location,
                            "serial_number_series": record.serial_number_series,
                            "record_set": record.record_set,
                            "confidence": record.confidence,
                        }
                        for record in records
                    ],
                }
            except Exception as exc:
                logger.exception("PDF ingest failed for %s", file.filename or "asset.pdf")
                raise HTTPException(
                    status_code=503,
                    detail=f"Unable to parse or store the PDF: {exc}",
                ) from exc
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @app.post(
        "/guide",
        response_model=GuideResponse,
        responses={
            422: {
                "description": "Validation error",
            },
            503: {
                "model": GuideErrorResponse,
                "description": "Qdrant or backend lookup failure",
            }
        },
    )
    def guide(req: GuideRequest):
        try:
            result = agent.guide(
                "",
                asset_id=req.asset_id,
                image_id=req.image_id,
                asset_name=req.asset_name,
                model_number=req.model_number,
                manufacturer=req.manufacturer or req.manufacturer_name,
                category=req.category,
            )
            return _guide_response(result)
        except Exception as exc:
            logger.exception("Guide lookup failed")
            raise HTTPException(
                status_code=503,
                detail=f"Unable to reach Qdrant: {exc}",
            ) from exc

    @app.post("/verify/image")
    async def verify_image(
        file: UploadFile = File(...),
        question: str = Form(""),
        asset_id: Optional[str] = Form(None),
        image_id: Optional[str] = Form(None),
        asset_name: Optional[str] = Form(None),
        model_id: Optional[str] = Form(None),
        model_number: Optional[str] = Form(None),
        manufacturer_name: Optional[str] = Form(None),
        manufacturer: Optional[str] = Form(None),
        category: Optional[str] = Form(None),
    ):
        suffix = os.path.splitext(file.filename or "serial.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
        try:
            try:
                result = agent.verify_image(
                    temp_path,
                    question,
                    asset_id=asset_id,
                    image_id=image_id,
                    asset_name=asset_name,
                    model_number=model_number or model_id,
                    manufacturer=manufacturer or manufacturer_name,
                    category=category,
                )
                return {
                    "found": result.found,
                    "verified": result.verified,
                    "needs_image": result.needs_image,
                    "serial_extracted": result.serial_extracted,
                    "message": result.message,
                    "asset": result.asset.to_payload() if result.asset else None,
                }
            except Exception as exc:
                logger.exception("Image verification failed")
                raise HTTPException(
                    status_code=503,
                    detail=f"Unable to reach Qdrant: {exc}",
                ) from exc
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return app


def _build_record(req: AssetBaseRequest, *, record_set: str, **fields) -> AssetRecord:
    asset_id = req.asset_id or req.image_id or _derive_asset_id(req)
    asset_location = fields.pop("asset_location", None) or req.location
    return AssetRecord(
        asset_id=asset_id,
        asset_name=req.asset_name,
        model_number=req.model_number or req.model_id,
        manufacturer=req.manufacturer or req.manufacturer_name,
        category=req.category,
        document_text=req.document_text
        or fields.get("serial_number_series")
        or fields.get("serial_number_location")
        or fields.get("serial_number_guide"),
        asset_location=asset_location,
        image_id=req.image_id,
        image_path=req.image_path,
        ai_attributes=req.ai_attributes,
        record_set=record_set,
        **fields,
    )


def _derive_asset_id(req: AssetBaseRequest) -> str:
    seed = "|".join(
        part
        for part in [
            req.asset_name,
            req.model_number or req.model_id,
            req.manufacturer or req.manufacturer_name,
            req.category,
            req.ai_attributes,
        ]
        if part
    )
    if not seed:
        seed = "serial-ai-agent"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _record_response(record: AssetRecord) -> dict:
    return {
        "asset_id": record.asset_id,
        "asset_name": record.asset_name,
        "model_number": record.model_number,
        "manufacturer": record.manufacturer,
        "category": record.category,
        "asset_location": record.asset_location,
        "image_id": record.image_id,
        "image_path": record.image_path,
        "ai_attributes": record.ai_attributes,
        "serial_number_location": record.serial_number_location,
        "serial_number_series": record.serial_number_series,
        "serial_number_guide": record.serial_number_guide,
        "record_set": record.record_set,
        "confidence": record.confidence,
    }


def _guide_response(result) -> dict:
    asset = result.asset
    return {
        "found": result.found,
        "needs_image": result.needs_image,
        "image_id": asset.image_id if asset else None,
        "asset_id": asset.asset_id if asset else None,
        "document_id": asset.document_id if asset else None,
        "asset_name": asset.asset_name if asset else None,
        "model_number": asset.model_number if asset else None,
        "manufacturer": asset.manufacturer if asset else None,
        "category": asset.category if asset else None,
        "asset_location": asset.asset_location if asset else None,
        "image_path": asset.image_path if asset else None,
        "ai_attributes": asset.ai_attributes if asset else None,
        "message": result.message,
        "set_name": asset.record_set if asset else None,
        "serial_number_location": asset.serial_number_location if asset else None,
    }


app = create_app() if FastAPI is not None else None
