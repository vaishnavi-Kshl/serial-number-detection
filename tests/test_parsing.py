import inspect

import pytest

from serial_agent.agent import SerialNumberAgent, parse_question
from serial_agent.api import _guide_response, create_app
from serial_agent.models import AssetRecord
from serial_agent.ocr import extract_serial_from_text
from serial_agent.parsing import parse_text
from serial_agent.repository import InMemorySerialRepository, QdrantSerialRepository
from serial_agent.verification import verify_serial_number
from types import SimpleNamespace


def test_parse_question_extracts_labels():
    query = parse_question("Model ID PH96E3 from Randell in Refrigerated Prep Table")
    assert query.model_number == "PH96E3"
    assert query.manufacturer == "Randell"
    assert query.category == "Refrigerated Prep Table"


def test_split_repository_guidance_and_verification():
    repo = InMemorySerialRepository()
    agent = SerialNumberAgent(repository=repo)

    agent.train_b1(
        AssetRecord(
            asset_id="asset-1",
            asset_name="Pizza Prep Table",
            model_number="PH96E3",
            manufacturer="Randell",
            category="Refrigerated Prep Table",
            serial_number_location="Please open Settings > About",
            serial_number_guide="Please open Settings > About",
            record_set="b1",
        )
    )
    agent.train_b2(
        AssetRecord(
            asset_id="asset-1",
            asset_name="Pizza Prep Table",
            model_number="PH96E3",
            manufacturer="Randell",
            category="Refrigerated Prep Table",
            serial_number_series="SN1000-SN1999",
            record_set="b2",
        )
    )

    guide = agent.guide("model PH96E3 Randell Refrigerated Prep Table")
    assert guide.found is True
    assert "Settings > About" in guide.message
    assert guide.needs_image is True

    repo_for_verify = InMemorySerialRepository(
        b1_records=repo.b1_records,
        b2_records=repo.b2_records,
    )
    verify_agent = SerialNumberAgent(repository=repo_for_verify)
    result = verify_agent.verify_image(
        image_path="not-used.jpg",
        question="model PH96E3 Randell Refrigerated Prep Table",
    )
    # This path should fail gracefully because the file does not exist,
    # but the agent should still resolve the correct asset before OCR.
    assert result.asset is not None
    assert result.asset.serial_number_series == "SN1000-SN1999"


def test_repository_upsert_splits_full_document_into_b1_and_b2():
    repo = InMemorySerialRepository()
    repo.upsert(
        AssetRecord(
            asset_id="asset-2",
            asset_name="Combo Asset",
            model_number="ZX-22",
            manufacturer="Acme",
            serial_number_series="SN2000-SN2999",
            serial_number_location="Inside the rear panel",
            serial_number_guide="Open the rear panel and look beside the power inlet",
        )
    )

    assert repo.get_b1("asset-2") is not None
    assert repo.get_b1("asset-2").serial_number_location == "Inside the rear panel"
    assert repo.get_b1("asset-2").serial_number_series is None

    assert repo.get_b2("asset-2") is not None
    assert repo.get_b2("asset-2").serial_number_series == "SN2000-SN2999"
    assert repo.get_b2("asset-2").serial_number_location is None


def test_ingest_pdf_upserts_each_page_separately(monkeypatch):
    repo = InMemorySerialRepository()
    agent = SerialNumberAgent(repository=repo)

    fake_records = [
        AssetRecord(
            asset_id="page-1",
            document_id="doc-1",
            asset_name="Water Softener",
            model_number="01020370",
            manufacturer="Culligan",
            serial_number_location="Located on the back of the unit.",
            source_page=1,
        ),
        AssetRecord(
            asset_id="page-2",
            document_id="doc-1",
            asset_name="POS System",
            model_number="027780",
            manufacturer="Elo",
            serial_number_location="Located on the back of the unit.",
            source_page=2,
        ),
    ]

    monkeypatch.setattr("serial_agent.agent.parse_pdf_pages", lambda *args, **kwargs: fake_records)

    records = agent.ingest_pdf("ignored.pdf", source_name="Asset Item Info (1).pdf")
    assert len(records) == 2
    assert repo.get_b1("page-1") is not None
    assert repo.get_b1("page-2") is not None
    assert repo.get_b1("page-1").document_id == "doc-1"
    assert repo.get_b1("page-2").document_id == "doc-1"


def test_ingest_pdf_can_target_custom_collection(monkeypatch):
    repo = InMemorySerialRepository()
    agent = SerialNumberAgent(repository=repo)

    fake_records = [
        AssetRecord(
            asset_id="page-1",
            document_id="doc-2",
            asset_name="Water Softener",
            model_number="01020370",
            manufacturer="Culligan",
            serial_number_location="Located on the back of the unit.",
            source_page=1,
        ),
    ]

    monkeypatch.setattr("serial_agent.agent.parse_pdf_pages", lambda *args, **kwargs: fake_records)

    records = agent.ingest_pdf(
        "ignored.pdf",
        collection_name="custom_qdrant_collection",
        source_name="Asset Item Info (1).pdf",
    )
    assert len(records) == 1
    assert repo.get_b2("page-1") is None
    assert repo.collection_records["custom_qdrant_collection"]["page-1"].document_id == "doc-2"


def test_ingest_pdf_route_uses_qdrant_collection_name_field():
    try:
        app = create_app()
    except RuntimeError as exc:
        if "fastapi is not installed" in str(exc):
            pytest.skip("fastapi is not installed in this environment")
        raise
    route = next(route for route in app.routes if getattr(route, "path", None) == "/ingest/pdf")
    param_names = list(inspect.signature(route.endpoint).parameters)

    assert "file" in param_names
    assert "qdrant_collection_name" in param_names
    assert "asset_id" not in param_names


def test_guide_route_has_explicit_response_and_error_schema():
    try:
        app = create_app()
    except RuntimeError as exc:
        if "fastapi is not installed" in str(exc):
            pytest.skip("fastapi is not installed in this environment")
        raise
    route = next(route for route in app.routes if getattr(route, "path", None) == "/guide")

    assert getattr(route, "response_model", None).__name__ == "GuideResponse"
    assert 422 in getattr(route, "responses", {})
    assert 503 in getattr(route, "responses", {})
    assert route.responses[503]["model"].__name__ == "GuideErrorResponse"


def test_qdrant_lookup_skips_invalid_point_ids():
    repo = QdrantSerialRepository(ensure_collection=False)
    called = []

    def retrieve(**kwargs):
        called.append(kwargs)
        raise AssertionError("retrieve should not be called for non-UUID ids")

    repo._client = SimpleNamespace(retrieve=retrieve)

    assert repo.get_set_a("asset-linked") is None
    assert called == []


def test_qdrant_missing_collection_is_treated_as_empty():
    repo = QdrantSerialRepository(ensure_collection=False)
    called = []

    def collection_exists(collection_name):
        called.append(("exists", collection_name))
        return False

    def retrieve(**kwargs):
        called.append(("retrieve", kwargs))
        raise AssertionError("retrieve should not be called when collection is missing")

    def scroll(**kwargs):
        called.append(("scroll", kwargs))
        raise AssertionError("scroll should not be called when collection is missing")

    repo._client = SimpleNamespace(
        collection_exists=collection_exists,
        retrieve=retrieve,
        scroll=scroll,
    )

    assert repo.get_b2("4fd1d2cf-2f0a-4a3d-8e3f-6f5ed8d9f4b8") is None
    assert repo.find_b2_by_document_id("doc-1") == []
    assert repo.find_b2_candidates(SimpleNamespace()) == []
    assert all(entry[0] == "exists" for entry in called)


def test_parse_text_extracts_asset_item_sheet_fields():
    text = """
Asset Item Information Sheet
Item Name: Water Softener

Item Model: 01020370

Item Manufacturer: Culligan

Special Instructions: A stick mirror or flexible camera will be need to capture serial
number which is located on the back of the unit.

Overall Photo
QR Code Placement

Serial Number Location
""".strip()

    record = parse_text(text)
    assert record.asset_name == "Water Softener"
    assert record.model_number == "01020370"
    assert record.manufacturer == "Culligan"
    assert "located on the back of the unit" in (record.serial_number_location or "")
    assert record.serial_number_series is None


def test_from_payload_ignores_geo_location_objects():
    record = AssetRecord.from_payload(
        {
            "asset_id": "asset-geo",
            "asset_name": "Deep Fryer",
            "model_number": "SSH55",
            "manufacturer": "Pitco",
            "location": {"lon": -113.0720046, "lat": 37.6805448},
        }
    )

    assert record.serial_number_location is None
    assert record.extra["location"] == {"lon": -113.0720046, "lat": 37.6805448}


def test_set_a_payload_treats_location_as_asset_location():
    record = AssetRecord.from_payload(
        {
            "asset_id": "asset-a",
            "set_name": "a",
            "asset_name": "Water Softener",
            "model_id": "01020370",
            "manufacturer_name": "Culligan",
            "location": "Kitchen",
            "image_id": "img-a",
            "image_path": "/images/asset.jpg",
        }
    )

    assert record.asset_location == "Kitchen"
    assert record.serial_number_location is None
    assert record.to_payload()["location"] == "Kitchen"


def test_guide_prefers_text_location_over_geo_point():
    repo = InMemorySerialRepository()
    repo.upsert_b1(
        AssetRecord(
            asset_id="geo-point",
            asset_name="Heated Holding Cabinet",
            model_number="CS82-CH8",
            manufacturer="Bevles",
            category="Holding Cabinet",
            extra={"location": {"lon": -84.1886773, "lat": 39.7836875}},
        )
    )
    repo.upsert_b1(
        AssetRecord(
            asset_id="text-point",
            asset_name="Heated Holding Cabinet",
            model_number="CS82-CH8",
            manufacturer="Bevles",
            category="Holding Cabinet",
            serial_number_location="Located on the back side near the lower right panel.",
        )
    )

    agent = SerialNumberAgent(repository=repo)
    result = agent.guide("model CS82-CH8 Bevles Holding Cabinet")

    assert result.found is True
    assert "Located on the back side" in result.message
    assert result.asset is not None
    assert result.asset.serial_number_location == "Located on the back side near the lower right panel."


def test_verify_image_returns_extracted_serial_even_without_b2(monkeypatch):
    repo = InMemorySerialRepository()
    repo.upsert_b1(
        AssetRecord(
            asset_id="asset-verify",
            asset_name="Heated Holding Cabinet",
            model_number="CS82-CH8",
            manufacturer="Bevles",
            category="Holding Cabinet",
            serial_number_location="Located on the top side of the cabinet.",
            serial_number_guide="Located on the top side of the cabinet.",
        )
    )
    agent = SerialNumberAgent(repository=repo)

    monkeypatch.setattr("serial_agent.agent.extract_serial_from_image", lambda _: "SN-12345")

    result = agent.verify_image(
        image_path="ignored.jpg",
        question="model CS82-CH8 Bevles Holding Cabinet",
    )

    assert result.found is False
    assert result.verified is False
    assert result.serial_extracted == "SN-12345"
    assert "Extracted serial number: SN-12345" in result.message
    assert "Add the serial number series first" in result.message


def test_verify_image_links_set_a_asset_context(monkeypatch):
    repo = InMemorySerialRepository()
    agent = SerialNumberAgent(repository=repo)

    agent.train_a(
        AssetRecord(
            asset_id="asset-linked",
            asset_name="Water Softener",
            model_number="01020370",
            manufacturer="Culligan",
            category="Softener",
            asset_location="Kitchen",
            image_id="img-linked",
            image_path="/images/asset.jpg",
            ai_attributes="stainless steel",
            record_set="a",
        )
    )
    agent.train_b1(
        AssetRecord(
            asset_id="asset-linked",
            asset_name="Water Softener",
            model_number="01020370",
            manufacturer="Culligan",
            category="Softener",
            serial_number_location="Kitchen",
            record_set="b1",
        )
    )
    agent.train_b2(
        AssetRecord(
            asset_id="asset-linked",
            asset_name="Water Softener",
            model_number="01020370",
            manufacturer="Culligan",
            category="Softener",
            serial_number_series="SN1000-SN1999",
            record_set="b2",
        )
    )

    monkeypatch.setattr("serial_agent.agent.extract_serial_from_image", lambda _: "SN1500")

    result = agent.verify_image(
        image_path="ignored.jpg",
        question="asset id asset-linked",
    )

    assert result.found is True
    assert result.verified is True
    assert result.asset is not None
    assert result.asset.asset_location == "Kitchen"
    assert result.asset.image_id == "img-linked"
    assert result.asset.serial_number_location == "Kitchen"
    assert result.asset.serial_number_series == "SN1000-SN1999"


def test_verify_image_returns_short_failure_when_serial_missing(monkeypatch):
    repo = InMemorySerialRepository()
    repo.upsert_b1(
        AssetRecord(
            asset_id="asset-verify",
            asset_name="Heated Holding Cabinet",
            model_number="CS82-CH8",
            manufacturer="Bevles",
            category="Holding Cabinet",
            serial_number_location="Located on the top side of the cabinet.",
            serial_number_guide="Located on the top side of the cabinet.",
        )
    )
    agent = SerialNumberAgent(repository=repo)

    monkeypatch.setattr("serial_agent.agent.extract_serial_from_image", lambda _: None)

    result = agent.verify_image(
        image_path="ignored.jpg",
        question="model CS82-CH8 Bevles Holding Cabinet",
    )

    assert result.found is False
    assert result.verified is False
    assert result.serial_extracted is None
    assert result.asset is None
    assert result.message == "No matching asset found in Qdrant set b2. Add the serial number series first."


def test_extract_serial_from_text():
    text = "Serial No: SN1500"
    assert extract_serial_from_text(text) == "SN1500"


def test_extract_serial_from_image_prefers_openai(monkeypatch):
    monkeypatch.setattr("serial_agent.ocr._extract_serial_with_openai", lambda _: "SN-12345")

    from serial_agent.ocr import extract_serial_from_image

    assert extract_serial_from_image("ignored.jpg") == "SN-12345"


def test_extract_serial_from_image_falls_back_to_local_ocr(monkeypatch):
    monkeypatch.setattr("serial_agent.ocr._extract_serial_with_openai", lambda _: None)
    monkeypatch.setattr("serial_agent.ocr.extract_text_from_image", lambda _: "Serial No: SN1500")

    from serial_agent.ocr import extract_serial_from_image

    assert extract_serial_from_image("ignored.jpg") == "SN1500"


def test_verify_serial_range():
    result = verify_serial_number("SN1500", "SN1000-SN1999")
    assert result.matched is True


def test_guide_response_is_flat():
    asset = AssetRecord(
        asset_id="asset-1",
        document_id="doc-1",
        image_id="img-1",
        asset_name="Heated Holding Cabinet",
        model_number="CS82-CH8",
        manufacturer="Bevles",
        category="Holding Cabinet",
        serial_number_location="Located on the back side.",
        record_set="b1",
    )
    result = SimpleNamespace(found=True, needs_image=True, message="ok", asset=asset)

    response = _guide_response(result)

    assert set(response.keys()) == {
        "found",
        "needs_image",
        "image_id",
        "asset_id",
        "document_id",
        "asset_name",
        "model_number",
        "manufacturer",
        "category",
        "asset_location",
        "image_path",
        "ai_attributes",
        "message",
        "set_name",
        "serial_number_location",
    }
    assert response["image_id"] == "img-1"
    assert response["serial_number_location"] == "Located on the back side."
