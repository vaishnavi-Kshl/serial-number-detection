# Serial AI Agent

Best flow for your setup:

1. Your asset pipeline writes the asset context into Qdrant set `a` with fields like `asset_name`, `image_id`, `model_id`, `manufacturer_name`, `image_path`, `ai_attributes`, and `location`.
2. Your training module writes the serial-location doc into Qdrant set `b1` with `serial_number_location` and `location_guide`.
3. Your training module writes the serial-series doc into Qdrant set `b2` with `serial_number_series`.
4. The agent resolves the asset from set `a`, then looks up the matching serial guidance in `b1`.
5. Then the user uploads a serial-number image.
6. The agent OCRs the image, extracts the serial, and matches it against the stored `b2` series while keeping the same asset context.

Why this is best:

- It matches your existing training flow.
- Qdrant becomes the source of truth.
- Full parsed documents can be split across `b1` and `b2` when a single source contains both location and series text.
- The guide step and verification step are separated cleanly.

Run the API:

```bash
uvicorn serial_agent.api:create_app --factory --reload
```

The app automatically loads credentials from `.env` through [serial_agent/config.py](/home/swap/Desktop/serial-ai-agent/serial_agent/config.py).

Main endpoints:

`POST /train/a`
`POST /train/b1`
`POST /train/b2`
`POST /ingest/pdf`
`POST /guide`
`POST /verify/image`

Use `POST /ingest/pdf` to upload a document like `Asset Item Info (1).pdf`; this parser recognizes the `Item Name`, `Item Model`, `Item Manufacturer`, and `Special Instructions` layout and stores one Qdrant point per page.
Send the PDF as the `file` upload and pass `qdrant_collection_name` if you want to write the parsed pages into a specific Qdrant collection.
The response now returns a `document_id`, `page_count`, `page_ids`, and a compact `pages` array instead of one combined document blob.

For `POST /train/a`, send the asset metadata from Set A so the serial flow can reuse the same asset record later.
For `POST /guide`, send model/manufacturer/category or your asset id. For `POST /verify/image`, upload the image file plus the same asset info so the agent can find the right `b2` record.
If you already have the extracted doc text, you can send `document_text` in the training payload and it will be stored in Qdrant too.

By default the app uses Qdrant collections `asset_set_a`, `serial_b1`, and `serial_b2`, but you can override them with `QDRANT_SET_A_COLLECTION`, `QDRANT_B1_COLLECTION`, and `QDRANT_B2_COLLECTION`.
