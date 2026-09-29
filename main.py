"""
AEM-AI - Logistics document extraction API (PDF / image / Word -> order JSON).

Run:
    uvicorn main:app --reload
    # or
    python main.py

Docs: http://127.0.0.1:8000/docs
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
for _name in ("api", "OCRAI"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        _logger.addHandler(_handler)

from api import api_router  # noqa: E402

app = FastAPI(title="AEM-AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.include_router(api_router)


def custom_openapi() -> dict:
    """Mark file fields as format=binary so Swagger UI shows a file picker on OpenAPI 3.1."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    for component in schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():
            for target in (prop, prop.get("items") or {}):
                if target.get("contentMediaType") == "application/octet-stream":
                    target["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
