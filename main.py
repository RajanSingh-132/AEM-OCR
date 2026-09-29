"""
AEM-AI - OCR + Gemini JSON extraction API.

Run:
    uvicorn main:app --reload
    # or
    python main.py

Docs: http://127.0.0.1:8000/docs
"""

import logging

import uvicorn
from fastapi import FastAPI

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
for _name in ("api", "OCRAI"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        _logger.addHandler(_handler)

from api import api_router  # noqa: E402

app = FastAPI(title="AEM-AI", version="1.0.0")
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
