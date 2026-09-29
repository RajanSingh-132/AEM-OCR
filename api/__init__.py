"""All API routers. Add each new feature's router here."""

from fastapi import APIRouter

from api.ocr_routes import router as ocr_router

api_router = APIRouter()
api_router.include_router(ocr_router, tags=["OCR AI"])
