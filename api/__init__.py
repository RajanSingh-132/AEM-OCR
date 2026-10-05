"""All API routers. Add each new feature's router here."""

from fastapi import APIRouter

from api.chat_routes import router as chat_router
from api.upload_routes import router as upload_router

api_router = APIRouter()
api_router.include_router(upload_router, tags=["Document extraction"])
api_router.include_router(chat_router, tags=["AEM chat assistant"])
