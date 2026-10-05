"""AEM chat assistant endpoint: questions answered from the eManifest records in MongoDB."""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Ai_Assistant.config import CHAT_MAX_QUESTION_CHARS, CHAT_MAX_TOP_K, CHAT_TOP_K
from Ai_Assistant.services.chat_service import answer_question
from Ai_Assistant.services.mongo_client import UnknownCompanyError
from OCRAI.llm_client import LLMError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=CHAT_MAX_QUESTION_CHARS,
                          examples=["Which manifests are still in Draft?"])
    top_k: int = Field(CHAT_TOP_K, ge=1, le=CHAT_MAX_TOP_K, description="How many records to search and use.")
    company_id: int = Field(..., examples=[5], description="Company whose eManifests are searched.")


class ChatSource(BaseModel):
    id: int
    trip_number: str | None
    status: str | None
    port: str | None
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about eManifests; the answer is based only on records found in the database."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    start = time.perf_counter()
    log.info("=" * 70)
    log.info("NEW CHAT QUESTION company %d (%d chars, top_k=%d)", request.company_id, len(question), request.top_k)
    try:
        # Embedding, MongoDB and LLM clients are blocking; keep them off the event loop.
        result = await asyncio.to_thread(answer_question, question, request.top_k, request.company_id)
        log.info("DONE in %.2fs", time.perf_counter() - start)
        return result
    except UnknownCompanyError as exc:
        log.warning("  Rejected: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LLMError as exc:
        log.error("  Answer generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="The assistant could not generate an answer. Please try again.") from exc
    except Exception as exc:
        log.exception("  Chat failed")
        raise HTTPException(status_code=503, detail="Searching the eManifest records failed. Please try again.") from exc
    finally:
        log.info("=" * 70)
