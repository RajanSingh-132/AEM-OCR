"""AEM chat assistant endpoint: questions answered from the eManifest records in MongoDB."""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from Ai_Assistant.config import CHAT_MAX_QUESTION_CHARS
from Ai_Assistant.prompts import COMPANY_NOT_READY_MESSAGE
from Ai_Assistant.services.chat_service import answer_question
from Ai_Assistant.services.mongo_client import CompanyDataNotFound
from OCRAI.llm_client import LLMError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=CHAT_MAX_QUESTION_CHARS,
                          examples=["Which manifests are still in Draft?"])
    company_id: int = Field(..., examples=[5], description="Company id = name of the MongoDB database searched.")


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about eManifests; the answer is based only on records found in the database."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    start = time.perf_counter()
    log.info("=" * 70)
    log.info("NEW CHAT QUESTION company %d (%d chars)", request.company_id, len(question))
    try:
        # Embedding, MongoDB and LLM clients are blocking; keep them off the event loop.
        result = await asyncio.to_thread(answer_question, question, request.company_id)
        log.info("DONE in %.2fs", time.perf_counter() - start)
        return result
    except CompanyDataNotFound as exc:
        log.warning("  No data for company %d (%s) -> 'working on the data' answer", request.company_id, exc)
        return {"answer": COMPANY_NOT_READY_MESSAGE.format(company_id=request.company_id)}
    except LLMError as exc:
        log.error("  Answer generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="The assistant could not generate an answer. Please try again.") from exc
    except Exception as exc:
        log.exception("  Chat failed")
        raise HTTPException(status_code=503, detail="Searching the eManifest records failed. Please try again.") from exc
    finally:
        log.info("=" * 70)
