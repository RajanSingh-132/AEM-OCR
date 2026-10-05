"""AEM assistant settings, loaded from .env (variable names as they are in .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB = os.getenv("DB", "").strip()
MONGO_COLLECTION = os.getenv("COLLECTION_NAME", "").strip()

# Company routing: company_id -> (database, collection) holding that company's eManifests.
# To add a company later, add its entry here (e.g. from more .env settings).
_company_id = os.getenv("COMPANY_ID", "").strip()
COMPANY_ROUTES: dict[int, tuple[str, str]] = {int(_company_id): (MONGO_DB, MONGO_COLLECTION)} if _company_id else {}

# AWS Bedrock embeddings (Amazon Titan Text Embeddings V2)
BEDROCK_EMBEDDING_MODEL = os.getenv("bedrockmodel", "amazon.titan-embed-text-v2:0").strip()
AWS_ACCESS_KEY = os.getenv("accesskey", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("secretaccesskey", "").strip()
AWS_REGION = os.getenv("awsregion", "us-east-1").strip()

# Titan V2 supports 256, 512 or 1024 dimensions.
EMBEDDING_DIMENSIONS = 1024
# Parallel embedding calls while storing (Bedrock rate limits apply).
EMBEDDING_WORKERS = 8

# Atlas Vector Search index on the "embedding" field.
VECTOR_INDEX_NAME = "emanifest_vector_index"
# Fields the vector search can pre-filter on (e.g. only one company's manifests).
# The original record is stored under "data".
VECTOR_FILTER_FIELDS = ("data.CompanyId", "data.USPort", "data.Status", "data.ArrivalDate", "data.TripNum", "data.Scac")

EMANIFEST_FILE = Path(__file__).resolve().parent / "eManifest.txt"

# Chat (POST /api/v1/chat)
CHAT_TOP_K = 10               # records retrieved per question and given to the LLM
CHAT_MAX_TOP_K = 50           # upper limit a caller may request
CHAT_NUM_CANDIDATES = 200     # vector search candidates (higher = better recall, slower)
CHAT_MAX_QUESTION_CHARS = 2000
CHAT_MAX_ANSWER_TOKENS = 4096


def require(*names_and_values: tuple[str, str]) -> None:
    """Fail fast with a clear message when a required .env setting is missing."""
    missing = [name for name, value in names_and_values if not value]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
