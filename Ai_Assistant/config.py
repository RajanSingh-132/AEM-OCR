"""AEM assistant settings, loaded from .env (variable names as they are in .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# MongoDB. Each company has its own database, named after its company id
# (company 10 -> database "10"); every company database uses this collection name.
MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_COLLECTION = os.getenv("COLLECTION_NAME", "").strip() or "eManifest"


def company_db_name(company_id: int) -> str:
    """MongoDB database name for a company: its id as text."""
    return str(company_id)

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
# How many records reach the answer LLM depends on the question (see services/query_planner.py):
LIST_DEFAULT_LIMIT = 10       # list questions that name no number ("which manifests are in Draft?")
LATEST_DEFAULT_LIMIT = 5      # "latest / most recent ..." with no number
QUERY_HARD_LIMIT = 25         # absolute maximum of records sent to the LLM for any question
GROUP_MAX_ROWS = 20           # "count per <field>": largest groups shown
VECTOR_MIN_SCORE = 0.58       # similarity search: drop records less similar than this
VECTOR_SCORE_WINDOW = 0.06    # similarity search: keep only records this close to the best match
VECTOR_MAX_RECORDS = 10       # similarity search: at most this many records
CHAT_NUM_CANDIDATES = 200     # vector search candidates (higher = better recall, slower)
# Internal fields removed from every record before it reaches the answer LLM, so they can never
# appear in a response (compared case-insensitively). They can still be used for lookups.
HIDDEN_RECORD_FIELDS = (
    "Id", "CompanyId", "Version", "ReferenceID", "AmendmentCode", "AmendmentTypeCode", "amendcode",
    "AceID", "AciID", "TruckId",
)
ENUM_VALUES_CACHE_SECONDS = 600  # allowed values of "enum" fields are read from the data, cached this long
CHAT_MAX_QUESTION_CHARS = 2000
CHAT_MAX_ANSWER_TOKENS = 4096

# Chat memory: one document per conversation (session), in a central database so that
# unknown company ids never create databases.
CHAT_MEMORY_DB = os.getenv("CHAT_MEMORY_DB", "").strip() or "AEM-AI"
CHAT_MEMORY_COLLECTION = "chat_sessions"
CHAT_HISTORY_TURNS = 6        # earlier question/answer pairs used for follow-up questions
CHAT_MAX_STORED_TURNS = 50    # older turns of a session are dropped
CHAT_HISTORY_ANSWER_CHARS = 1500  # long answers are shortened in the history sent to the answer LLM
PLANNER_HISTORY_ANSWER_CHARS = 400  # ... and shortened more for the planner (it only needs the references)
CHAT_SESSION_TTL_DAYS = 7     # sessions unused this long are deleted automatically


def require(*names_and_values: tuple[str, str]) -> None:
    """Fail fast with a clear message when a required .env setting is missing."""
    missing = [name for name, value in names_and_values if not value]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
