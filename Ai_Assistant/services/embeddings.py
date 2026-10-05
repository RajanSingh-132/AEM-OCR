"""Text -> 1024-d vector with Amazon Titan Text Embeddings V2 on AWS Bedrock."""

import json
import logging
from functools import lru_cache

import boto3

from Ai_Assistant.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    BEDROCK_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    require,
)

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _bedrock():
    require(("accesskey", AWS_ACCESS_KEY), ("secretaccesskey", AWS_SECRET_ACCESS_KEY))
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def embed_text(text: str) -> list[float]:
    """Embed one text. Vectors are normalized, so cosine similarity works directly."""
    response = _bedrock().invoke_model(
        modelId=BEDROCK_EMBEDDING_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text, "dimensions": EMBEDDING_DIMENSIONS, "normalize": True}),
    )
    embedding = json.loads(response["body"].read())["embedding"]
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(embedding)}")
    return embedding
