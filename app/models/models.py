from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class Source(str, Enum):
    wikipedia = "wikipedia"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: Optional[float] = None


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class Query(BaseModel):
    query: str
    top_k: Optional[int] = 3


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
    answer: Optional[str] = None
