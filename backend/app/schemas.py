from pydantic import BaseModel
from typing import Optional, List


class ChunkBase(BaseModel):
    text: str


class Chunk(ChunkBase):
    id: int
    doc_id: int

    class Config:
        from_attributes = True  # replaces orm_mode in Pydantic v2


class DocumentBase(BaseModel):
    title: str
    content: str


class DocumentCreate(DocumentBase):
    summary: Optional[str] = None
    chunks: Optional[List[ChunkBase]] = None


class Document(DocumentBase):
    id: int
    summary: Optional[str] = None
    chunks: List[Chunk] = []

    class Config:
        from_attributes = True  # replaces orm_mode
