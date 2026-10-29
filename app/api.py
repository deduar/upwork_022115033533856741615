from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import rag

app = FastAPI(title="Docs RAG Chatbot")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    ref: int
    document: str
    page: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    found: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return rag.answer(req.question)
