import re
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app import config

SYSTEM_PROMPT = """You are a support assistant. Answer ONLY using the numbered context passages below.

Rules:
- Every factual sentence must end with the citation(s) of the passage(s) it comes from, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: NOT_FOUND
- Do not use outside knowledge. Do not guess. Be concise.

Context:
{context}"""


@lru_cache
def _store() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX,
        embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
    )


@lru_cache
def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def retrieve(question: str):
    hits = _store().similarity_search_with_score(question, k=config.TOP_K)
    return [(doc, score) for doc, score in hits if score >= config.MIN_SCORE]


def answer(question: str) -> dict:
    hits = retrieve(question)
    if not hits:
        return {"answer": config.NOT_FOUND_MESSAGE, "sources": [], "found": False}

    context = "\n\n".join(
        f"[{i}] ({doc.metadata['source']}, page {int(doc.metadata['page'])})\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, 1)
    )
    reply = _llm().invoke(
        [("system", SYSTEM_PROMPT.format(context=context)), ("human", question)]
    ).content.strip()

    if "NOT_FOUND" in reply:
        return {"answer": config.NOT_FOUND_MESSAGE, "sources": [], "found": False}

    # Only return the sources the model actually cited.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply) if 0 < int(n) <= len(hits)})
    sources = [
        {
            "ref": n,
            "document": hits[n - 1][0].metadata["source"],
            "page": int(hits[n - 1][0].metadata["page"]),
            "score": round(hits[n - 1][1], 3),
            "excerpt": hits[n - 1][0].page_content[:300],
        }
        for n in cited
    ]
    return {"answer": reply, "sources": sources, "found": True}
