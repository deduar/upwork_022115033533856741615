"""Ingest every PDF in PDF_DIR into Pinecone.

Usage: python -m app.ingest
Re-running is safe: chunk IDs are deterministic, so existing vectors are overwritten.
"""
import hashlib
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from app import config


def ensure_index(pc: Pinecone) -> None:
    if config.PINECONE_INDEX in pc.list_indexes().names():
        return
    print(f"Creating index '{config.PINECONE_INDEX}'...")
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)


def load_chunks(pdf_dir: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        # One Document per page, so every chunk keeps an exact page number.
        pages = PyPDFLoader(str(pdf)).load()
        for page in pages:
            page.page_content = " ".join(page.page_content.split())
            page.metadata = {
                "source": pdf.name,
                "page": page.metadata.get("page", 0) + 1,  # 1-based for humans
            }
        doc_chunks = [c for c in splitter.split_documents(pages) if len(c.page_content) > 50]
        print(f"{pdf.name}: {len(pages)} pages -> {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    return chunks


def chunk_id(chunk) -> str:
    key = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.page_content}"
    return hashlib.sha1(key.encode()).hexdigest()


def main() -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    ensure_index(pc)

    chunks = load_chunks(config.PDF_DIR)
    if not chunks:
        raise SystemExit(f"No PDFs found in {config.PDF_DIR}")

    store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX,
        embedding=PineconeEmbeddings(model=config.EMBEDDING_MODEL),
    )
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    print(f"Upserted {len(chunks)} chunks into '{config.PINECONE_INDEX}'")


if __name__ == "__main__":
    main()
