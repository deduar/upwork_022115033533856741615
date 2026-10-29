import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "docs-rag-demo")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

TOP_K = int(os.getenv("TOP_K", "5"))
# Chunks scoring below this cosine similarity are discarded. If none survive,
# the bot refuses without calling the LLM. Tune with eval/run_eval.py.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))

NOT_FOUND_MESSAGE = (
    "I couldn't find the answer to that in the documentation. "
    "Please contact the support team."
)
