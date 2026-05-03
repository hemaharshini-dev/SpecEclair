# Configuration settings
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "dataset.pdf"
INDEX_PATH = BASE_DIR / "index" / "chunks.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = 5

# Hybrid Search Settings
ALPHA = 0.7  # Weight for semantic search (1-ALPHA for keyword search)
USE_CROSS_ENCODER = True
CROSS_ENCODER_TOP_K = 10 # Number of candidates to re-rank
