"""Centralized configuration for the ChatBot application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== OpenAI Configuration ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ==================== Weaviate Configuration ====================
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# ==================== Phoenix Configuration ====================
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

# ==================== Data Paths ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
PRODUCTS_FILE = DATA_DIR / "clothes_json.joblib"
FAQ_FILE = DATA_DIR / "faq.joblib"
CLASSIFICATION_FILE = DATA_DIR / "faq_or_products.csv"

# ==================== Application Settings ====================
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "20"))
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
SIMPLIFIED_MODE = os.getenv("SIMPLIFIED_MODE", "false").lower() == "true"

# ==================== Flask Configuration ====================
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ==================== Validation ====================
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please set it.")

# ==================== Security Configuration ====================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Rate Limiting
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/hour")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "20/minute")

# Validation
if not JWT_SECRET_KEY:
    import secrets
    JWT_SECRET_KEY = secrets.token_hex(32)
    print("⚠️  WARNING: JWT_SECRET_KEY not set. Using random key (not for production!)")
    
# ==================== Export commonly used groups ====================
__all__ = [
    'OPENAI_API_KEY', 'OPENAI_MODEL',
    'WEAVIATE_URL', 'WEAVIATE_GRPC_PORT',
    'PHOENIX_COLLECTOR_ENDPOINT',
    'DATA_DIR', 'PRODUCTS_FILE', 'FAQ_FILE',
    'CONTEXT_WINDOW', 'DEFAULT_TEMPERATURE', 'DEFAULT_MAX_TOKENS'
]