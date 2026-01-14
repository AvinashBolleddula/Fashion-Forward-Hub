"""Input validation using Pydantic."""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Literal, Optional


class RegisterRequest(BaseModel):
    """Registration request validation."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class LoginRequest(BaseModel):
    """Login request validation."""
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    """Chat request validation with advanced retrieval options."""
    message: str = Field(..., min_length=1, max_length=2000)
    
    # RAG control
    use_rag: bool = True
    
    # Retrieval options
    retriever_type: Literal["bm25", "semantic", "hybrid"] = "semantic"
    simplified: bool = False
    top_k: int = Field(default=20, ge=1, le=100)
    
    # Hybrid retrieval parameters
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)  # RRF alpha
    k: Optional[int] = Field(default=60, ge=1, le=100)  # RRF k constant
    
    # Reranking options
    use_reranker: bool = False
    rerank_query: Optional[str] = None
    
    @validator('message')
    def sanitize_message(cls, v):
        """Basic sanitization."""
        return v.strip()
    
    @validator('alpha')
    def validate_alpha(cls, v, values):
        """Alpha only matters for hybrid retrieval."""
        if values.get('retriever_type') == 'hybrid' and v is None:
            return 0.5  # Default
        return v
    
    @validator('k')
    def validate_k(cls, v, values):
        """K only matters for hybrid retrieval."""
        if values.get('retriever_type') == 'hybrid' and v is None:
            return 60  # Default
        return v