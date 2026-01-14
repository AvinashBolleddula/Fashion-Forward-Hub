"""Data models for authentication and user management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from werkzeug.security import generate_password_hash, check_password_hash


@dataclass
class User:
    """User model for authentication."""
    id: str
    email: str
    password_hash: str
    created_at: datetime
    is_active: bool = True
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storing."""
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    def verify_password(self, password: str) -> bool:
        """Check if provided password matches hash."""
        return check_password_hash(self.password_hash, password)