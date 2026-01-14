"""Authentication service with JWT and user management."""

import uuid
from datetime import datetime
from typing import Optional, Dict
import redis
import json

from models import User
from config import REDIS_URL

# Redis client for user storage (in production, use PostgreSQL)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class AuthService:
    """Handle user authentication and management."""
    
    USER_PREFIX = "user:"
    EMAIL_INDEX = "email:"
    
    @staticmethod
    def create_user(email: str, password: str) -> User:
        """Create a new user account."""
        # Check if user exists
        if AuthService.get_user_by_email(email):
            raise ValueError("User already exists")
        
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            email=email,
            password_hash=User.hash_password(password),
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        # Store in Redis (in production, use PostgreSQL)
        user_key = f"{AuthService.USER_PREFIX}{user_id}"
        email_key = f"{AuthService.EMAIL_INDEX}{email}"
        
        user_data = {
            'id': user.id,
            'email': user.email,
            'password_hash': user.password_hash,
            'created_at': user.created_at.isoformat(),
            'is_active': user.is_active
        }
        
        redis_client.set(user_key, json.dumps(user_data))
        redis_client.set(email_key, user_id)
        
        return user
    
    @staticmethod
    def get_user_by_email(email: str) -> Optional[User]:
        """Get user by email."""
        email_key = f"{AuthService.EMAIL_INDEX}{email}"
        user_id = redis_client.get(email_key)
        
        if not user_id:
            return None
        
        return AuthService.get_user_by_id(user_id)
    
    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by ID."""
        user_key = f"{AuthService.USER_PREFIX}{user_id}"
        user_data = redis_client.get(user_key)
        
        if not user_data:
            return None
        
        data = json.loads(user_data)
        return User(
            id=data['id'],
            email=data['email'],
            password_hash=data['password_hash'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data['is_active']
        )
    
    @staticmethod
    def authenticate(email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = AuthService.get_user_by_email(email)
        
        if not user or not user.is_active:
            return None
        
        if not user.verify_password(password):
            return None
        
        return user