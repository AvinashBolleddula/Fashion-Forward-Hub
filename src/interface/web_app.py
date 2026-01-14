"""Production-grade Flask web application with authentication."""

# 1) Imports + setup
# Imports Flask utilities (Flask, request, jsonify, render_template)
# Enables: 	CORS (so frontend can call backend APIs), JWT auth (flask_jwt_extended), Rate limiting (flask_limiter with Redis)
# Pydantic schemas to validate request JSON (RegisterRequest, LoginRequest, ChatRequest)
# 
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, get_jwt
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import uuid
import logging
from datetime import datetime, timedelta
from pydantic import ValidationError


# Adds your project src/ to Python path so it can import:
# ChatBot, config, AuthService, schemas
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from chatbot import ChatBot
from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    JWT_SECRET_KEY, JWT_ACCESS_TOKEN_EXPIRES,
    REDIS_URL, RATE_LIMIT_DEFAULT, RATE_LIMIT_CHAT
)
from auth import AuthService
from schemas import RegisterRequest, LoginRequest, ChatRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
# Creates the Flask app and points it to your templates/static directories.
app = Flask(__name__, 
            template_folder=str(BASE_DIR / 'templates'),
            static_folder=str(BASE_DIR / 'static'))

# Configures JWT signing + token expiry
# # Sets Flask secret key.
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=JWT_ACCESS_TOKEN_EXPIRES)
app.secret_key = JWT_SECRET_KEY

# Enables CORS + initializes JWT manager.
CORS(app)
jwt = JWTManager(app)

# Rate limiting(uses Redis)
# Applies a default rate limit to all endpoints
# Stores rate limit counters in Redis (so it works across server restarts / multiple instances)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=[RATE_LIMIT_DEFAULT]
)

# Store active chatbot sessions
# A Python dictionary that holds ChatBot objects per user.
chatbot_sessions = {}



def get_or_create_chatbot(user_id: str) -> ChatBot:
    """Get existing chatbot session or create new one."""
    if user_id not in chatbot_sessions:
        chatbot_sessions[user_id] = ChatBot()  # â† Remove simplified parameter
        logger.info(f"Created new chatbot session: {user_id}")
    return chatbot_sessions[user_id]

# serves index.html (probably login/register page)
@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

# serves chat.html (chat UI page)
# /chat is not protected by JWT here. 
# The real protection is done in chat.js by redirecting if token missing
@app.route('/chat')
def chat_page():
    """Render chat interface (requires authentication)."""
    return render_template('chat.html')

# Rate limited: 5/hour
# Validates JSON using RegisterRequest
# Calls AuthService.create_user(email, password)
# Returns success JSON or validation errors
@app.route('/api/register', methods=['POST'])
@limiter.limit("5/hour")
def register():
    """Register a new user."""
    try:
        data = RegisterRequest(**request.get_json())
        
        user = AuthService.create_user(data.email, data.password)
        
        logger.info(f"New user registered: {user.email}")
        
        return jsonify({
            'message': 'User created successfully',
            'user_id': user.id
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# Rate limited: 10/hour
# Validates JSON using LoginRequest
# Calls AuthService.authenticate(email, password)
# If valid, creates JWT token with create_access_token(identity=user.id)
# Returns token and user info or errors
# This is what your frontend stores in localStorage as access_token and user_email.
@app.route('/api/login', methods=['POST'])
@limiter.limit("10/hour")
def login():
    """Login and get JWT token."""
    try:
        data = LoginRequest(**request.get_json())
        
        user = AuthService.authenticate(data.email, data.password)
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        access_token = create_access_token(identity=user.id)
        
        logger.info(f"User logged in: {user.email}")
        
        return jsonify({
            'access_token': access_token,
            'user_id': user.id,
            'email': user.email
        }), 200
        
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


# Chat API (this is what Send calls)
# @jwt_required() â†’ must include valid Bearer token
# @limiter.limit(RATE_LIMIT_CHAT) â†’ rate limited chat endpoint
@app.route('/api/chat', methods=['POST'])
@jwt_required()
@limiter.limit(RATE_LIMIT_CHAT)
def chat():
    """Handle chat messages with advanced retrieval options."""
    try:
        # Gets user id from JWT token.
        # Validates and parses request payload from frontend.
        user_id = get_jwt_identity()
        data = ChatRequest(**request.get_json())
        
        # Get chatbot (use simplified from data, not session-based)
        # Gets the userâ€™s chatbot session (normal or simplified).
        chatbot = get_or_create_chatbot(user_id)
        
        # Get response with all new parameters
        # This is where your RAG / retriever / reranker / hybrid alpha+k settings actually get used.
        # Logs key info (RAG on/off, retriever type, tokens)
        # Calls chatbot.chat() with all the new parameters from ChatRequest.
        # Returns response content + token usage + config back to frontend.
        response = chatbot.chat(
            prompt=data.message,
            return_stats=True,
            use_rag=data.use_rag,
            retriever_type=data.retriever_type,
            top_k=data.top_k,
            alpha=data.alpha,
            k=data.k,
            use_reranker=data.use_reranker,
            rerank_query=data.rerank_query,
            simplified=data.simplified
        )
        
        # Logs key info (RAG on/off, retriever type, tokens)
        logger.info(
            f"User {user_id}: Q: {data.message[:50]}... | "
            f"Retriever: {data.retriever_type} | "
            f"RAG: {data.use_rag} | "
            f"Reranker: {data.use_reranker} | "
            f"Tokens: {response['total_tokens']}"
        )
        
        # Returns JSON to frontend:
        # This JSON is exactly what chat.js reads as data.response, data.tokens, etc.
        return jsonify({
            'response': response['content'],
            'tokens': {
                'rag': response['rag_tokens'],
                'llm': response['llm_tokens'],
                'total': response['total_tokens']
            },
            'retrieval_config': {
                'use_rag': data.use_rag,
                'retriever_type': data.retriever_type,
                'top_k': data.top_k,
                'use_reranker': data.use_reranker
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except ValidationError as e:
        return jsonify({'error': e.errors()}), 400
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
@jwt_required()
def clear_conversation():
    """Clear conversation history."""
    try:
        user_id = get_jwt_identity()
        
        # Clear both simplified and standard sessions
        for simplified in [True, False]:
            session_key = f"{user_id}:{simplified}"
            if session_key in chatbot_sessions:
                chatbot_sessions[session_key].clear_conversation()
                logger.info(f"Cleared session: {session_key}")
        
        return jsonify({'message': 'Conversation cleared'})
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(chatbot_sessions),
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    logger.info(f"Starting Flask app on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)