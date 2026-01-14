"""LLM client and generation functions using OpenAI API."""

import os
from openai import OpenAI
from typing import List, Dict, Optional, Any
from config import OPENAI_API_KEY, OPENAI_MODEL


DEFAULT_MODEL = OPENAI_MODEL
# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_with_single_input(
    prompt: str,
    role: str = 'user',
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_tokens: int = 500,
    model: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a response from the LLM with a single input prompt.
    
    Args:
        prompt: The text prompt to send to the model
        role: The role of the message sender (default: 'user')
        top_p: Controls diversity via nucleus sampling (default: 1.0)
        temperature: Controls randomness (0=deterministic, 2=very random) (default: 1.0)
        max_tokens: Maximum number of tokens to generate (default: 500)
        model: Model to use (default: from env or gpt-4o-mini)
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        dict: Response from OpenAI API containing:
            - id: Request ID
            - choices: List of response choices
            - usage: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
            - model: Model used
    
    Example:
        >>> response = generate_with_single_input("What is RAG?", temperature=0.7)
        >>> print(response['choices'][0]['message']['content'])
    """
    if model is None:
        model = DEFAULT_MODEL
    
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    try:
        response = client.chat.completions.create(**payload)
        # Convert to dict for easier access
        json_dict = response.model_dump()
        
        # Add convenience fields for backward compatibility
        json_dict['content'] = json_dict['choices'][0]['message']['content']
        json_dict['total_tokens'] = json_dict['usage']['total_tokens']
        
        return json_dict
    except Exception as e:
        raise Exception(f"Failed to get correct output from LLM call. Error: {e}")


def generate_with_multiple_input(
    messages: List[Dict],
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_tokens: int = 500,
    model: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a response from the LLM with multiple messages (conversation history).
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  Example: [
                      {'role': 'user', 'content': 'Hello'},
                      {'role': 'assistant', 'content': 'Hi there!'},
                      {'role': 'user', 'content': 'How are you?'}
                  ]
        top_p: Controls diversity via nucleus sampling (default: 1.0)
        temperature: Controls randomness (default: 1.0)
        max_tokens: Maximum number of tokens to generate (default: 500)
        model: Model to use (default: from env or gpt-4o-mini)
        **kwargs: Additional parameters to pass to the API
    
    Returns:
        dict: Response from OpenAI API (same format as generate_with_single_input)
    
    Example:
        >>> messages = [
        ...     {'role': 'user', 'content': 'What is RAG?'},
        ...     {'role': 'assistant', 'content': 'RAG is Retrieval-Augmented Generation'},
        ...     {'role': 'user', 'content': 'Give me an example'}
        ... ]
        >>> response = generate_with_multiple_input(messages)
    """
    if model is None:
        model = DEFAULT_MODEL
    
    payload = {
        "model": model,
        "messages": messages,
        "top_p": top_p,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    try:
        response = client.chat.completions.create(**payload)
        json_dict = response.model_dump()
        
        # Add convenience fields
        json_dict['content'] = json_dict['choices'][0]['message']['content']
        json_dict['total_tokens'] = json_dict['usage']['total_tokens']
        
        return json_dict
    except Exception as e:
        raise Exception(f"Failed to get correct output from LLM call. Error: {e}")


def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role: str = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = None
) -> Dict[str, Any]:
    """
    Generate a parameters dictionary for LLM calls.
    
    This is useful for preparing parameters that will be passed to generation functions later.
    
    Args:
        prompt: The text prompt
        temperature: Controls randomness
        role: Message role (default: 'user')
        top_p: Controls diversity
        max_tokens: Maximum tokens to generate
        model: Model to use
    
    Returns:
        dict: Dictionary of parameters ready to pass to generate_with_single_input
    
    Example:
        >>> params = generate_params_dict("Hello", temperature=0.7)
        >>> response = generate_with_single_input(**params)
    """
    if model is None:
        model = DEFAULT_MODEL
    
    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model
    }
    
    return kwargs


def chat_with_context(
    prompt: str,
    context: List[Dict],
    role: str = 'user',
    **kwargs
) -> Dict[str, Any]:
    """
    Continue a conversation by adding a new message to the context.
    
    Args:
        prompt: New user message
        context: Conversation history (list of message dicts)
        role: Role of the new message (default: 'user')
        **kwargs: Additional parameters for the LLM call
    
    Returns:
        dict: LLM response
    
    Example:
        >>> context = []
        >>> response = chat_with_context("What is RAG?", context)
        >>> context.append({'role': 'user', 'content': 'What is RAG?'})
        >>> context.append({'role': 'assistant', 'content': response['content']})
        >>> response = chat_with_context("Give an example", context)
    """
    # Add new message to context
    context.append({'role': role, 'content': prompt})
    
    # Generate response
    response = generate_with_multiple_input(context, **kwargs)
    
    # Add assistant response to context
    context.append({
        'role': 'assistant',
        'content': response['content']
    })
    
    return response


