"""Metadata filtering for product queries.

This module extracts metadata filters from natural language queries
and converts them to Weaviate filters for precise product search.
"""

import json
import joblib
from typing import Dict, List, Optional, Tuple
from weaviate.classes.query import Filter
from llm import generate_with_single_input
from config import PRODUCTS_FILE
from tracer import tracer

products_data = joblib.load(str(PRODUCTS_FILE))  # Convert Path to string

# Build dictionary of possible values for each metadata field
# This helps the LLM know what values are valid
values = {}
for product in products_data:
    for key, val in product.items():
        # Skip fields that aren't used for filtering
        if key in ('product_id', 'price', 'productDisplayName', 'subCategory', 'year'):
            continue
        if key not in values:
            values[key] = set()
        values[key].add(val)

# Convert sets to lists for JSON serialization
values = {k: list(v) for k, v in values.items()}


def parse_json_output(llm_output: str) -> Optional[Dict]:
    """
    Parse and clean JSON output from LLM.
    
    LLMs sometimes return malformed JSON with extra quotes, newlines, or brackets.
    This function cleans and parses the output.
    
    Args:
        llm_output: Raw string output from LLM (expected to be JSON)
    
    Returns:
        dict: Parsed JSON object, or None if parsing fails
    
    Example:
        >>> llm_output = '{"gender": ["Men"]}'
        >>> parse_json_output(llm_output)
        {'gender': ['Men']}
    """
    try:
        # Clean common formatting issues
        llm_output = llm_output.replace("\n", '')      # Remove newlines
        llm_output = llm_output.replace("'", '')       # Remove single quotes
        llm_output = llm_output.replace("}}", "}")     # Fix double closing braces
        llm_output = llm_output.replace("{{", "{")     # Fix double opening braces
        
        # Parse JSON
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print(f"Raw output: {llm_output}")
        return None


def get_filter_by_metadata(json_output: Optional[Dict] = None) -> Optional[List[Filter]]:
    """
    Convert metadata dictionary to Weaviate filters.
    
    Takes a dictionary of metadata (e.g., {"gender": ["Men"], "price": {"min": 0, "max": 100}})
    and converts it to Weaviate Filter objects for database queries.
    
    Args:
        json_output: Dictionary with metadata keys and values
                     - Text fields: {"gender": ["Men", "Women"]}
                     - Price field: {"price": {"min": 50, "max": 200}}
    
    Returns:
        list[Filter]: List of Weaviate filters, or None if input is None
    
    Example:
        >>> metadata = {
        ...     "gender": ["Men"],
        ...     "baseColour": ["Blue", "Navy"],
        ...     "price": {"min": 50, "max": 150}
        ... }
        >>> filters = get_filter_by_metadata(metadata)
        >>> # Returns: [Filter for gender=Men, Filter for color in [Blue,Navy], Filter for 50<price<150]
    """
    if json_output is None:
        return None
    
    # Valid metadata keys that can be used for filtering
    valid_keys = (
        'gender',
        'masterCategory',
        'articleType',
        'baseColour',
        'price',
        'usage',
        'season',
    )
    
    filters = []
    
    for key, value in json_output.items():
        # Skip invalid keys
        if key not in valid_keys:
            continue
        
        # Special handling for price (range filter)
        if key == 'price':
            if not isinstance(value, dict):
                continue
            
            min_price = value.get('min')
            max_price = value.get('max')
            
            # Skip if missing min/max
            if min_price is None or max_price is None:
                continue
            
            # Skip if min_price is invalid or max_price is infinity
            if min_price <= 0 or max_price == 'inf':
                continue
            
            # Add range filters
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        
        else:
            # For text fields, use contains_any (matches any value in list)
            # e.g., baseColour contains_any ["Blue", "Navy"]
            filters.append(Filter.by_property(key).contains_any(value))
    
    return filters


def generate_metadata_from_query(query: str) -> Tuple[str, int]:
    """
    Extract metadata filters from a natural language query using LLM.
    
    This function uses an LLM to analyze a query like "blue dresses for women under $100"
    and extract structured metadata like:
    {
        "gender": ["Women"],
        "articleType": ["Dresses"],
        "baseColour": ["Blue"],
        "price": {"min": 0, "max": 100}
    }
    
    Args:
        query: Natural language query about clothing
               e.g., "Show me cheap winter jackets for men"
    
    Returns:
        tuple: (json_string, total_tokens)
            - json_string: String containing JSON metadata
            - total_tokens: Number of tokens used in LLM call
    
    Example:
        >>> query = "blue t-shirts for men under $50"
        >>> json_str, tokens = generate_metadata_from_query(query)
        >>> print(json_str)
        '{"gender": ["Men"], "articleType": ["Tshirts"], "baseColour": ["Blue"], "price": {"min": 0, "max": 50}, ...}'
    """


    with tracer.start_as_current_span("generate_metadata") as span:
        span.set_attribute("query", query)
    
        # Construct prompt for LLM
        PROMPT = f"""
    One query will be provided. For the given query, there will be a call on vector database to query relevant cloth items. 
    Generate a JSON with useful metadata to filter the products in the query. Possible values for each feature is in the following json: {values}

    Provide a JSON with the features that best fit in the query (can be more than one, write in a list). Also, if present, add a price key, saying if there is a price range (between values, greater than or smaller than some value).
    Only return the JSON, nothing more. price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound). 
    Always include gender, masterCategory, articleType, baseColour, price, usage and season as keys. All values must be within lists.
    If there is no price set, add min = 0 and max = inf.
    Only include values that are given in the json above. 

    Example of expected JSON:

    {{
    "gender": ["Women"],
    "masterCategory": ["Apparel"],
    "articleType": ["Dresses"],
    "baseColour": ["Blue"],
    "price": {{"min": 0, "max": "inf"}},
    "usage": ["Formal"],
    "season": ["All seasons"]
    }}

    Query: {query}
    """
        
        # Call LLM with low temperature (deterministic) and high max_tokens
        response = generate_with_single_input(
            PROMPT,
            temperature=0,      # Deterministic output
            max_tokens=1500     # Allow for detailed JSON
        )
        
        # Extract content and token count
        content = response['content']
        total_tokens = response['total_tokens']
        span.set_attribute("content", content)
        span.set_attribute("total_tokens", total_tokens)
        return content, total_tokens


def generate_filters_from_query(query: str) -> Tuple[Optional[List[Filter]], int]:
    """
    Complete pipeline: Query -> Metadata JSON -> Weaviate Filters.
    
    This is the main function that combines all steps:
    1. Extract metadata from query using LLM
    2. Parse the JSON output
    3. Convert to Weaviate filters
    
    Args:
        query: Natural language query
    
    Returns:
        tuple: (filters, total_tokens)
            - filters: List of Weaviate Filter objects (or None if failed)
            - total_tokens: Token count from LLM call
    
    Example:
        >>> query = "Show me blue summer dresses for women under $100"
        >>> filters, tokens = generate_filters_from_query(query)
        >>> print(f"Generated {len(filters)} filters using {tokens} tokens")
    """
    # Step 1: Generate metadata JSON from query
    json_string, total_tokens = generate_metadata_from_query(query)
    
    # Step 2: Parse JSON
    json_output = parse_json_output(json_string)
    
    # Step 3: Convert to Weaviate filters
    filters = get_filter_by_metadata(json_output)
    
    return filters, total_tokens

