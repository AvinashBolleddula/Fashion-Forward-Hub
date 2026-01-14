"""Query router to classify user queries as FAQ or Product-related.

This module determines whether a user query should be answered using:
- FAQ database (store policies, contact info, general questions)
- Product database (item search, recommendations, outfit ideas)
"""

from typing import Tuple
from llm import generate_params_dict, generate_with_single_input
from tracer import tracer


def check_if_faq_or_product(query: str, simplified: bool = False) -> Tuple[str, int]:
    """
    Classify a query as FAQ-related or Product-related.
    
    Args:
        query: User's natural language query
        simplified: If True, use shorter prompt (lower token cost)
    
    Returns:
        tuple: (label, total_tokens)
            - label: 'FAQ', 'Product', or 'undefined'
            - total_tokens: Token count from LLM call
    
    Examples:
        FAQ queries:
            - "What is your return policy?"
            - "Where are your stores located?"
            - "How can I contact customer support?"
        
        Product queries:
            - "Show me blue t-shirts under $50"
            - "I need outfit ideas for a beach party"
            - "Do you have winter jackets?"
    """
    with tracer.start_as_current_span("routing_faq_or_product") as span:
        span.set_attribute("query", query)
        span.set_attribute("simplified", simplified)

        if not simplified:
            # Full prompt with detailed examples (higher token cost, better accuracy)
            PROMPT = f"""Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
            Product related answers are answers specific about product information or that needs to use the products to give an answer.
            Examples:
                    Is there a refund for incorrectly bought clothes? Label: FAQ
                    Where are your stores located?: Label: FAQ
                    Tell me about the cheapest T-shirts that you have. Label: Product
                    Do you have blue T-shirts under 100 dollars? Label: Product
                    What are the available sizes for the t-shirts? Label: FAQ
                    How can I contact you via phone? Label: FAQ
                    How can I find the promotions? Label: FAQ
                    Give me ideas for a sunny look. Label: Product
            Return only one of the two labels: FAQ or Product, nothing more.
            Query to classify: {query}
                    """
        else:
            # Simplified prompt (lower token cost, still accurate)
            PROMPT = f"""
    Label the query as FAQ or Product for a clothing store.

    FAQ: store info, policies (refund/return), contact/support, promotions/newsletter, account, sizes.
    Product: asks for items or recommendations using catalog (color/type/price/availability) or outfit/look ideas.

    Examples: refund→FAQ; store location→FAQ; sizes→FAQ; contact/support→FAQ; promotions→FAQ;
    cheapest T-shirts→Product; blue T-shirts under $100→Product; sunny look ideas→Product.

    Return only: FAQ or Product.
    Query: {query}
    """
        
        # Generate parameters for LLM call
        kwargs = generate_params_dict(
            PROMPT,
            temperature=0,      # Deterministic output
            max_tokens=10       # Short response (just "FAQ" or "Product")
        )
        
        # Call LLM
        response = generate_with_single_input(**kwargs)
        
        # Extract label and token count
        label = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']
        
        # Clean up label (handle cases where LLM outputs extra text)
        label_lower = label.lower()
        if 'faq' in label_lower:
            label = 'FAQ'
        elif 'product' in label_lower:
            label = 'Product'
        else:
            label = 'undefined'

        span.set_attribute("label", label)
        span.set_attribute("total_tokens", total_tokens)
        
        return label, total_tokens

    
    


def route_query(query: str, simplified: bool = False) -> Tuple[str, int]:
    """
    Main routing function - alias for check_if_faq_or_product.
    
    This provides a cleaner API for the main RAG pipeline.
    
    Args:
        query: User query
        simplified: Use simplified prompt
    
    Returns:
        tuple: (route, total_tokens)
            - route: 'FAQ' or 'Product'
            - total_tokens: Token usage
    """
    return check_if_faq_or_product(query, simplified)

