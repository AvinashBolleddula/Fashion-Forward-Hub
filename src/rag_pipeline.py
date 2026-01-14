"""Main RAG pipeline that orchestrates the entire query answering process.

This module ties together all components:
1. Route query (FAQ vs Product)
2. Retrieve relevant context (FAQs or Products)
3. Generate prompts with context
4. Return parameters for final LLM call
"""

import joblib
from typing import Dict, Tuple, List, Optional, Any
from weaviate.classes.query import Filter
import logging

from database import get_client
from llm import generate_params_dict, generate_with_single_input
from query_router import check_if_faq_or_product
from metadata_filter import generate_filters_from_query
from config import FAQ_FILE, OPENAI_MODEL as DEFAULT_MODEL
from tracer import tracer
from reranker import rerank_results

logger = logging.getLogger(__name__)

# Load FAQ data
faq_data = joblib.load(str(FAQ_FILE))


def generate_faq_layout(faq_list: list) -> str:
    """Format FAQ list into a readable string layout."""
    layout = ""
    for faq in faq_list:
        layout += f"Question: {faq['question']} Answer: {faq['answer']} Type: {faq['type']}\n"
    return layout


def query_on_faq(query: str, simplified: bool = False, top_k: int = 20,**kwargs) -> Dict[str, Any]:
    """
    Handle FAQ queries by retrieving relevant FAQs and preparing prompt.
    
    Args:
        query: User's FAQ question
        simplified: If True, use semantic search for top 5 FAQs (saves tokens)
        **kwargs: Additional parameters for LLM generation
    
    Returns:
        dict: Parameters dictionary ready for LLM call
    """
    with tracer.start_as_current_span("query_on_faq") as span:
        span.set_attribute("query", query)
        span.set_attribute("simplified", simplified)
        

        if not simplified:
            # Use all FAQs (higher token cost, comprehensive coverage)
            faq_layout = generate_faq_layout(faq_data)
            
            PROMPT = f"""You will be provided with an FAQ for a clothing store. 
Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ. 
<FAQ_ITEMS>
PROVIDED FAQ: {faq_layout}
</FAQ_ITEMS>
Question: {query}
"""
        else:
            span.set_attribute("top_k", top_k)
            # Use semantic search for top 5 most relevant FAQs (lower token cost)
            client = get_client()
            faq_collection = client.collections.get("faq")
            
            # Semantic search
            results = faq_collection.query.near_text(query, limit=top_k)
            
            # Convert to list of dicts and reverse (most relevant last, closer to query)
            faq_list = [obj.properties for obj in results.objects]
            faq_list.reverse()
            
            faq_layout = generate_faq_layout(faq_list)
            client.close()
            
            PROMPT = (f"You will be provided with a query for a clothing store regarding FAQ. It will be provided relevant FAQ from the clothing store. "
                     f"Answer the query based on the relevant FAQ provided. They are ordered in decreasing relevance, so the first is the most relevant FAQ and the last is the least relevant. "
                     f"Answer the instruction based on them. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.\n"
                     f"<FAQ>\n"
                     f"RELEVANT FAQ ITEMS:\n{faq_layout}\n"
                     f"</FAQ>\n"
                     f"Query: {query}")
        
        # Generate parameters dict
        params = generate_params_dict(PROMPT, **kwargs)
        return params


def retrieve_bm25(collection, query: str, top_k: int = 20) -> List:
    """BM25 keyword-based retrieval using Weaviate."""
    with tracer.start_as_current_span("retrieve_bm25") as span:
        span.set_attribute("query", query)
        span.set_attribute("top_k", top_k)
        
        try:
            results = collection.query.bm25(query=query, limit=top_k)
            span.set_attribute("num_results", len(results.objects))
            logger.info(f"BM25 retrieved {len(results.objects)} results")
            return results.objects
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            span.set_attribute("error", str(e))
            return []


def retrieve_semantic(
    collection,
    query: str,
    simplified: bool = False,
    top_k: int = 20,
    filters: List[Filter] = None
) -> List:
    """Semantic vector-based retrieval."""
    with tracer.start_as_current_span("retrieve_semantic") as span:
        span.set_attribute("query", query)
        span.set_attribute("simplified", simplified)
        span.set_attribute("top_k", top_k)
        span.set_attribute("has_filters", filters is not None)

        try:
            if simplified or not filters:
                # Direct semantic search
                results = collection.query.near_text(query=query, limit=top_k)
            else:
                # Semantic with filters
                results = collection.query.near_text(
                    query=query,
                    filters=Filter.all_of(filters),
                    limit=top_k
                )
            
            span.set_attribute("num_results", len(results.objects))
            logger.info(f"Semantic retrieved {len(results.objects)} results")
            return results.objects
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            span.set_attribute("error", str(e))
            return []


def reciprocal_rank_fusion(
    bm25_results: List,
    semantic_results: List,
    alpha: float = 0.5,
    k: int = 60
) -> List:
    """Combine BM25 and semantic results using Reciprocal Rank Fusion."""
    with tracer.start_as_current_span("reciprocal_rank_fusion") as span:
        span.set_attribute("alpha", alpha)
        span.set_attribute("k", k)
        span.set_attribute("bm25_count", len(bm25_results))
        span.set_attribute("semantic_count", len(semantic_results))

        scores = {}
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            result_id = result.uuid if hasattr(result, 'uuid') else str(result)
            scores[result_id] = {
                'result': result,
                'bm25_score': alpha / (k + rank + 1),
                'semantic_score': 0.0
            }
        
        # Score semantic results
        for rank, result in enumerate(semantic_results):
            result_id = result.uuid if hasattr(result, 'uuid') else str(result)
            if result_id in scores:
                scores[result_id]['semantic_score'] = (1 - alpha) / (k + rank + 1)
            else:
                scores[result_id] = {
                    'result': result,
                    'bm25_score': 0.0,
                    'semantic_score': (1 - alpha) / (k + rank + 1)
                }
        
        # Calculate final scores
        for result_id in scores:
            scores[result_id]['final_score'] = (
                scores[result_id]['bm25_score'] + 
                scores[result_id]['semantic_score']
            )
        
        # Sort by final score
        ranked = sorted(
            scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        span.set_attribute("fused_count", len(ranked))
        logger.info(f"RRF fused {len(ranked)} results")
        return [item['result'] for item in ranked]


def retrieve_hybrid(
    collection,
    query: str,
    simplified: bool = False,
    top_k: int = 20,
    alpha: float = 0.5,
    k: int = 60,
    filters: List[Filter] = None
) -> List:
    """Hybrid retrieval: BM25 + Semantic + RRF fusion."""
    with tracer.start_as_current_span("retrieve_hybrid") as span:
        span.set_attribute("query", query)
        span.set_attribute("retriever_type", "hybrid")
        span.set_attribute("alpha", alpha)
        span.set_attribute("k", k)

        # Get BM25 results
        bm25_results = retrieve_bm25(collection, query, top_k * 2)
        
        # Get semantic results
        semantic_results = retrieve_semantic(
            collection, query, simplified, top_k * 2, filters
        )
        
        # Fuse with RRF
        fused_results = reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha, k
        )

        span.set_attribute("final_count", len(fused_results[:top_k]))
        return fused_results[:top_k]


def get_relevant_products_from_query(
    query: str,
    retriever_type: str = "semantic",
    simplified: bool = False,
    top_k: int = 20,
    alpha: float = 0.5,
    k: int = 60,
    use_reranker: bool = False,
    rerank_query: str = None
) -> Tuple[List, int]:
    """Retrieve relevant products with configurable retrieval method."""
    with tracer.start_as_current_span("get_relevant_products") as span:
        span.set_attribute("query", query)
        span.set_attribute("retriever_type", retriever_type)
        span.set_attribute("simplified", simplified)
        span.set_attribute("top_k", top_k)
        span.set_attribute("use_reranker", use_reranker)
    
        client = get_client()
        products_collection = client.collections.get("products")
        
        total_tokens = 0
        filters = None
        
        # Generate filters for semantic/hybrid (unless simplified)
        if retriever_type in ["semantic", "hybrid"] and not simplified:
            filters, tokens = generate_filters_from_query(query)
            total_tokens += tokens
            span.set_attribute("filters_generated", filters is not None)
        
        # Retrieve based on type
        if retriever_type == "bm25":
            results = retrieve_bm25(products_collection, query, top_k)
        
        elif retriever_type == "semantic":
            results = retrieve_semantic(
                products_collection, query, simplified, top_k, filters
            )
        
        elif retriever_type == "hybrid":
            results = retrieve_hybrid(
                products_collection, query, simplified, top_k, alpha, k, filters
            )
        
        else:
            logger.error(f"Unknown retriever type: {retriever_type}")
            results = []
    
        # Apply reranking if requested
        if use_reranker and results:
            results = rerank_results(query, results, top_k, rerank_query)
        
        span.set_attribute("final_results", len(results))
        span.set_attribute("total_tokens", total_tokens)
        
        client.close()
        return results, total_tokens


def generate_products_layout(products: List) -> str:
    """Format product objects into a readable layout for the LLM."""
    layout = ""
    for product in products:
        props = product.properties
        layout += (f"Product ID: {props['product_id']}. "
                  f"Product name: {props['productDisplayName']}. "
                  f"Product Color: {props['baseColour']}. "
                  f"Product Season: {props['season']}. "
                  f"Product Year: {props['year']}.\n")
    return layout


def query_on_products(
    query: str,
    retriever_type: str = "semantic",
    simplified: bool = False,
    top_k: int = 20,
    alpha: float = 0.5,
    k: int = 60,
    use_reranker: bool = False,
    rerank_query: str = None,
    **kwargs
) -> Tuple[Dict[str, Any], int]:
    """Handle product queries with configurable retrieval."""
    with tracer.start_as_current_span("query_on_products") as span:
        span.set_attribute("query", query)
        span.set_attribute("retriever_type", retriever_type)
        span.set_attribute("simplified", simplified)
        span.set_attribute("top_k", top_k)
        span.set_attribute("use_reranker", use_reranker)
        
        if retriever_type == "hybrid":
            span.set_attribute("alpha", alpha)
            span.set_attribute("k", k)
        
        # Get relevant products with ALL parameters
        products, total_tokens = get_relevant_products_from_query(
            query=query,
            retriever_type=retriever_type,
            simplified=simplified,
            top_k=top_k,
            alpha=alpha,
            k=k,
            use_reranker=use_reranker,
            rerank_query=rerank_query
        )
        
        span.set_attribute("products_retrieved", len(products))
        span.set_attribute("total_tokens", total_tokens)
        
        # Format products
        products_layout = generate_products_layout(products)
        
        # Create prompt
        PROMPT = f"""You are a helpful fashion assistant. You will be provided with a list of products from our catalog.
Based on these products, answer the user's query. Provide specific product recommendations with their IDs and names.

<PRODUCTS>
{products_layout}
</PRODUCTS>

User Query: {query}

Provide helpful recommendations based on the available products above."""
        
        # Generate parameters dict
        params = generate_params_dict(PROMPT, **kwargs)
        return params, total_tokens


def answer_query(
    query: str,
    model: str = None,
    use_rag: bool = True,
    retriever_type: str = "semantic",
    simplified: bool = False,
    top_k: int = 20,
    alpha: float = 0.5,
    k: int = 60,
    use_reranker: bool = False,
    rerank_query: str = None
) -> Tuple[Dict[str, Any], int]:
    """Main RAG pipeline with configurable retrieval."""
    with tracer.start_as_current_span("answer_query") as span:
        span.set_attribute("query", query)
        span.set_attribute("use_rag", use_rag)
        span.set_attribute("model", model if model else DEFAULT_MODEL)

        # Only log retrieval params if RAG is enabled
        if use_rag:
            span.set_attribute("retriever_type", retriever_type)
            span.set_attribute("simplified", simplified)
            span.set_attribute("top_k", top_k)
            span.set_attribute("use_reranker", use_reranker)
            
            if retriever_type == "hybrid":
                span.set_attribute("alpha", alpha)
                span.set_attribute("k", k)
        
        if model is None:
            model = DEFAULT_MODEL
        
        span.set_attribute("model", model)
        total_tokens = 0
        
        # If RAG disabled, use direct LLM with GENERIC system prompt
        if not use_rag:
            span.set_attribute("route", "direct_llm")
            PROMPT = f"""Answer the following question based on your general knowledge. Do not make up specific company policies or information.

Question: {query}"""
            params = generate_params_dict(PROMPT, model=model)
            return params, 0
        
        # Step 1: Route the query
        label, tokens = check_if_faq_or_product(query, simplified=simplified)
        total_tokens += tokens
        span.set_attribute("route", label)
        span.set_attribute("routing_tokens", tokens)
        
        # Step 2: Handle based on label
        if label == 'FAQ':
            params = query_on_faq(query, simplified=simplified, top_k=top_k)
        
        elif label == 'Product':
            try:
                # Pass ALL parameters correctly
                params, tokens = query_on_products(
                    query=query,
                    retriever_type=retriever_type,
                    simplified=simplified,
                    top_k=top_k,
                    alpha=alpha,
                    k=k,
                    use_reranker=use_reranker,
                    rerank_query=rerank_query
                )
                total_tokens += tokens
                span.set_attribute("retrieval_tokens", tokens)
            except Exception as e:
                logger.error(f"Product query failed: {e}", exc_info=True)
                span.set_attribute("error", str(e))
                return {
                    "prompt": f"Error processing query. Please try rephrasing. Query: {query}",
                    "model": model
                }, total_tokens
        
        else:
            span.set_attribute("route", "undefined")
            return {
                "prompt": f"Answer based on your general knowledge. Query: {query}",
                "model": model
            }, total_tokens
        
        # Step 3: Set model
        params['model'] = model
        span.set_attribute("total_tokens", total_tokens)
        
        return params, total_tokens