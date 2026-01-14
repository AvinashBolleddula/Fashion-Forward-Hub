"""Cross-encoder reranker for improving retrieval results."""

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import logging

from tracer import tracer

logger = logging.getLogger(__name__)

# Load cross-encoder model (happens once on import)
try:
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("Reranker model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load reranker model: {e}")
    reranker_model = None


def rerank_results(
    query: str,
    results: List[Any],
    top_k: int = 5,
    rerank_query: str = None
) -> List[Any]:
    """
    Rerank retrieved results using cross-encoder.
    
    Args:
        query: Original user query
        results: List of Weaviate result objects
        top_k: Number of top results to return after reranking
        rerank_query: Optional alternative query for reranking
    
    Returns:
        List of reranked results (top_k)
    """

    with tracer.start_as_current_span("rerank_results") as span:
        span.set_attribute("query", query)
        span.set_attribute("input_count", len(results))
        span.set_attribute("top_k", top_k)
        span.set_attribute("has_custom_query", rerank_query is not None)

        if not reranker_model:
            logger.warning("Reranker model not available, returning original results")
            span.set_attribute("status", "model_unavailable")
            return results[:top_k]
        
        if not results:
            span.set_attribute("status", "no_results")
            return []
        
        # Use rerank_query if provided, otherwise use original query
        query_to_use = rerank_query if rerank_query else query
        
        try:
            # Extract text content from results
            documents = []
            for result in results:
                if hasattr(result, 'properties'):
                    # For products: use productDisplayName + metadata
                    props = result.properties
                    if 'productDisplayName' in props:
                        doc_text = f"{props['productDisplayName']} {props.get('baseColour', '')} {props.get('season', '')}"
                    # For FAQ: use question + answer
                    elif 'question' in props:
                        doc_text = f"{props['question']} {props['answer']}"
                    else:
                        doc_text = str(props)
                    documents.append(doc_text)
                else:
                    documents.append(str(result))
            
            # Create query-document pairs
            pairs = [[query_to_use, doc] for doc in documents]
            
            # Get relevance scores
            scores = reranker_model.predict(pairs)
            
            # Sort by score (descending) and get top_k
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            reranked_results = [results[i] for i in ranked_indices]
            
            span.set_attribute("output_count", len(reranked_results))
            span.set_attribute("status", "success")
            logger.info(f"Reranked {len(results)} results to top {top_k}")
            return reranked_results
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            span.set_attribute("status", "error")
            span.set_attribute("error", str(e))
            return results[:top_k]


def rerank_with_scores(
    query: str,
    results: List[Any],
    top_k: int = 5
) -> List[Tuple[Any, float]]:
    """
    Rerank and return results with their relevance scores.
    
    Returns:
        List of (result, score) tuples
    """
    if not reranker_model or not results:
        return [(r, 0.0) for r in results[:top_k]]
    
    try:
        documents = []
        for result in results:
            if hasattr(result, 'properties'):
                props = result.properties
                if 'productDisplayName' in props:
                    doc_text = f"{props['productDisplayName']} {props.get('baseColour', '')} {props.get('season', '')}"
                elif 'question' in props:
                    doc_text = f"{props['question']} {props['answer']}"
                else:
                    doc_text = str(props)
                documents.append(doc_text)
            else:
                documents.append(str(result))
        
        pairs = [[query, doc] for doc in documents]
        scores = reranker_model.predict(pairs)
        
        # Sort by score
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:top_k]
        
        return ranked
        
    except Exception as e:
        logger.error(f"Reranking with scores failed: {e}")
        return [(r, 0.0) for r in results[:top_k]]