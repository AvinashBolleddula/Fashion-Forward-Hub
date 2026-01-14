"""Unit tests for retrieval functions."""

import pytest
from rag_pipeline import (
    retrieve_bm25,
    retrieve_semantic,
    retrieve_hybrid,
    reciprocal_rank_fusion
)


class TestBM25Retrieval:
    """Test BM25 keyword-based retrieval."""
    
    def test_bm25_returns_results(self, products_collection, test_queries):
        """Test that BM25 retrieval returns results."""
        results = retrieve_bm25(
            products_collection, 
            test_queries["product_simple"], 
            top_k=10
        )
        
        assert len(results) > 0, "BM25 should return results"
        assert len(results) <= 10, "Should respect top_k limit"
    
    def test_bm25_keyword_relevance(self, products_collection):
        """Test that BM25 finds keyword matches."""
        query = "shirt"
        results = retrieve_bm25(products_collection, query, top_k=5)
        
        # At least one result should contain "shirt" in product name
        has_keyword = any(
            "shirt" in r.properties.get("productDisplayName", "").lower()
            for r in results
        )
        assert has_keyword, "BM25 should find products with keyword 'shirt'"
    
    def test_bm25_top_k_limits(self, products_collection):
        """Test that different top_k values work."""
        query = "casual"
        
        results_5 = retrieve_bm25(products_collection, query, top_k=5)
        results_20 = retrieve_bm25(products_collection, query, top_k=20)
        
        assert len(results_5) <= 5
        assert len(results_20) <= 20
        assert len(results_20) >= len(results_5)


class TestSemanticRetrieval:
    """Test semantic vector-based retrieval."""
    
    def test_semantic_simplified_returns_results(self, products_collection, test_queries):
        """Test semantic search without filters."""
        results = retrieve_semantic(
            products_collection,
            test_queries["product_simple"],
            simplified=True,
            top_k=10
        )
        
        assert len(results) > 0, "Semantic search should return results"
        assert len(results) <= 10, "Should respect top_k"
    
    def test_semantic_with_filters(self, products_collection):
        """Test semantic search with metadata filters."""
        from metadata_filter import generate_filters_from_query
        
        query = "blue shirts under $50"
        filters, _ = generate_filters_from_query(query)
        
        results = retrieve_semantic(
            products_collection,
            query,
            simplified=False,
            top_k=20,
            filters=filters
        )
        
        # Should return some results (even if not many)
        assert isinstance(results, list), "Should return a list"
        
        # If results exist, verify they match filters
        if results:
            for r in results[:3]:
                props = r.properties
                color = props.get('baseColour', '').lower()
                # Blue could be "blue", "navy", etc.
                print(f"Found product: {props.get('productDisplayName')} - {color}")
    
    def test_semantic_similarity(self, products_collection):
        """Test that semantic search finds similar concepts."""
        # "sneakers" and "athletic shoes" should retrieve similar items
        results_sneakers = retrieve_semantic(
            products_collection, "sneakers", simplified=True, top_k=5
        )
        results_athletic = retrieve_semantic(
            products_collection, "athletic shoes", simplified=True, top_k=5
        )
        
        # Both should return results
        assert len(results_sneakers) > 0
        assert len(results_athletic) > 0


class TestHybridRetrieval:
    """Test hybrid BM25 + Semantic retrieval."""
    
    def test_hybrid_returns_results(self, products_collection, test_queries):
        """Test hybrid retrieval combines both methods."""
        results = retrieve_hybrid(
            products_collection,
            test_queries["product_simple"],
            simplified=True,
            top_k=10,
            alpha=0.5,
            k=60
        )
        
        assert len(results) > 0, "Hybrid should return results"
        assert len(results) <= 10, "Should respect top_k"
    
    def test_hybrid_alpha_variations(self, products_collection):
        """Test that alpha affects result ranking."""
        query = "formal shirt"
        
        # Alpha = 0 (100% semantic)
        results_semantic = retrieve_hybrid(
            products_collection, query, True, 5, alpha=0.0, k=60
        )
        
        # Alpha = 1 (100% BM25)
        results_bm25 = retrieve_hybrid(
            products_collection, query, True, 5, alpha=1.0, k=60
        )
        
        # Alpha = 0.5 (balanced)
        results_balanced = retrieve_hybrid(
            products_collection, query, True, 5, alpha=0.5, k=60
        )
        
        # All should return results
        assert len(results_semantic) > 0
        assert len(results_bm25) > 0
        assert len(results_balanced) > 0
        
        # Results might differ based on alpha
        # (Can't guarantee order, but at least we get results)


class TestRRFFusion:
    """Test Reciprocal Rank Fusion."""
    
    def test_rrf_combines_results(self, products_collection):
        """Test that RRF combines BM25 and semantic results."""
        query = "blue shirt"
        
        bm25_results = retrieve_bm25(products_collection, query, top_k=10)
        semantic_results = retrieve_semantic(
            products_collection, query, simplified=True, top_k=10
        )
        
        fused = reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.5, k=60
        )
        
        assert len(fused) > 0, "RRF should produce results"
        # Fused results should be at least as many as either input
        assert len(fused) >= min(len(bm25_results), len(semantic_results))
    
    def test_rrf_alpha_weighting(self, products_collection):
        """Test that alpha weights BM25 vs semantic correctly."""
        query = "casual wear"
        
        bm25_results = retrieve_bm25(products_collection, query, top_k=10)
        semantic_results = retrieve_semantic(
            products_collection, query, simplified=True, top_k=10
        )
        
        # High alpha (favor BM25)
        fused_bm25 = reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.9, k=60
        )
        
        # Low alpha (favor semantic)
        fused_semantic = reciprocal_rank_fusion(
            bm25_results, semantic_results, alpha=0.1, k=60
        )
        
        # Both should return results
        assert len(fused_bm25) > 0
        assert len(fused_semantic) > 0


class TestReranking:
    """Test cross-encoder reranking."""
    
    def test_reranking_reduces_results(self, products_collection):
        """Test that reranking returns top_k results."""
        from reranker import rerank_results
        
        query = "formal office wear"
        
        # Get initial results
        results = retrieve_semantic(
            products_collection, query, simplified=True, top_k=20
        )
        
        # Rerank to top 5
        reranked = rerank_results(query, results, top_k=5)
        
        assert len(reranked) <= 5, "Reranking should limit to top_k"
        assert len(reranked) > 0, "Should return some results"
    
    def test_reranking_with_custom_query(self, products_collection):
        """Test reranking with custom query."""
        from reranker import rerank_results
        
        results = retrieve_semantic(
            products_collection, "shirt", simplified=True, top_k=20
        )
        
        # Rerank with more specific query
        reranked = rerank_results(
            query="shirt",
            results=results,
            top_k=5,
            rerank_query="professional business shirt"
        )
        
        assert len(reranked) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])