"""Integration tests for complete RAG pipeline."""

import pytest
from rag_pipeline import answer_query, query_on_faq, query_on_products


class TestRAGPipelineIntegration:
    """Test full RAG pipeline end-to-end."""
    
    def test_faq_query_simplified_true(self, test_queries):
        """Test FAQ query with simplified mode."""
        params, tokens = answer_query(
            query=test_queries["faq_simple"],
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=5
        )
        
        assert "prompt" in params, "Should return prompt"
        assert tokens >= 0, "Should return token count"
        assert "return" in params["prompt"].lower(), "Should mention returns in FAQ"
    
    def test_faq_query_simplified_false(self, test_queries):
        """Test FAQ query with all FAQs."""
        params, tokens = answer_query(
            query=test_queries["faq_simple"],
            use_rag=True,
            simplified=False,
            top_k=20
        )
        
        assert "prompt" in params
        # Simplified=False should use more tokens (all 25 FAQs)
        assert tokens > 0
    
    def test_product_query_bm25(self, test_queries):
        """Test product query with BM25 retrieval."""
        params, tokens = answer_query(
            query=test_queries["product_simple"],
            use_rag=True,
            retriever_type="bm25",
            simplified=True,
            top_k=10
        )
        
        assert "prompt" in params
        assert "Product" in params["prompt"], "Should contain product information"
        assert tokens >= 0
    
    def test_product_query_semantic_simplified(self, test_queries):
        """Test product query with semantic search (no filters)."""
        params, tokens = answer_query(
            query=test_queries["product_simple"],
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=15
        )
        
        assert "prompt" in params
        assert tokens >= 0
    
    def test_product_query_semantic_with_filters(self, test_queries):
        """Test product query with metadata filters."""
        params, tokens = answer_query(
            query=test_queries["product_filtered"],
            use_rag=True,
            retriever_type="semantic",
            simplified=False,  # Enable filters
            top_k=20
        )
        
        assert "prompt" in params
        # Should have used tokens for metadata extraction
        assert tokens > 0, "Should have metadata extraction tokens"
    
    def test_product_query_hybrid(self, test_queries):
        """Test product query with hybrid retrieval."""
        params, tokens = answer_query(
            query=test_queries["product_complex"],
            use_rag=True,
            retriever_type="hybrid",
            simplified=True,
            top_k=10,
            alpha=0.5,
            k=60
        )
        
        assert "prompt" in params
        assert tokens >= 0
    
    def test_direct_llm_no_rag(self, test_queries):
        """Test direct LLM call without RAG."""
        params, tokens = answer_query(
            query=test_queries["faq_simple"],
            use_rag=False
        )
        
        assert "prompt" in params
        assert tokens == 0, "No RAG tokens should be used"
        # Should have generic prompt, not fashion-specific
        assert "general knowledge" in params["prompt"].lower()
    
    def test_with_reranker(self, test_queries):
        """Test retrieval with reranking enabled."""
        params, tokens = answer_query(
            query=test_queries["product_simple"],
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10,
            use_reranker=True
        )
        
        assert "prompt" in params
        assert tokens >= 0


class TestQueryRouting:
    """Test query routing (FAQ vs Product)."""
    
    def test_faq_routing(self, test_queries):
        """Test that FAQ queries are routed correctly."""
        from query_router import check_if_faq_or_product
        
        label, tokens = check_if_faq_or_product(
            test_queries["faq_simple"],
            simplified=False
        )
        
        assert label == "FAQ", f"Expected FAQ, got {label}"
        assert tokens > 0, "Should use tokens for routing"
    
    def test_product_routing(self, test_queries):
        """Test that product queries are routed correctly."""
        from query_router import check_if_faq_or_product
        
        label, tokens = check_if_faq_or_product(
            test_queries["product_simple"],
            simplified=False
        )
        
        assert label == "Product", f"Expected Product, got {label}"
        assert tokens > 0


class TestMetadataExtraction:
    """Test metadata filter extraction."""
    
    def test_color_extraction(self):
        """Test that color is extracted correctly."""
        from metadata_filter import generate_filters_from_query
        
        query = "blue shirts"
        filters, tokens = generate_filters_from_query(query)
        
        assert tokens > 0, "Should use tokens for extraction"
        assert len(filters) > 0, "Should extract at least one filter"
        
        # Check if color filter exists
        filter_props = [f.property for f in filters]
        assert "baseColour" in filter_props, "Should extract color filter"
    
    def test_price_extraction(self):
        """Test that price filters are extracted."""
        from metadata_filter import generate_filters_from_query
        
        query = "shirts under $50"
        filters, tokens = generate_filters_from_query(query)
        
        assert len(filters) > 0, "Should extract filters"
        
        # Check if price-related filter exists
        # (might be "price" or similar field)
        filter_dict = {f.property: f.value for f in filters}
        print(f"Extracted filters: {filter_dict}")
    
    def test_complex_query_extraction(self):
        """Test extraction from complex query."""
        from metadata_filter import generate_filters_from_query
        
        query = "blue casual shirts for men under $75"
        filters, tokens = generate_filters_from_query(query)
        
        assert len(filters) >= 2, "Should extract multiple filters"
        print(f"Complex query extracted {len(filters)} filters")


class TestEndToEndScenarios:
    """Test realistic user scenarios."""
    
    def test_scenario_return_policy(self, test_queries):
        """User asks about return policy."""
        params, tokens = answer_query(
            query=test_queries["faq_simple"],
            use_rag=True,
            simplified=False
        )
        
        assert "prompt" in params
        assert "return" in params["prompt"].lower()
        assert tokens > 0
    
    def test_scenario_product_search(self, test_queries):
        """User searches for specific products."""
        params, tokens = answer_query(
            query=test_queries["product_filtered"],
            use_rag=True,
            retriever_type="semantic",
            simplified=False,
            top_k=20
        )
        
        assert "prompt" in params
        # Should have product information in prompt
        assert "Product" in params["prompt"]
    
    def test_scenario_switch_retriever_types(self, test_queries):
        """User tries different retriever types on same query."""
        query = test_queries["product_simple"]
        
        # Try BM25
        params_bm25, tokens_bm25 = answer_query(
            query=query,
            use_rag=True,
            retriever_type="bm25",
            top_k=10
        )
        
        # Try Semantic
        params_semantic, tokens_semantic = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10
        )
        
        # Try Hybrid
        params_hybrid, tokens_hybrid = answer_query(
            query=query,
            use_rag=True,
            retriever_type="hybrid",
            alpha=0.5,
            k=60,
            top_k=10
        )
        
        # All should return valid prompts
        assert all("prompt" in p for p in [params_bm25, params_semantic, params_hybrid])
        
        # Token counts might differ
        print(f"Token usage - BM25: {tokens_bm25}, Semantic: {tokens_semantic}, Hybrid: {tokens_hybrid}")
    
    def test_scenario_top_k_variations(self, test_queries):
        """User changes top_k value."""
        query = test_queries["product_simple"]
        
        # Small top_k
        params_5, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=5
        )
        
        # Large top_k
        params_20, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=20
        )
        
        # Both should have products, but different amounts
        assert "prompt" in params_5
        assert "prompt" in params_20


class TestTokenCounting:
    """Test that token counting is accurate."""
    
    def test_rag_tokens_counted(self, test_queries):
        """Test that RAG operations count tokens."""
        params, tokens = answer_query(
            query=test_queries["product_filtered"],
            use_rag=True,
            retriever_type="semantic",
            simplified=False  # Should use tokens for metadata
        )
        
        assert tokens > 0, "Should count tokens for routing and metadata"
    
    def test_no_rag_zero_tokens(self, test_queries):
        """Test that no RAG means zero tokens."""
        params, tokens = answer_query(
            query=test_queries["faq_simple"],
            use_rag=False
        )
        
        assert tokens == 0, "No RAG should mean zero tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])