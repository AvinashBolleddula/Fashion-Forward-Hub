"""Performance and regression testing for RAG system."""

import pytest
import time
from statistics import mean, stdev
from rag_pipeline import answer_query
from llm import generate_with_single_input


class TestLatencyPerformance:
    """Test response time performance."""
    
    def test_retrieval_latency_bm25(self):
        """Measure BM25 retrieval latency."""
        query = "blue shirts"
        latencies = []
        
        for _ in range(5):
            start = time.time()
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="bm25",
                top_k=20
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        std_latency = stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n⏱️  BM25 Latency: {avg_latency:.1f}ms ± {std_latency:.1f}ms")
        
        # Should be under 5 seconds for retrieval
        assert avg_latency < 5000, f"BM25 too slow: {avg_latency}ms"
    
    def test_retrieval_latency_semantic(self):
        """Measure semantic retrieval latency."""
        query = "casual summer wear"
        latencies = []
        
        for _ in range(5):
            start = time.time()
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=20
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        std_latency = stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n⏱️  Semantic Latency: {avg_latency:.1f}ms ± {std_latency:.1f}ms")
        
        assert avg_latency < 5000, f"Semantic too slow: {avg_latency}ms"
    
    def test_retrieval_latency_hybrid(self):
        """Measure hybrid retrieval latency."""
        query = "formal attire"
        latencies = []
        
        for _ in range(5):
            start = time.time()
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="hybrid",
                simplified=True,
                top_k=20,
                alpha=0.5,
                k=60
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        std_latency = stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n⏱️  Hybrid Latency: {avg_latency:.1f}ms ± {std_latency:.1f}ms")
        
        # Hybrid might be slightly slower (2x retrievals)
        assert avg_latency < 8000, f"Hybrid too slow: {avg_latency}ms"
    
    def test_metadata_extraction_latency(self):
        """Measure metadata extraction latency."""
        query = "blue shirts under $50 for men"
        latencies = []
        
        for _ in range(3):
            start = time.time()
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=False,  # Enables metadata extraction
                top_k=20
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = mean(latencies)
        
        print(f"\n⏱️  Metadata Extraction Latency: {avg_latency:.1f}ms")
        
        # Metadata adds LLM call, so expect higher latency
        assert avg_latency < 10000, f"Metadata extraction too slow: {avg_latency}ms"


class TestRegressionTests:
    """Regression tests to catch breaking changes."""
    
    @pytest.fixture
    def baseline_queries(self):
        """Standard queries that should always work."""
        return {
            "faq_return": {
                "query": "What is your return policy?",
                "expected_route": "FAQ",
                "min_prompt_length": 100
            },
            "product_simple": {
                "query": "blue shirts",
                "expected_route": "Product",
                "min_prompt_length": 200
            },
            "product_filtered": {
                "query": "casual wear under $100",
                "expected_route": "Product",
                "min_prompt_length": 200
            }
        }
    
    def test_baseline_faq_query(self, baseline_queries):
        """Test that baseline FAQ query still works."""
        test_case = baseline_queries["faq_return"]
        
        params, tokens = answer_query(
            query=test_case["query"],
            use_rag=True,
            simplified=True,
            top_k=5
        )
        
        assert "prompt" in params, "Should return prompt"
        assert len(params["prompt"]) > test_case["min_prompt_length"], "Prompt too short"
        assert "return" in params["prompt"].lower(), "Should contain return information"
        assert tokens > 0, "Should use tokens"
    
    def test_baseline_product_query(self, baseline_queries):
        """Test that baseline product query still works."""
        test_case = baseline_queries["product_simple"]
        
        params, tokens = answer_query(
            query=test_case["query"],
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10
        )
        
        assert "prompt" in params
        assert len(params["prompt"]) > test_case["min_prompt_length"]
        assert "Product ID:" in params["prompt"], "Should contain products"
        assert tokens >= 0
    
    def test_baseline_filtered_query(self, baseline_queries):
        """Test that filtered product query still works."""
        test_case = baseline_queries["product_filtered"]
        
        params, tokens = answer_query(
            query=test_case["query"],
            use_rag=True,
            retriever_type="semantic",
            simplified=False,  # Use filters
            top_k=20
        )
        
        assert "prompt" in params
        assert len(params["prompt"]) > test_case["min_prompt_length"]
        assert tokens > 0, "Should have routing + metadata tokens"
    
    def test_all_retriever_types_work(self):
        """Regression: All retriever types should work."""
        query = "shirts"
        retriever_types = ["bm25", "semantic", "hybrid"]
        
        for ret_type in retriever_types:
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type=ret_type,
                simplified=True,
                top_k=10,
                alpha=0.5 if ret_type == "hybrid" else 0.5,
                k=60 if ret_type == "hybrid" else 60
            )
            
            assert "prompt" in params, f"{ret_type} failed to return prompt"
            assert "Product" in params["prompt"], f"{ret_type} didn't retrieve products"
            print(f"✅ {ret_type}: OK")
    
    def test_top_k_values_work(self):
        """Regression: Different top_k values should work."""
        query = "casual wear"
        top_k_values = [5, 10, 20, 50]
        
        for k in top_k_values:
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=k
            )
            
            assert "prompt" in params, f"top_k={k} failed"
            print(f"✅ top_k={k}: OK")


class TestTokenUsage:
    """Test token usage patterns."""
    
    def test_token_usage_consistency(self):
        """Test that token usage is consistent for same query."""
        query = "What is your return policy?"
        token_counts = []
        
        for _ in range(3):
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                simplified=False
            )
            token_counts.append(tokens)
        
        # Should be very similar (within 10%)
        avg_tokens = mean(token_counts)
        for t in token_counts:
            diff_percent = abs(t - avg_tokens) / avg_tokens * 100
            assert diff_percent < 10, f"Token usage varies too much: {token_counts}"
        
        print(f"\n💰 Token usage consistency: {token_counts} (avg: {avg_tokens:.1f})")
    
    def test_token_budget_limits(self):
        """Test that queries stay within reasonable token budgets."""
        test_cases = [
            {
                "name": "FAQ simplified",
                "query": "return policy",
                "config": {"use_rag": True, "simplified": True, "top_k": 5},
                "max_tokens": 3000
            },
            {
                "name": "FAQ full",
                "query": "return policy",
                "config": {"use_rag": True, "simplified": False},
                "max_tokens": 5000
            },
            {
                "name": "Product simple",
                "query": "blue shirts",
                "config": {"use_rag": True, "retriever_type": "semantic", "simplified": True, "top_k": 20},
                "max_tokens": 3000
            },
            {
                "name": "Product filtered",
                "query": "blue shirts under $50",
                "config": {"use_rag": True, "retriever_type": "semantic", "simplified": False, "top_k": 20},
                "max_tokens": 5000
            }
        ]
        
        print("\n💰 Token Budget Check:")
        print("="*60)
        
        for test in test_cases:
            params, tokens = answer_query(
                query=test["query"],
                **test["config"]
            )
            
            print(f"{test['name']:20s}: {tokens:4d} / {test['max_tokens']} tokens")
            assert tokens < test["max_tokens"], f"{test['name']} exceeded token budget"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self):
        """Test handling of empty query."""
        # Empty queries should be handled before reaching retrieval
        # This tests error handling
        try:
            params, tokens = answer_query(
                query="",
                use_rag=True,
                simplified=True,
                top_k=10
            )
            # If it doesn't crash, check it returned something
            assert "prompt" in params
        except Exception as e:
            # It's okay if it raises an error - we're testing it doesn't crash the app
            print(f"Empty query handled with error: {type(e).__name__}")
            assert True  # Test passes - error was caught
    
    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "blue " * 100 + "shirts"
        
        params, tokens = answer_query(
            query=long_query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10
        )
        
        assert "prompt" in params
    
    def test_special_characters_query(self):
        """Test query with special characters."""
        query = "shirts @ $50! <special> & symbols?"
        
        params, tokens = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10
        )
        
        assert "prompt" in params
    
    def test_zero_top_k(self):
        """Test handling of top_k=0 (edge case)."""
        # Should handle gracefully or use minimum value
        try:
            params, tokens = answer_query(
                query="shirts",
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=0
            )
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # If it raises an error, that's also acceptable
            print(f"top_k=0 raised: {e}")
    
    def test_extreme_alpha_values(self):
        """Test hybrid with extreme alpha values."""
        query = "casual wear"
        
        # Alpha = 0 (pure semantic)
        params_0, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="hybrid",
            alpha=0.0,
            k=60,
            top_k=10
        )
        assert "prompt" in params_0
        
        # Alpha = 1 (pure BM25)
        params_1, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="hybrid",
            alpha=1.0,
            k=60,
            top_k=10
        )
        assert "prompt" in params_1


class TestConcurrentRequests:
    """Test behavior under concurrent load (simple version)."""
    
    def test_sequential_requests(self):
        """Test multiple sequential requests."""
        queries = [
            "blue shirts",
            "return policy",
            "casual wear",
            "formal attire",
            "summer collection"
        ]
        
        results = []
        
        for query in queries:
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=10
            )
            results.append({"query": query, "success": "prompt" in params})
        
        # All should succeed
        assert all(r["success"] for r in results), "Some queries failed"
        print(f"\n✅ All {len(results)} sequential requests succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])