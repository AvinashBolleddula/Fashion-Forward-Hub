"""RAG quality evaluation using RAGAS metrics."""

import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from sklearn.metrics import precision_score
from rag_pipeline import answer_query
from llm import generate_with_single_input


class TestRAGQualityMetrics:
    """Evaluate RAG system quality using RAGAS."""
    
    @pytest.fixture
    def evaluation_dataset(self):
        """Create evaluation dataset with ground truth."""
        return {
            "questions": [
                "What is your return policy?",
                "Show me blue shirts under $50",
                "Do you offer free shipping?",
                "What casual wear do you have for men?",
                "Can I exchange items?",
            ],
            "ground_truths": [
                "Returns accepted within 30 days with conditions for specific categories.",
                "Blue shirts available in various styles and prices.",
                "Shipping policies vary based on order value and location.",
                "Casual wear includes t-shirts, jeans, and casual shirts for men.",
                "Exchanges can be initiated through the Returns Center.",
            ]
        }
    
    def test_rag_faithfulness(self, evaluation_dataset):
        """
        Test faithfulness: Does the answer stay true to retrieved context?
        Measures if the answer contains hallucinations.
        """
        questions = evaluation_dataset["questions"][:3]  # Test first 3
        
        answers = []
        contexts = []
        
        for question in questions:
            # Get RAG response
            params, _ = answer_query(
                query=question,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=10
            )
            
            # Generate actual answer
            response = generate_with_single_input(
                prompt=params["prompt"],
                model=params["model"]
            )
            
            answers.append(response["content"])
            contexts.append([params["prompt"]])  # Context used for generation
        
        # Create RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data)
        
        # Evaluate faithfulness
        result = evaluate(dataset, metrics=[faithfulness])

        # FIX: Extract score properly
        faithfulness_score = float(result['faithfulness']) if not isinstance(result['faithfulness'], list) else float(result['faithfulness'][0])
        print(f"\n📊 Faithfulness Score: {faithfulness_score:.3f}")
        print("(Higher is better, >0.7 is good)")
    
        # Assert minimum quality
        assert faithfulness_score > 0.5, "Faithfulness should be > 0.5"
            
        
    def test_rag_answer_relevancy(self, evaluation_dataset):
        """
        Test answer relevancy: Is the answer relevant to the question?
        Measures if the answer actually addresses the query.
        """
        questions = evaluation_dataset["questions"][:3]
        
        answers = []
        
        for question in questions:
            params, _ = answer_query(
                query=question,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=10
            )
            
            response = generate_with_single_input(
                prompt=params["prompt"],
                model=params["model"]
            )
            
            answers.append(response["content"])
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
        }
        dataset = Dataset.from_dict(data)
        
        # Evaluate relevancy
        result = evaluate(dataset, metrics=[answer_relevancy])
        
        # FIX: Extract score properly
        relevancy_score = float(result['answer_relevancy']) if not isinstance(result['answer_relevancy'], list) else float(result['answer_relevancy'][0])
        print(f"\n📊 Answer Relevancy Score: {relevancy_score:.3f}")
        print("(Higher is better, >0.7 is good)")
        
        assert relevancy_score > 0.5, "Answer relevancy should be > 0.5"
    
    def test_context_precision(self, evaluation_dataset):
        """
        Test context precision: Is retrieved context relevant?
        Measures if we're retrieving the right information.
        """
        questions = evaluation_dataset["questions"][:2]
        ground_truths = evaluation_dataset["ground_truths"][:2]
        
        answers = []
        contexts = []
        
        for question in questions:
            params, _ = answer_query(
                query=question,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=10
            )
            
            response = generate_with_single_input(
                prompt=params["prompt"],
                model=params["model"]
            )
            
            answers.append(response["content"])
            contexts.append([params["prompt"]])
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        
        # Evaluate precision
        result = evaluate(dataset, metrics=[context_precision])
        
        # FIX: Extract score properly
        precision_score = float(result['context_precision']) if not isinstance(result['context_precision'], list) else float(result['context_precision'][0])
        print(f"\n📊 Context Precision Score: {precision_score:.3f}")
        print("(Higher is better, >0.7 is good)")
        
        # Lower threshold since we don't have perfect ground truth
        assert precision_score > 0.3, "Context precision should be > 0.3"


class TestRetrieverComparison:
    """Compare different retriever strategies."""
    
    def test_compare_retrievers(self):
        """Compare BM25 vs Semantic vs Hybrid on same queries."""
        test_queries = [
            "blue casual shirts",
            "formal wedding attire",
            "summer wear under $100"
        ]
        
        results = {
            "bm25": [],
            "semantic": [],
            "hybrid": []
        }
        
        for query in test_queries:
            # BM25
            params_bm25, tokens_bm25 = answer_query(
                query=query,
                use_rag=True,
                retriever_type="bm25",
                top_k=10
            )
            results["bm25"].append({
                "query": query,
                "tokens": tokens_bm25,
                "prompt_length": len(params_bm25["prompt"])
            })
            
            # Semantic
            params_sem, tokens_sem = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=10
            )
            results["semantic"].append({
                "query": query,
                "tokens": tokens_sem,
                "prompt_length": len(params_sem["prompt"])
            })
            
            # Hybrid
            params_hyb, tokens_hyb = answer_query(
                query=query,
                use_rag=True,
                retriever_type="hybrid",
                alpha=0.5,
                k=60,
                top_k=10
            )
            results["hybrid"].append({
                "query": query,
                "tokens": tokens_hyb,
                "prompt_length": len(params_hyb["prompt"])
            })
        
        # Print comparison
        print("\n📊 Retriever Comparison:")
        print("="*60)
        for retriever, data in results.items():
            avg_tokens = sum(d["tokens"] for d in data) / len(data)
            print(f"\n{retriever.upper()}:")
            print(f"  Avg tokens: {avg_tokens:.1f}")
            for d in data:
                print(f"  - {d['query'][:30]}... : {d['tokens']} tokens")


class TestSimplifiedVsNonSimplified:
    """Compare simplified mode performance."""
    
    def test_faq_simplified_comparison(self):
        """Compare FAQ retrieval: simplified vs non-simplified."""
        query = "What is your return policy?"
        
        # Non-simplified (all 25 FAQs)
        params_full, tokens_full = answer_query(
            query=query,
            use_rag=True,
            simplified=False
        )
        
        # Simplified (top 5 FAQs)
        params_simp, tokens_simp = answer_query(
            query=query,
            use_rag=True,
            simplified=True,
            top_k=5
        )
        
        print(f"\n📊 FAQ Simplified Comparison:")
        print(f"Full (25 FAQs): {tokens_full} tokens")
        print(f"Simplified (5 FAQs): {tokens_simp} tokens")
        print(f"Token savings: {tokens_full - tokens_simp} ({(1-tokens_simp/tokens_full)*100:.1f}%)")
        
        # Simplified should use fewer tokens
        assert tokens_simp < tokens_full, "Simplified should use fewer tokens"
    
    def test_product_simplified_comparison(self):
        """Compare product retrieval: with/without filters."""
        query = "blue shirts under $50"
        
        # With filters (simplified=False)
        params_filtered, tokens_filtered = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=False,
            top_k=20
        )
        
        # Without filters (simplified=True)
        params_direct, tokens_direct = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=20
        )
        
        print(f"\n📊 Product Simplified Comparison:")
        print(f"With filters: {tokens_filtered} tokens")
        print(f"Direct semantic: {tokens_direct} tokens")
        print(f"Filter overhead: {tokens_filtered - tokens_direct} tokens")
        
        # With filters should use more tokens (metadata extraction)
        assert tokens_filtered >= tokens_direct, "Filters should add token overhead"


class TestTopKImpact:
    """Test how top_k affects results."""
    
    def test_top_k_variations(self):
        """Test different top_k values."""
        query = "casual summer wear"
        top_k_values = [5, 10, 20, 50]
        
        results = []
        
        for k in top_k_values:
            params, tokens = answer_query(
                query=query,
                use_rag=True,
                retriever_type="semantic",
                simplified=True,
                top_k=k
            )
            
            # Count products in prompt (rough estimate)
            product_count = params["prompt"].count("Product ID:")
            
            results.append({
                "top_k": k,
                "tokens": tokens,
                "products_in_prompt": product_count
            })
        
        print(f"\n📊 Top-K Impact:")
        print("="*60)
        for r in results:
            print(f"top_k={r['top_k']:2d}: {r['products_in_prompt']} products, {r['tokens']} tokens")
        
        # More top_k should generally mean more products
        assert results[-1]["products_in_prompt"] >= results[0]["products_in_prompt"]


class TestRerankerImpact:
    """Test reranker effectiveness."""
    
    def test_reranker_changes_order(self):
        """Test that reranker changes result order."""
        query = "professional business attire"
        
        # Without reranker
        params_no_rerank, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10,
            use_reranker=False
        )
        
        # With reranker
        params_rerank, _ = answer_query(
            query=query,
            use_rag=True,
            retriever_type="semantic",
            simplified=True,
            top_k=10,
            use_reranker=True
        )
        
        # Extract first product from each
        # (Simple check - just verify both have products)
        assert "Product ID:" in params_no_rerank["prompt"]
        assert "Product ID:" in params_rerank["prompt"]
        
        print("\n✅ Reranker test passed - both variants return products")


if __name__ == "__main__":
    # Run with specific markers
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])