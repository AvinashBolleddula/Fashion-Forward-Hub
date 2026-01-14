"""Comprehensive test runner with reporting."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


def run_unit_tests():
    """Run unit tests only."""
    print("\n" + "="*70)
    print("🧪 RUNNING UNIT TESTS")
    print("="*70)
    
    result = pytest.main([
        "tests/unit",
        "-v",
        "-s",
        "--tb=short",
        "--color=yes"
    ])
    return result


def run_integration_tests():
    """Run integration tests."""
    print("\n" + "="*70)
    print("🔗 RUNNING INTEGRATION TESTS")
    print("="*70)
    
    result = pytest.main([
        "tests/integration",
        "-v",
        "-s",
        "--tb=short",
        "--color=yes"
    ])
    return result


def run_evaluation_tests():
    """Run evaluation/quality tests."""
    print("\n" + "="*70)
    print("📊 RUNNING EVALUATION TESTS")
    print("="*70)
    print("⚠️  Note: These tests call OpenAI API and will use tokens")
    
    result = pytest.main([
        "tests/evaluation",
        "-v",
        "-s",
        "--tb=short",
        "--color=yes"
    ])
    return result


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "="*70)
    print("🚀 RUNNING ALL TESTS")
    print("="*70)
    
    results = {
        "unit": run_unit_tests(),
        "integration": run_integration_tests(),
        "evaluation": run_evaluation_tests()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    for test_type, result in results.items():
        status = "✅ PASSED" if result == 0 else "❌ FAILED"
        print(f"{test_type.upper():15s}: {status}")
    
    total_failed = sum(1 for r in results.values() if r != 0)
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_failed} test suite(s) failed")
        return 1


def run_quick_smoke_test():
    """Run quick smoke tests (subset)."""
    print("\n" + "="*70)
    print("⚡ RUNNING QUICK SMOKE TESTS")
    print("="*70)
    
    result = pytest.main([
        "tests/unit/test_retrieval.py::TestBM25Retrieval::test_bm25_returns_results",
        "tests/unit/test_retrieval.py::TestSemanticRetrieval::test_semantic_simplified_returns_results",
        "tests/integration/test_rag_pipeline.py::TestRAGPipelineIntegration::test_faq_query_simplified_true",
        "tests/integration/test_rag_pipeline.py::TestRAGPipelineIntegration::test_product_query_semantic_simplified",
        "-v",
        "-s",
        "--tb=line"
    ])
    
    if result == 0:
        print("\n✅ Smoke tests passed!")
    else:
        print("\n❌ Smoke tests failed!")
    
    return result


def run_performance_only():
    """Run only performance tests."""
    print("\n" + "="*70)
    print("⏱️  RUNNING PERFORMANCE TESTS")
    print("="*70)
    
    result = pytest.main([
        "tests/evaluation/test_performance.py",
        "-v",
        "-s",
        "--tb=short"
    ])
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            exit_code = run_unit_tests()
        elif test_type == "integration":
            exit_code = run_integration_tests()
        elif test_type == "evaluation":
            exit_code = run_evaluation_tests()
        elif test_type == "smoke":
            exit_code = run_quick_smoke_test()
        elif test_type == "performance":
            exit_code = run_performance_only()
        elif test_type == "all":
            exit_code = run_all_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python tests/run_tests.py [unit|integration|evaluation|smoke|performance|all]")
            exit_code = 1
    else:
        # Default: run all
        exit_code = run_all_tests()
    
    sys.exit(exit_code)