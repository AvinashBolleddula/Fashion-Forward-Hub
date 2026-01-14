# RAG Chatbot Testing Guide

## Test Structure
```
tests/
├── conftest.py                     # Shared fixtures
├── unit/                           # Unit tests (fast)
│   └── test_retrieval.py          # Retrieval functions
├── integration/                    # Integration tests
│   └── test_rag_pipeline.py       # Full pipeline tests
├── evaluation/                     # Quality metrics
│   ├── test_rag_quality.py        # RAGAS metrics
│   └── test_performance.py        # Performance tests
└── run_tests.py                   # Test runner
```

## Running Tests

### Quick Smoke Test (30 seconds)
```bash
python tests/run_tests.py smoke
```

### Unit Tests Only (1-2 minutes)
```bash
python tests/run_tests.py unit
```

### Integration Tests (2-3 minutes)
```bash
python tests/run_tests.py integration
```

### Evaluation Tests (5-10 minutes, uses API tokens)
```bash
python tests/run_tests.py evaluation
```

### Performance Tests Only
```bash
python tests/run_tests.py performance
```

### All Tests (10-15 minutes)
```bash
python tests/run_tests.py all
```

### Using pytest directly
```bash
# Run specific test file
pytest tests/unit/test_retrieval.py -v

# Run specific test
pytest tests/unit/test_retrieval.py::TestBM25Retrieval::test_bm25_returns_results -v

# Run with coverage
pytest --cov=src tests/
```

## Test Categories

### Unit Tests
- Individual component testing
- Fast execution (< 1 min)
- No API calls to OpenAI (only Weaviate)

### Integration Tests
- End-to-end pipeline testing
- Tests component interactions
- Validates parameter flow

### Evaluation Tests
- RAG quality metrics (RAGAS)
- Retriever comparisons
- Performance benchmarks
- ⚠️ Uses OpenAI API tokens

### Performance Tests
- Latency measurements
- Regression tests
- Token usage validation

## Test Coverage

Current coverage:
- ✅ BM25 retrieval
- ✅ Semantic retrieval
- ✅ Hybrid retrieval (RRF)
- ✅ Metadata extraction
- ✅ Query routing
- ✅ Reranking
- ✅ FAQ handling
- ✅ Full pipeline
- ✅ Parameter flow
- ✅ Edge cases

## Interpreting Results

### RAGAS Metrics
- **Faithfulness** (>0.7 good): No hallucinations
- **Answer Relevancy** (>0.7 good): Answers the question
- **Context Precision** (>0.7 good): Retrieved right info

### Performance Benchmarks
- BM25 latency: < 5s
- Semantic latency: < 5s
- Hybrid latency: < 8s
- Metadata extraction: < 10s

## CI/CD Integration

Add to `.github/workflows/tests.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      weaviate:
        image: semitechnologies/weaviate:latest
        ports:
          - 8080:8080
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python tests/run_tests.py smoke
```

## Adding New Tests

1. Create test file in appropriate directory
2. Import fixtures from `conftest.py`
3. Use descriptive test names
4. Add docstrings
5. Use assertions with messages
6. Print useful debug info

Example:
```python
def test_my_feature(products_collection, test_queries):
    """Test that my feature works correctly."""
    result = my_function(test_queries["product_simple"])
    
    assert result is not None, "Result should not be None"
    assert len(result) > 0, "Should return results"
    
    print(f"✅ Got {len(result)} results")
```