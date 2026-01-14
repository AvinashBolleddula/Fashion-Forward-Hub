"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from database import get_client


@pytest.fixture(scope="session")
def weaviate_client():
    """Provide Weaviate client for tests."""
    client = get_client()
    yield client
    client.close()


@pytest.fixture(scope="session")
def products_collection(weaviate_client):
    """Provide products collection."""
    return weaviate_client.collections.get("products")


@pytest.fixture(scope="session")
def faq_collection(weaviate_client):
    """Provide FAQ collection."""
    return weaviate_client.collections.get("faq")


# Test queries for different scenarios
@pytest.fixture
def test_queries():
    """Standard test queries for consistency."""
    return {
        "faq_simple": "what is your return policy?",
        "faq_complex": "can I return items after 30 days if they're damaged?",
        "product_simple": "blue shirts",
        "product_filtered": "blue shirts under $50",
        "product_complex": "casual summer wear for men under $100",
        "ambiguous": "tell me about shipping",
    }