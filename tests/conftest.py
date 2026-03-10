"""Shared pytest fixtures for the Financial RAG Agent test suite."""

import pytest

from src.ingestion.models import FilingMetadata


@pytest.fixture
def sample_filing_metadata() -> FilingMetadata:
    """Create a sample FilingMetadata for use in tests."""
    ...


@pytest.fixture
def sample_filing_html() -> str:
    """Return a minimal sample 10-K HTML string for parsing tests."""
    ...


@pytest.fixture
def sample_chunks() -> list:
    """Return a list of sample DocumentChunk objects for retrieval tests."""
    ...
