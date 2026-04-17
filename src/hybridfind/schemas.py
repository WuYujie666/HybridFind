"""Shared data structures for HybridFind."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document stored in the search index."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list, repr=False)


@dataclass
class SearchResult:
    """A single search result."""

    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
