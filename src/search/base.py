from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResponse:
    """Top-k search output for a batch of queries."""

    scores: np.ndarray
    indices: np.ndarray


class SearchBackend(ABC):
    """Common interface for CPU and GPU vector-search backends."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build or load the searchable index."""

    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int) -> SearchResponse:
        """Return top-k matches for each query vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable backend name used in results."""

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def index_residency(self) -> str:
        return "host"

    def synchronize(self) -> None:
        """Wait for backend work; CPU implementations are synchronous."""
        return None

    def configure_profiling(self, *, nvtx_enabled: bool = False) -> None:
        """Enable optional annotations without changing the search API."""
        return None
