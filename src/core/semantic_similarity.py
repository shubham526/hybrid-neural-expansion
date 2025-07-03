"""
Semantic Similarity Module

Clean semantic similarity computation using sentence transformers.
Optimized for query expansion in IR tasks.

Author: Your Name
"""

import logging
import numpy as np
import torch
from typing import List, Union, Dict, Optional
from functools import lru_cache
import threading

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """
    Semantic similarity computation using sentence transformers.
    Optimized for query expansion use cases with thread-safe caching.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 cache_size: int = 1000):
        """
        Initialize semantic similarity computer.

        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_size: Size of embedding cache (0 to disable)
        """
        self.model_name = model_name

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.device = self.model.device
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Thread-safe cache for embeddings
        self.cache_size = cache_size
        if cache_size > 0:
            self._embedding_cache = {}
            self._cache_lock = threading.Lock()
        else:
            self._embedding_cache = None
            self._cache_lock = None

        logger.info(f"Loaded sentence transformer: {model_name} on {self.device}")
        logger.debug(f"Embedding cache size: {cache_size}")

    def _encode_single_cached(self, text: str) -> np.ndarray:
        """Encode single text with thread-safe caching."""
        if self._embedding_cache is None:
            return self._encode_single_uncached(text)

        # Check cache first
        with self._cache_lock:
            if text in self._embedding_cache:
                return self._embedding_cache[text]

        # Encode if not in cache
        embedding = self._encode_single_uncached(text)

        # Store in cache
        with self._cache_lock:
            # If cache is full, remove oldest entry (simple FIFO)
            if len(self._embedding_cache) >= self.cache_size:
                # Remove one item (arbitrary which one)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]

            self._embedding_cache[text] = embedding

        return embedding

    def _encode_single_uncached(self, text: str) -> np.ndarray:
        """Encode single text without caching."""
        try:
            with torch.no_grad():
                embedding = self.model.encode([text], convert_to_tensor=False)[0]
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text '{text[:50]}...': {e}")
            # Return zero embedding as fallback
            return np.zeros(self.model.get_sentence_embedding_dimension())

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing multiple texts

        Returns:
            Single embedding array or list of embedding arrays
        """
        if isinstance(texts, str):
            return self._encode_single_cached(texts)
        else:
            try:
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        convert_to_tensor=False,
                        show_progress_bar=False  # Disable progress bar for cleaner logs
                    )
                return embeddings
            except Exception as e:
                logger.error(f"Failed to encode {len(texts)} texts: {e}")
                # Return zero embeddings as fallback
                dim = self.model.get_sentence_embedding_dimension()
                return [np.zeros(dim) for _ in texts]

    def compute_similarity(self,
                          text1: Union[str, np.ndarray],
                          text2: Union[str, np.ndarray]) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text (string or pre-computed embedding)
            text2: Second text (string or pre-computed embedding)

        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            # Get embeddings
            if isinstance(text1, str):
                emb1 = self._encode_single_cached(text1)
            else:
                emb1 = text1

            if isinstance(text2, str):
                emb2 = self._encode_single_cached(text2)
            else:
                emb2 = text2

            # Compute cosine similarity
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return max(0.0, float(similarity))  # Ensure non-negative

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def compute_query_expansion_similarities(self,
                                           query: str,
                                           expansion_terms: List[str]) -> Dict[str, float]:
        """
        Compute similarities between query and expansion terms.
        Optimized for query expansion use case.

        Args:
            query: Original query
            expansion_terms: List of expansion terms

        Returns:
            Dictionary mapping terms to similarity scores
        """
        if not expansion_terms:
            return {}

        try:
            # Encode query once
            query_emb = self._encode_single_cached(query)

            # Compute similarities for all terms
            similarities = {}
            for term in expansion_terms:
                if not term or not term.strip():
                    similarities[term] = 0.0
                    continue

                term_emb = self._encode_single_cached(term)
                similarity = self.compute_similarity(query_emb, term_emb)
                similarities[term] = similarity

            return similarities

        except Exception as e:
            logger.error(f"Error computing query expansion similarities: {e}")
            return {term: 0.0 for term in expansion_terms}

    def clear_cache(self):
        """Clear the embedding cache."""
        if self._embedding_cache is not None:
            with self._cache_lock:
                self._embedding_cache.clear()
            logger.info("Embedding cache cleared")

    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the cache."""
        if self._embedding_cache is None:
            return {"cache_enabled": False}

        with self._cache_lock:
            return {
                "cache_enabled": True,
                "cache_size": len(self._embedding_cache),
                "max_cache_size": self.cache_size
            }

    def preload_embeddings(self, texts: List[str]):
        """
        Preload embeddings for a list of texts into cache.

        Args:
            texts: List of texts to preload
        """
        if self._embedding_cache is None:
            logger.warning("Cache is disabled, cannot preload embeddings")
            return

        logger.info(f"Preloading embeddings for {len(texts)} texts...")
        for text in texts:
            if text and text.strip():
                self._encode_single_cached(text)

        cache_info = self.get_cache_info()
        logger.info(f"Preloading complete. Cache size: {cache_info['cache_size']}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Semantic Similarity Module Test")
    print("=" * 35)

    # Initialize
    sim_computer = SemanticSimilarity('all-MiniLM-L6-v2')

    # Example usage
    query = "machine learning algorithms"
    expansion_terms = ["neural", "networks", "supervised", "classification", "deep", "learning"]

    print(f"Query: '{query}'")
    print(f"Expansion terms: {expansion_terms}")
    print()

    # Compute similarities
    similarities = sim_computer.compute_query_expansion_similarities(query, expansion_terms)

    print("Expansion term similarities:")
    for term, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {term:<15} {sim:.4f}")
    print()

    # Test caching
    print("Cache information:")
    cache_info = sim_computer.get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    print()

    # Test similarity computation
    print("Direct similarity tests:")
    test_pairs = [
        ("machine learning", "artificial intelligence"),
        ("neural networks", "deep learning"),
        ("classification", "regression"),
        ("computer", "banana")  # Should be low similarity
    ]

    for text1, text2 in test_pairs:
        sim = sim_computer.compute_similarity(text1, text2)
        print(f"  '{text1}' ↔ '{text2}': {sim:.4f}")

    print()
    print("Final cache info:")
    final_cache_info = sim_computer.get_cache_info()
    for key, value in final_cache_info.items():
        print(f"  {key}: {value}")