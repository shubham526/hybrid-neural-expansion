"""
Similarity computation variants for bi-encoder retrieval.

This module provides different similarity functions and scoring strategies
that can be plugged into the bi-encoder for experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimilarityFunction(Enum):
    """Enum for different similarity function types."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    LEARNED = "learned"
    BILINEAR = "bilinear"


class BiEncoderSimilarity:
    """
    Static similarity computation strategies for bi-encoder retrieval.

    All methods expect:
    - query_emb: [embedding_dim]
    - doc_embs: [num_docs, embedding_dim]
    Returns: [num_docs] similarity scores
    """

    @staticmethod
    def cosine_similarity(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Standard cosine similarity.

        Most common choice for dense retrieval.
        """
        return F.cosine_similarity(query_emb.unsqueeze(0), doc_embs, dim=1)

    @staticmethod
    def dot_product_similarity(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Dot product similarity.

        Equivalent to cosine if embeddings are L2 normalized.
        Often faster than cosine similarity.
        """
        return torch.matmul(doc_embs, query_emb)

    @staticmethod
    def euclidean_distance(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Negative euclidean distance (higher = more similar).

        Less common for dense retrieval but can be useful for debugging.
        """
        distances = torch.cdist(query_emb.unsqueeze(0), doc_embs, p=2)
        return -distances.squeeze(0)

    @staticmethod
    def manhattan_distance(query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """Negative Manhattan (L1) distance."""
        distances = torch.cdist(query_emb.unsqueeze(0), doc_embs, p=1)
        return -distances.squeeze(0)


class LearnedSimilarity(nn.Module):
    """
    Learnable similarity function for bi-encoder.

    This adds a small neural network on top of concatenated embeddings
    to learn a more complex similarity function.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.similarity_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute learned similarity scores.

        Args:
            query_emb: [embedding_dim]
            doc_embs: [num_docs, embedding_dim]

        Returns:
            [num_docs] similarity scores
        """
        batch_size = doc_embs.size(0)

        # Expand query to match document batch
        query_expanded = query_emb.unsqueeze(0).expand(batch_size, -1)

        # Concatenate query and document embeddings
        combined = torch.cat([query_expanded, doc_embs], dim=1)

        # Compute similarity scores
        scores = self.similarity_net(combined).squeeze(-1)

        return scores


class BilinearSimilarity(nn.Module):
    """
    Bilinear similarity function.

    Learns a matrix W such that similarity = query^T W doc
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.bilinear.weight)
        if self.bilinear.bias is not None:
            nn.init.zeros_(self.bilinear.bias)

    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute bilinear similarity scores.

        Args:
            query_emb: [embedding_dim]
            doc_embs: [num_docs, embedding_dim]

        Returns:
            [num_docs] similarity scores
        """
        batch_size = doc_embs.size(0)

        # Expand query to match document batch
        query_expanded = query_emb.unsqueeze(0).expand(batch_size, -1)

        # Compute bilinear similarity
        scores = self.bilinear(query_expanded, doc_embs).squeeze(-1)

        return scores


class SimilarityComputer:
    """
    Unified interface for computing similarities with different functions.
    """

    def __init__(self,
                 similarity_function: SimilarityFunction = SimilarityFunction.COSINE,
                 embedding_dim: Optional[int] = None,
                 **kwargs):
        """
        Initialize similarity computer.

        Args:
            similarity_function: Type of similarity function to use
            embedding_dim: Required for learned similarity functions
            **kwargs: Additional arguments for learned functions
        """
        self.similarity_function = similarity_function

        if similarity_function == SimilarityFunction.LEARNED:
            if embedding_dim is None:
                raise ValueError("embedding_dim required for learned similarity")
            self.learned_sim = LearnedSimilarity(embedding_dim, **kwargs)
        elif similarity_function == SimilarityFunction.BILINEAR:
            if embedding_dim is None:
                raise ValueError("embedding_dim required for bilinear similarity")
            self.bilinear_sim = BilinearSimilarity(embedding_dim)
        else:
            self.learned_sim = None
            self.bilinear_sim = None

    def compute_similarity(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores using the configured function.

        Args:
            query_emb: [embedding_dim]
            doc_embs: [num_docs, embedding_dim]

        Returns:
            [num_docs] similarity scores
        """
        if self.similarity_function == SimilarityFunction.COSINE:
            return BiEncoderSimilarity.cosine_similarity(query_emb, doc_embs)

        elif self.similarity_function == SimilarityFunction.DOT_PRODUCT:
            return BiEncoderSimilarity.dot_product_similarity(query_emb, doc_embs)

        elif self.similarity_function == SimilarityFunction.EUCLIDEAN:
            return BiEncoderSimilarity.euclidean_distance(query_emb, doc_embs)

        elif self.similarity_function == SimilarityFunction.LEARNED:
            return self.learned_sim(query_emb, doc_embs)

        elif self.similarity_function == SimilarityFunction.BILINEAR:
            return self.bilinear_sim(query_emb, doc_embs)

        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")

    def parameters(self):
        """Get parameters for learned similarity functions."""
        if self.learned_sim is not None:
            return self.learned_sim.parameters()
        elif self.bilinear_sim is not None:
            return self.bilinear_sim.parameters()
        else:
            return []

    def to(self, device):
        """Move learned components to device."""
        if self.learned_sim is not None:
            self.learned_sim = self.learned_sim.to(device)
        if self.bilinear_sim is not None:
            self.bilinear_sim = self.bilinear_sim.to(device)
        return self

    def train(self):
        """Set learned components to training mode."""
        if self.learned_sim is not None:
            self.learned_sim.train()
        if self.bilinear_sim is not None:
            self.bilinear_sim.train()

    def eval(self):
        """Set learned components to evaluation mode."""
        if self.learned_sim is not None:
            self.learned_sim.eval()
        if self.bilinear_sim is not None:
            self.bilinear_sim.eval()


# Utility functions
def normalize_embeddings(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 normalize embeddings."""
    return F.normalize(embeddings, p=2, dim=dim)


def compute_similarity_matrix(query_embs: torch.Tensor,
                              doc_embs: torch.Tensor,
                              similarity_function: SimilarityFunction = SimilarityFunction.COSINE) -> torch.Tensor:
    """
    Compute similarity matrix between multiple queries and documents.

    Args:
        query_embs: [num_queries, embedding_dim]
        doc_embs: [num_docs, embedding_dim]
        similarity_function: Type of similarity to compute

    Returns:
        [num_queries, num_docs] similarity matrix
    """
    if similarity_function == SimilarityFunction.COSINE:
        # Normalize embeddings for cosine similarity
        query_embs_norm = normalize_embeddings(query_embs)
        doc_embs_norm = normalize_embeddings(doc_embs)
        return torch.matmul(query_embs_norm, doc_embs_norm.t())

    elif similarity_function == SimilarityFunction.DOT_PRODUCT:
        return torch.matmul(query_embs, doc_embs.t())

    else:
        # For other similarity functions, compute row by row
        similarity_computer = SimilarityComputer(similarity_function, query_embs.size(-1))

        similarities = []
        for i in range(query_embs.size(0)):
            query_emb = query_embs[i]
            sim_scores = similarity_computer.compute_similarity(query_emb, doc_embs)
            similarities.append(sim_scores)

        return torch.stack(similarities)


# Factory function
def create_similarity_computer(similarity_type: str = "cosine",
                               embedding_dim: Optional[int] = None,
                               **kwargs) -> SimilarityComputer:
    """
    Factory function to create similarity computer.

    Args:
        similarity_type: String name of similarity function
        embedding_dim: Embedding dimension (for learned functions)
        **kwargs: Additional arguments

    Returns:
        Configured SimilarityComputer
    """
    similarity_map = {
        "cosine": SimilarityFunction.COSINE,
        "dot_product": SimilarityFunction.DOT_PRODUCT,
        "euclidean": SimilarityFunction.EUCLIDEAN,
        "learned": SimilarityFunction.LEARNED,
        "bilinear": SimilarityFunction.BILINEAR
    }

    if similarity_type not in similarity_map:
        raise ValueError(f"Unknown similarity type: {similarity_type}. "
                         f"Available: {list(similarity_map.keys())}")

    similarity_function = similarity_map[similarity_type]

    return SimilarityComputer(
        similarity_function=similarity_function,
        embedding_dim=embedding_dim,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing similarity functions...")

    # Create test embeddings
    embedding_dim = 384
    num_docs = 1000

    query_emb = torch.randn(embedding_dim)
    doc_embs = torch.randn(num_docs, embedding_dim)

    # Test static similarity functions
    print("Testing static similarity functions:")

    cosine_scores = BiEncoderSimilarity.cosine_similarity(query_emb, doc_embs)
    print(f"Cosine similarity shape: {cosine_scores.shape}")
    print(f"Cosine similarity range: [{cosine_scores.min():.3f}, {cosine_scores.max():.3f}]")

    dot_scores = BiEncoderSimilarity.dot_product_similarity(query_emb, doc_embs)
    print(f"Dot product range: [{dot_scores.min():.3f}, {dot_scores.max():.3f}]")

    # Test learned similarity
    print("\nTesting learned similarity:")
    learned_sim = LearnedSimilarity(embedding_dim)
    learned_scores = learned_sim(query_emb, doc_embs)
    print(f"Learned similarity range: [{learned_scores.min():.3f}, {learned_scores.max():.3f}]")

    # Test similarity computer
    print("\nTesting similarity computer:")
    sim_computer = create_similarity_computer("cosine")
    computed_scores = sim_computer.compute_similarity(query_emb, doc_embs)
    print(f"Computed scores match cosine: {torch.allclose(cosine_scores, computed_scores)}")

    # Test similarity matrix
    print("\nTesting similarity matrix:")
    query_embs = torch.randn(5, embedding_dim)
    sim_matrix = compute_similarity_matrix(query_embs, doc_embs[:10])
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    print("\nSimilarity module test completed!")