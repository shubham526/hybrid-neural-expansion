"""
Neural Reranker with Learnable Importance Weights for RM3 + Semantic Similarity

Architecture:
1. Learn α (RM3) and β (semantic) weights for expansion terms
2. Combine query + weighted term embeddings via linear layer → enhanced query embedding
3. Score enhanced query + document embeddings via linear layer → final score
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ImportanceWeightedNeuralReranker(nn.Module):
    """
    Neural reranker that learns importance weights and combines query+terms+document.

    Architecture:
    1. Learnable weights: α (RM3), β (semantic similarity)
    2. Linear combination: [query, weighted_terms] → enhanced_query
    3. Scoring: [enhanced_query, document] → score
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_expansion_terms: int = 15,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 device: str = None):
        """
        Initialize neural reranker.

        Args:
            model_name: Sentence transformer model
            max_expansion_terms: Maximum expansion terms to use
            hidden_dim: Hidden dimension for neural layers
            dropout: Dropout rate
            device: Device for computation
        """
        super().__init__()

        self.model_name = model_name
        self.max_expansion_terms = max_expansion_terms
        self.hidden_dim = hidden_dim

        # Load sentence transformer
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.device = self.encoder.device

        # KEY CONTRIBUTION: Learnable importance weights
        self.alpha = nn.Parameter(torch.tensor(1.0))  # RM3 weight
        self.beta = nn.Parameter(torch.tensor(1.0))   # Semantic weight

        # Query enhancement layer: [query + weighted_terms] → enhanced_query
        # Input: query_emb + max_terms * term_emb = (1 + max_terms) * embedding_dim
        query_enhancement_input_dim = (1 + max_expansion_terms) * self.embedding_dim

        self.query_enhancement = nn.Sequential(
            nn.Linear(query_enhancement_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.embedding_dim)  # Output same size as embedding
        ).to(self.device)

        # Scoring layer: [enhanced_query + document] → score
        scoring_input_dim = 2 * self.embedding_dim

        self.scoring_layers = nn.Sequential(
            nn.Linear(scoring_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        ).to(self.device)

        # Initialize weights
        self._init_weights()

        logger.info(f"ImportanceWeightedNeuralReranker initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Max expansion terms: {max_expansion_terms}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Learnable weights: α={self.alpha.item():.3f}, β={self.beta.item():.3f}")

    def _init_weights(self):
        """Initialize neural network weights."""
        for module in [self.query_enhancement, self.scoring_layers]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into embedding."""
        with torch.no_grad():
            embedding = self.encoder.encode([text], convert_to_tensor=True)[0]
        return embedding.to(self.device)

    def encode_terms(self, terms: List[str]) -> torch.Tensor:
        """Encode multiple terms into embeddings."""
        if not terms:
            return torch.zeros(1, self.embedding_dim, device=self.device)

        with torch.no_grad():
            embeddings = self.encoder.encode(terms, convert_to_tensor=True)
        return embeddings.to(self.device)

    def compute_importance_weights(self,
                                   expansion_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute importance weights using learnable α and β.

        Args:
            expansion_features: {term: {'rm_weight': float, 'semantic_score': float}}

        Returns:
            {term: importance_weight}
        """
        importance_weights = {}

        for term, features in expansion_features.items():
            rm_weight = features['rm_weight']
            semantic_score = features['semantic_score']

            # CORE CONTRIBUTION: Learnable linear combination
            importance = self.alpha * rm_weight + self.beta * semantic_score
            importance_weights[term] = importance.item()

        return importance_weights

    def create_enhanced_query_embedding(self,
                                        query: str,
                                        expansion_features: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """
        Create enhanced query embedding via linear combination.

        Args:
            query: Query text
            expansion_features: Features for expansion terms

        Returns:
            Enhanced query embedding [embedding_dim]
        """
        # 1. Get query embedding
        query_embedding = self.encode_text(query)  # [embedding_dim]

        # 2. Get expansion terms (limit to max)
        expansion_terms = list(expansion_features.keys())[:self.max_expansion_terms]

        # 3. Prepare input for linear layer
        embeddings_to_combine = [query_embedding]  # Start with query

        if expansion_terms:
            # Get term embeddings
            term_embeddings = self.encode_terms(expansion_terms)  # [num_terms, embedding_dim]

            # Compute importance weights
            importance_weights = self.compute_importance_weights(expansion_features)

            # Weight the term embeddings
            for i, term in enumerate(expansion_terms):
                weight = importance_weights.get(term, 0.0)
                weighted_embedding = term_embeddings[i] * weight
                embeddings_to_combine.append(weighted_embedding)

        # 4. Pad to max_expansion_terms if needed
        while len(embeddings_to_combine) < (1 + self.max_expansion_terms):
            # Add zero embeddings for missing terms
            embeddings_to_combine.append(torch.zeros_like(query_embedding))

        # 5. Concatenate all embeddings
        combined_input = torch.cat(embeddings_to_combine, dim=0)  # [(1+max_terms) * embedding_dim]

        # 6. Pass through linear layer to get enhanced query
        enhanced_query = self.query_enhancement(combined_input)  # [embedding_dim]

        return enhanced_query

    def forward(self,
                query: str,
                expansion_features: Dict[str, Dict[str, float]],
                document: str) -> torch.Tensor:
        """
        Forward pass: query + expansion features + document → relevance score.

        Args:
            query: Query text
            expansion_features: Features for expansion terms
            document: Document text

        Returns:
            Relevance score
        """
        # 1. Create enhanced query embedding
        enhanced_query_emb = self.create_enhanced_query_embedding(query, expansion_features)

        # 2. Get document embedding
        document_emb = self.encode_text(document)

        # 3. Combine enhanced query + document
        combined_emb = torch.cat([enhanced_query_emb, document_emb], dim=0)  # [2 * embedding_dim]

        # 4. Score via neural network
        score = self.scoring_layers(combined_emb)  # [1]

        return score.squeeze()  # scalar

    def get_learned_weights(self) -> Tuple[float, float]:
        """Get current learned α and β values."""
        return self.alpha.item(), self.beta.item()

    def rerank_candidates(self,
                          query: str,
                          expansion_features: Dict[str, Dict[str, float]],
                          candidates: List[Tuple[str, float]],
                          document_texts: Dict[str, str],
                          top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Rerank candidates using the neural model.

        Args:
            query: Query text
            expansion_features: Expansion term features
            candidates: List of (doc_id, first_stage_score)
            document_texts: {doc_id: doc_text}
            top_k: Number of results to return

        Returns:
            Reranked results [(doc_id, neural_score)]
        """
        if not candidates:
            return []

        self.eval()
        results = []

        with torch.no_grad():
            for doc_id, first_stage_score in candidates:
                if doc_id not in document_texts:
                    # Fallback to first-stage score if no document text
                    results.append((doc_id, first_stage_score))
                    continue

                try:
                    # Get neural score
                    neural_score = self.forward(
                        query,
                        expansion_features,
                        document_texts[doc_id]
                    ).item()

                    results.append((doc_id, neural_score))

                except Exception as e:
                    logger.warning(f"Error scoring document {doc_id}: {e}")
                    results.append((doc_id, first_stage_score))

        # Sort by neural score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Factory function
def create_neural_reranker(model_name: str = 'all-MiniLM-L6-v2',
                           **kwargs) -> ImportanceWeightedNeuralReranker:
    """
    Factory function to create neural reranker.

    Args:
        model_name: Sentence transformer model
        **kwargs: Additional arguments

    Returns:
        Neural reranker instance
    """
    return ImportanceWeightedNeuralReranker(model_name=model_name, **kwargs)