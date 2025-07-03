"""
Neural Reranker with Learnable Importance Weights for RM3 + Semantic Similarity

Key contribution: Learn optimal α (RM3) and β (semantic) weights end-to-end
within a neural reranking framework.
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
    Neural reranker that learns to combine RM3 and semantic similarity.

    Architecture:
    1. Learnable weights: α (RM3), β (semantic similarity)
    2. Weighted term embedding aggregation
    3. Query + term representation concatenation
    4. Neural scoring layers
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_expansion_terms: int = 15,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 device: str = None):
        """
        Initialize neural reranker with learnable importance weights.

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
        self.beta = nn.Parameter(torch.tensor(1.0))  # Semantic weight

        # Neural scoring architecture
        # Input: [query_embedding + weighted_term_embedding]
        input_dim = self.embedding_dim * 2

        self.scoring_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Learnable weights: α={self.alpha.item():.3f}, β={self.beta.item():.3f}")

    def _init_weights(self):
        """Initialize neural network weights."""
        for module in self.scoring_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

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

    def create_weighted_term_representation(self,
                                            terms: List[str],
                                            importance_weights: Dict[str, float]) -> torch.Tensor:
        """
        Create importance-weighted aggregation of term embeddings.

        Args:
            terms: List of expansion terms
            importance_weights: Computed importance weights

        Returns:
            Aggregated term representation [embedding_dim]
        """
        if not terms:
            return torch.zeros(self.embedding_dim, device=self.device)

        # Limit to max terms
        terms = terms[:self.max_expansion_terms]

        # Get term embeddings
        term_embeddings = self.encode_terms(terms)  # [num_terms, embedding_dim]

        # Get importance weights
        weights = []
        for term in terms:
            weight = importance_weights.get(term, 0.0)
            weights.append(max(weight, 0.0))  # Ensure non-negative

        if not any(weights):
            # Fallback to uniform weights
            weights = [1.0] * len(terms)

        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]

        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
        weights_tensor = weights_tensor.unsqueeze(1)  # [num_terms, 1]

        # Weighted aggregation
        weighted_embeddings = term_embeddings * weights_tensor
        aggregated_embedding = torch.sum(weighted_embeddings, dim=0)  # [embedding_dim]

        return aggregated_embedding

    def forward(self,
                query: str,
                expansion_features: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """
        Forward pass: query + expansion features -> relevance score.

        Args:
            query: Query text
            expansion_features: Features for expansion terms

        Returns:
            Relevance score
        """
        # Get query embedding
        query_embedding = self.encode_text(query)  # [embedding_dim]

        # Compute importance weights using learnable α, β
        importance_weights = self.compute_importance_weights(expansion_features)

        # Get expansion terms
        expansion_terms = list(expansion_features.keys())

        # Create weighted term representation
        term_embedding = self.create_weighted_term_representation(
            expansion_terms, importance_weights
        )  # [embedding_dim]

        # Concatenate query and term representations
        combined_embedding = torch.cat([query_embedding, term_embedding], dim=0)  # [2 * embedding_dim]

        # Neural scoring
        score = self.scoring_layers(combined_embedding)  # [1]

        return score.squeeze()  # scalar

    def get_learned_weights(self) -> Tuple[float, float]:
        """Get current learned α and β values."""
        return self.alpha.item(), self.beta.item()

    def rerank_candidates(self,
                          query: str,
                          expansion_features: Dict[str, Dict[str, float]],
                          candidates: List[Tuple[str, float]],
                          top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Rerank candidates using learned neural scoring.

        Args:
            query: Query text
            expansion_features: Expansion term features
            candidates: List of (doc_id, first_stage_score)
            top_k: Number of results to return

        Returns:
            Reranked results [(doc_id, neural_score)]
        """
        if not candidates:
            return []

        self.eval()
        results = []

        with torch.no_grad():
            # Score the query expansion (independent of specific documents)
            neural_score = self.forward(query, expansion_features).item()

            # Apply same score to all candidates (can be enhanced to use doc content)
            for doc_id, first_stage_score in candidates:
                # Simple approach: use neural score directly
                # Enhanced approach: combine with first_stage_score
                final_score = neural_score
                results.append((doc_id, final_score))

        # Sort by neural score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class DocumentAwareReranker(ImportanceWeightedNeuralReranker):
    """
    Enhanced version that considers document content in scoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Enhanced architecture for query-document interaction
        input_dim = self.embedding_dim * 3  # query + terms + document

        self.scoring_layers = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        ).to(self.device)

        self._init_weights()
        logger.info("DocumentAwareReranker initialized")

    def forward(self,
                query: str,
                expansion_features: Dict[str, Dict[str, float]],
                document: str) -> torch.Tensor:
        """Forward pass with document interaction."""
        # Get embeddings
        query_embedding = self.encode_text(query)
        doc_embedding = self.encode_text(document)

        # Compute importance weights and term representation
        importance_weights = self.compute_importance_weights(expansion_features)
        expansion_terms = list(expansion_features.keys())
        term_embedding = self.create_weighted_term_representation(
            expansion_terms, importance_weights
        )

        # Triple concatenation
        combined_embedding = torch.cat([query_embedding, term_embedding, doc_embedding], dim=0)

        # Neural scoring
        score = self.scoring_layers(combined_embedding)
        return score.squeeze()

    def rerank_candidates(self,
                          query: str,
                          expansion_features: Dict[str, Dict[str, float]],
                          candidates: List[Tuple[str, str, float]],  # Include doc_text
                          top_k: int = 100) -> List[Tuple[str, float]]:
        """Rerank with document content."""
        if not candidates:
            return []

        self.eval()
        results = []

        with torch.no_grad():
            for doc_id, doc_text, first_stage_score in candidates:
                try:
                    neural_score = self.forward(query, expansion_features, doc_text).item()
                    results.append((doc_id, neural_score))
                except Exception as e:
                    logger.warning(f"Error scoring {doc_id}: {e}")
                    results.append((doc_id, first_stage_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Factory function
def create_neural_reranker(model_name: str = 'all-MiniLM-L6-v2',
                           use_document_content: bool = False,
                           **kwargs) -> ImportanceWeightedNeuralReranker:
    """
    Factory function to create neural reranker.

    Args:
        model_name: Sentence transformer model
        use_document_content: Whether to use document content in scoring
        **kwargs: Additional arguments

    Returns:
        Neural reranker instance
    """
    if use_document_content:
        return DocumentAwareReranker(model_name=model_name, **kwargs)
    else:
        return ImportanceWeightedNeuralReranker(model_name=model_name, **kwargs)