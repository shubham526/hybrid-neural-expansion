"""
Bi-Encoder for Dense Query Expansion with Hybrid Lexical-Semantic Term Weighting

This module implements the approach described in the paper:
"Hybrid Lexical-Semantic Term Weighting for Dense Query Expansion"

Key insight: Importance should be encoded in representation magnitude, not just presence.
Rather than concatenating expansion terms uniformly, we explicitly scale term embeddings
by their importance to create more focused query representations.

Uses RM3 weights (not BM25) combined with semantic similarity for term importance.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Import from cross-encoder modules for compatibility
from cross_encoder.src.core.semantic_similarity import SemanticSimilarity
from bi_encoder.src.models.similarity import SimilarityComputer, SimilarityFunction, create_similarity_computer

logger = logging.getLogger(__name__)


class HybridTermWeightingBiEncoder(nn.Module):
    """
    Bi-encoder with hybrid lexical-semantic term weighting for dense query expansion.

    This model combines:
    1. RM3 weights (statistical relevance from relevance model)
    2. Cosine similarity (semantic alignment)
    to weight expansion terms before embedding integration.

    The core innovation is explicit embedding scaling by importance rather than
    uniform concatenation or averaging of expansion terms.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_expansion_terms: int = 15,
                 expansion_weight: float = 0.3,
                 similarity_function: str = 'cosine',
                 device: str = None,
                 force_hf: bool = False,
                 pooling_strategy: str = 'cls'):
        """
        Initialize the hybrid bi-encoder.

        Args:
            model_name: Sentence transformer or HF model name
            max_expansion_terms: Maximum number of expansion terms to use
            expansion_weight: Weight for combining original query with expansion
            similarity_function: Similarity function to use ('cosine', 'dot_product', 'learned', etc.)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            force_hf: Force using HuggingFace transformers
            pooling_strategy: Pooling strategy for HF models
        """
        super().__init__()

        self.model_name = model_name
        self.max_expansion_terms = max_expansion_terms
        self.expansion_weight = expansion_weight
        self.similarity_function_name = similarity_function
        self.force_hf = force_hf
        self.pooling_strategy = pooling_strategy

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize semantic similarity module
        self.encoder = SemanticSimilarity(
            model_name=model_name,
            device=str(self.device),
            force_hf=force_hf,
            pooling_strategy=pooling_strategy
        )

        # Get embedding dimension
        if self.encoder.model_type == "sentence_transformer":
            self.embedding_dim = self.encoder.model.get_sentence_embedding_dimension()
        else:  # huggingface
            self.embedding_dim = self.encoder.embedding_dim

        # Learnable importance combination weights
        self.alpha = nn.Parameter(torch.tensor(0.6, device=self.device, dtype=torch.float32))  # RM3 weight
        self.beta = nn.Parameter(torch.tensor(0.4, device=self.device, dtype=torch.float32))   # Semantic weight

        # Optional: Learnable expansion weight (alternative to fixed)
        self.learnable_expansion_weight = nn.Parameter(
            torch.tensor(expansion_weight, device=self.device, dtype=torch.float32)
        )

        # Initialize similarity computer
        self.similarity_computer = create_similarity_computer(
            similarity_type=similarity_function,
            embedding_dim=self.embedding_dim
        )
        self.similarity_computer = self.similarity_computer.to(self.device)

        # Move encoder to device
        self._move_encoder_to_device()

        logger.info(f"HybridTermWeightingBiEncoder initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Max expansion terms: {max_expansion_terms}")
        logger.info(f"  Similarity function: {similarity_function}")
        logger.info(f"  Initial α={self.alpha.item():.3f}, β={self.beta.item():.3f}")

    def _move_encoder_to_device(self):
        """Ensure encoder is on correct device."""
        try:
            if self.encoder.model_type == "sentence_transformer":
                self.encoder.model = self.encoder.model.to(self.device)
            else:  # huggingface
                self.encoder.model = self.encoder.model.to(self.device)
            self.encoder.device = self.device
        except Exception as e:
            logger.warning(f"Could not move encoder to device {self.device}: {e}")

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding tensor on correct device."""
        embedding = self.encoder.encode(text)

        # Convert to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)

        embedding = embedding.to(self.device, dtype=torch.float32)
        return embedding

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """
        Encode a batch of documents.

        Args:
            documents: List of document texts

        Returns:
            Tensor of shape [num_docs, embedding_dim]
        """
        if not documents:
            return torch.zeros(0, self.embedding_dim, device=self.device)

        # Use batch encoding for efficiency
        embeddings = self.encoder.encode(documents)

        # Convert to tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        elif isinstance(embeddings, list):
            embeddings = torch.stack([
                torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
                for emb in embeddings
            ])

        embeddings = embeddings.to(self.device, dtype=torch.float32)
        return embeddings

    def compute_term_importance(self,
                               expansion_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute hybrid importance scores for expansion terms.

        This is the core of our approach: combining RM3 weights (statistical) and
        cosine similarity (semantic) signals to determine term importance.

        Args:
            expansion_features: {term: {'rm_weight': float, 'semantic_score': float}}

        Returns:
            Dictionary mapping terms to importance scores
        """
        importance_scores = {}

        for term, features in expansion_features.items():
            rm3_weight = features.get('rm_weight', 0.0)
            semantic_score = features.get('semantic_score', 0.0)

            # Hybrid importance: α * RM3_weight + β * semantic_similarity
            # Apply sigmoid to ensure importance scores are in (0, 1)
            raw_importance = torch.sigmoid(self.alpha) * rm3_weight + torch.sigmoid(self.beta) * semantic_score

            # Convert to float for storage
            importance_scores[term] = float(raw_importance.item())

        return importance_scores

    def create_expanded_query_embedding(self,
                                      query: str,
                                      expansion_features: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """
        Create expanded query embedding using importance-weighted term scaling.

        This implements the paper's key insight: importance should be encoded in
        representation magnitude, not just presence.

        Args:
            query: Original query text
            expansion_features: Expansion term features

        Returns:
            Enhanced query embedding tensor
        """
        # Encode original query
        query_embedding = self.encode_text(query)

        # Get top-k expansion terms
        expansion_terms = list(expansion_features.keys())[:self.max_expansion_terms]

        if not expansion_terms:
            logger.debug("No expansion terms available, returning original query")
            return query_embedding

        # Compute importance scores using hybrid weighting
        importance_scores = self.compute_term_importance(expansion_features)

        # Get original terms for semantic encoding (handle stemming)
        original_terms_for_encoding = []
        term_importance_list = []

        for term in expansion_terms:
            if term in expansion_features:
                # Use original term if available, otherwise use stemmed term
                original_term = expansion_features[term].get('original_term', term)
                original_terms_for_encoding.append(original_term)
                term_importance_list.append(importance_scores.get(term, 0.0))

        if not original_terms_for_encoding:
            return query_embedding

        try:
            # Encode expansion terms
            term_embeddings = []
            for term in original_terms_for_encoding:
                term_emb = self.encode_text(term)
                term_embeddings.append(term_emb)

            # Stack term embeddings
            term_embeddings = torch.stack(term_embeddings)  # [num_terms, embedding_dim]

            # Convert importance scores to tensor
            importance_tensor = torch.tensor(
                term_importance_list,
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(1)  # [num_terms, 1]

            # KEY INSIGHT: Scale embeddings by importance (magnitude encoding)
            # This is different from concatenation - we're modulating the magnitude
            weighted_term_embeddings = term_embeddings * importance_tensor

            # Aggregate weighted terms (could also use attention here)
            if len(weighted_term_embeddings) > 0:
                # Normalize by sum of importance scores to prevent magnitude explosion
                total_importance = importance_tensor.sum()
                if total_importance > 1e-8:
                    expansion_embedding = weighted_term_embeddings.sum(dim=0) / total_importance
                else:
                    expansion_embedding = weighted_term_embeddings.mean(dim=0)
            else:
                expansion_embedding = torch.zeros_like(query_embedding)

            # Combine original query with expansion using learnable weight
            expansion_weight = torch.sigmoid(self.learnable_expansion_weight)
            enhanced_query = (1 - expansion_weight) * query_embedding + expansion_weight * expansion_embedding

            return enhanced_query

        except Exception as e:
            logger.warning(f"Error creating expanded query embedding: {e}")
            return query_embedding

    def forward(self,
                query: str,
                expansion_features: Dict[str, Dict[str, float]] = None) -> torch.Tensor:
        """
        Forward pass: create enhanced query representation.

        Args:
            query: Query text
            expansion_features: Optional expansion features

        Returns:
            Enhanced query embedding
        """
        if expansion_features:
            return self.create_expanded_query_embedding(query, expansion_features)
        else:
            return self.encode_text(query)

    def encode_query(self,
                    query: str,
                    expansion_features: Dict[str, Dict[str, float]] = None) -> torch.Tensor:
        """Encode query with optional expansion."""
        return self.forward(query, expansion_features)

    def retrieve(self,
                query: str,
                document_embeddings: torch.Tensor,
                document_ids: List[str],
                expansion_features: Dict[str, Dict[str, float]] = None,
                top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query text
            document_embeddings: Pre-computed document embeddings [num_docs, embedding_dim]
            document_ids: List of document IDs corresponding to embeddings
            expansion_features: Optional expansion features
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, score) pairs sorted by relevance
        """
        # Encode query (with expansion if provided)
        query_embedding = self.encode_query(query, expansion_features)

        # Compute similarities using the configured similarity function
        similarities = self.similarity_computer.compute_similarity(query_embedding, document_embeddings)

        # Get top-k documents
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(document_ids)))

        # Convert to list of (doc_id, score) tuples
        results = []
        for i, idx in enumerate(top_indices):
            doc_id = document_ids[idx.item()]
            score = top_scores[i].item()
            results.append((doc_id, score))

        return results

    def batch_retrieve(self,
                      queries: List[str],
                      document_embeddings: torch.Tensor,
                      document_ids: List[str],
                      expansion_features_list: List[Dict[str, Dict[str, float]]] = None,
                      top_k: int = 100) -> Dict[int, List[Tuple[str, float]]]:
        """
        Batch retrieval for multiple queries.

        Args:
            queries: List of query texts
            document_embeddings: Pre-computed document embeddings
            document_ids: List of document IDs
            expansion_features_list: List of expansion features (one per query)
            top_k: Number of documents to retrieve per query

        Returns:
            Dictionary mapping query index to results
        """
        results = {}

        for i, query in enumerate(queries):
            expansion_features = None
            if expansion_features_list and i < len(expansion_features_list):
                expansion_features = expansion_features_list[i]

            query_results = self.retrieve(
                query, document_embeddings, document_ids, expansion_features, top_k
            )
            results[i] = query_results

        return results

    def get_learned_weights(self) -> Tuple[float, float, float]:
        """Get current learned weights."""
        alpha = torch.sigmoid(self.alpha).item()
        beta = torch.sigmoid(self.beta).item()
        expansion_weight = torch.sigmoid(self.learnable_expansion_weight).item()
        return alpha, beta, expansion_weight

    def set_expansion_weight(self, weight: float):
        """Set the expansion weight (for ablation studies)."""
        with torch.no_grad():
            # Convert from probability space back to logit space
            self.learnable_expansion_weight.data = torch.logit(torch.tensor(weight, device=self.device))

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim

    def to(self, device):
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)

        # Move the module
        super().to(device)

        # Update device reference
        self.device = device

        # Move encoder and similarity computer
        self._move_encoder_to_device()
        self.similarity_computer = self.similarity_computer.to(device)

        logger.info(f"BiEncoder moved to device: {device}")
        return self


class BiEncoderIndexer:
    """
    Helper class for indexing documents with the bi-encoder.
    """

    def __init__(self, bi_encoder: HybridTermWeightingBiEncoder):
        """
        Initialize indexer with bi-encoder model.

        Args:
            bi_encoder: Trained bi-encoder model
        """
        self.bi_encoder = bi_encoder
        self.document_embeddings = None
        self.document_ids = None

    def index_documents(self,
                       documents: List[str],
                       document_ids: List[str],
                       batch_size: int = 32) -> torch.Tensor:
        """
        Index a collection of documents.

        Args:
            documents: List of document texts
            document_ids: List of document IDs
            batch_size: Batch size for encoding

        Returns:
            Document embeddings tensor
        """
        logger.info(f"Indexing {len(documents)} documents...")

        all_embeddings = []

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = self.bi_encoder.encode_documents(batch_docs)
            all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        self.document_embeddings = torch.cat(all_embeddings, dim=0)
        self.document_ids = document_ids

        logger.info(f"Indexed {len(self.document_ids)} documents")
        logger.info(f"Embedding shape: {self.document_embeddings.shape}")

        return self.document_embeddings

    def save_index(self, filepath: Path):
        """Save the document index."""
        if self.document_embeddings is None:
            raise ValueError("No documents indexed yet")

        torch.save({
            'embeddings': self.document_embeddings,
            'document_ids': self.document_ids,
            'embedding_dim': self.bi_encoder.embedding_dim,
            'model_name': self.bi_encoder.model_name
        }, filepath)

        logger.info(f"Saved index to {filepath}")

    def load_index(self, filepath: Path):
        """Load a document index."""
        data = torch.load(filepath, map_location=self.bi_encoder.device)

        self.document_embeddings = data['embeddings'].to(self.bi_encoder.device)
        self.document_ids = data['document_ids']

        logger.info(f"Loaded index from {filepath}")
        logger.info(f"Index contains {len(self.document_ids)} documents")

    def search(self,
              query: str,
              expansion_features: Dict[str, Dict[str, float]] = None,
              top_k: int = 100) -> List[Tuple[str, float]]:
        """Search the indexed documents."""
        if self.document_embeddings is None:
            raise ValueError("No documents indexed yet")

        return self.bi_encoder.retrieve(
            query, self.document_embeddings, self.document_ids, expansion_features, top_k
        )


# Factory function
def create_hybrid_bi_encoder(model_name: str = 'all-MiniLM-L6-v2',
                            max_expansion_terms: int = 15,
                            expansion_weight: float = 0.3,
                            similarity_function: str = 'cosine',
                            force_hf: bool = False,
                            pooling_strategy: str = 'cls',
                            **kwargs) -> HybridTermWeightingBiEncoder:
    """
    Factory function to create the hybrid bi-encoder.

    Args:
        model_name: Sentence transformer or HF model name
        max_expansion_terms: Maximum expansion terms
        expansion_weight: Weight for combining query with expansion
        similarity_function: Similarity function ('cosine', 'dot_product', 'learned', etc.)
        force_hf: Force using HuggingFace transformers
        pooling_strategy: Pooling strategy for HF models
        **kwargs: Additional arguments

    Returns:
        Configured bi-encoder model
    """
    return HybridTermWeightingBiEncoder(
        model_name=model_name,
        max_expansion_terms=max_expansion_terms,
        expansion_weight=expansion_weight,
        similarity_function=similarity_function,
        force_hf=force_hf,
        pooling_strategy=pooling_strategy,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Hybrid Lexical-Semantic Bi-Encoder...")

    # Create bi-encoder with cosine similarity
    bi_encoder = create_hybrid_bi_encoder(
        model_name='all-MiniLM-L6-v2',
        max_expansion_terms=10,
        similarity_function='cosine'
    )

    # Test query encoding
    query = "machine learning algorithms"
    expansion_features = {
        "neural": {"rm_weight": 0.5, "semantic_score": 0.8, "original_term": "neural"},
        "network": {"rm_weight": 0.7, "semantic_score": 0.9, "original_term": "networks"},
        "algorithm": {"rm_weight": 0.6, "semantic_score": 0.7, "original_term": "algorithms"}
    }

    print(f"Query: '{query}'")
    print(f"Expansion terms: {list(expansion_features.keys())}")

    # Test encoding
    basic_embedding = bi_encoder.encode_query(query)
    expanded_embedding = bi_encoder.encode_query(query, expansion_features)

    print(f"Basic embedding shape: {basic_embedding.shape}")
    print(f"Expanded embedding shape: {expanded_embedding.shape}")

    # Check if embeddings are different
    similarity = F.cosine_similarity(basic_embedding, expanded_embedding, dim=0)
    print(f"Similarity between basic and expanded: {similarity.item():.4f}")

    # Test document indexing
    documents = [
        "Neural networks are machine learning models inspired by biological neural networks.",
        "Support vector machines are supervised learning algorithms used for classification.",
        "Deep learning uses multiple layers to model and understand complex patterns in data."
    ]
    doc_ids = ["doc1", "doc2", "doc3"]

    # Create indexer and index documents
    indexer = BiEncoderIndexer(bi_encoder)
    indexer.index_documents(documents, doc_ids)

    # Test retrieval
    results = indexer.search(query, expansion_features, top_k=3)

    print(f"\nRetrieval results for '{query}':")
    for i, (doc_id, score) in enumerate(results):
        print(f"  {i+1}. {doc_id}: {score:.4f}")

    # Show learned weights
    alpha, beta, exp_weight = bi_encoder.get_learned_weights()
    print(f"\nLearned weights:")
    print(f"  α (RM3 weight): {alpha:.3f}")
    print(f"  β (Semantic weight): {beta:.3f}")
    print(f"  Expansion weight: {exp_weight:.3f}")

    print("\nHybrid bi-encoder test completed!")