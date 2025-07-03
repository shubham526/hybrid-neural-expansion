"""
Device-Fixed Neural Reranker

Fixes the "Expected all tensors to be on the same device" error by ensuring
all tensor operations use the correct device consistently.
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
    Device-aware neural reranker with proper tensor device handling.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_expansion_terms: int = 15,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 device: str = None):
        super().__init__()

        self.model_name = model_name
        self.max_expansion_terms = max_expansion_terms
        self.hidden_dim = hidden_dim

        # Load sentence transformer
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.device = self.encoder.device

        logger.info(f"Model device: {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # FIX 1: Ensure learnable parameters are on correct device
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, device=self.device, dtype=torch.float32))

        # Register parameters explicitly
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('beta', self.beta)

        # Scoring layers
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
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Initial weights: α={self.alpha.item():.3f}, β={self.beta.item():.3f}")

    def _init_weights(self):
        """Initialize neural network weights."""
        for layer in self.scoring_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text ensuring output is on correct device."""
        with torch.no_grad():
            embedding = self.encoder.encode([text], convert_to_tensor=True)[0]

        # FIX 2: Ensure embedding is on correct device
        embedding = embedding.to(self.device, dtype=torch.float32)
        return embedding

    def encode_terms(self, terms: List[str]) -> torch.Tensor:
        """Encode multiple terms ensuring output is on correct device."""
        if not terms:
            return torch.zeros(1, self.embedding_dim, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            embeddings = self.encoder.encode(terms, convert_to_tensor=True)

        # FIX 3: Ensure term embeddings are on correct device
        embeddings = embeddings.to(self.device, dtype=torch.float32)
        return embeddings

    def create_enhanced_query_embedding(self,
                                        query: str,
                                        expansion_features: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """
        Create enhanced query embedding with proper device handling.
        """
        # Get base query embedding
        query_embedding = self.encode_text(query)  # Already on correct device

        # Validate shape and device
        assert query_embedding.dim() == 1, f"Query embedding should be 1D, got {query_embedding.shape}"
        assert query_embedding.device == self.device, f"Query embedding on wrong device: {query_embedding.device}"

        # Get the top-k stemmed terms for iteration
        stemmed_expansion_terms = list(expansion_features.keys())[:self.max_expansion_terms]

        if not stemmed_expansion_terms:
            logger.debug("No expansion terms, returning original query embedding")
            return query_embedding

        # --- FIX STARTS HERE ---
        # Create a new list of the *original* (unstemmed) words for semantic encoding.
        # We fall back to the stemmed term if 'original_term' is somehow missing.
        original_terms_for_encoding = [
            word
            for term in stemmed_expansion_terms
            for word in expansion_features[term].get('original_term', [term])
        ]
        # --- FIX ENDS HERE ---

        # Get term embeddings using the correct, original words
        try:
            # Pass the list of original terms to the encoder
            term_embeddings = self.encode_terms(original_terms_for_encoding)
            assert term_embeddings.device == self.device, f"Term embeddings on wrong device: {term_embeddings.device}"
        except Exception as e:
            logger.warning(f"Error encoding terms: {e}")
            return query_embedding

        # FIX 4: Ensure all intermediate tensors are on correct device
        weighted_term_sum = torch.zeros_like(query_embedding, device=self.device, dtype=torch.float32)
        total_weight = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        for i, term in enumerate(stemmed_expansion_terms):
            if term in expansion_features:
                features = expansion_features[term]

                # FIX 5: Create tensors directly on correct device
                rm_weight = torch.tensor(
                    features.get('rm_weight', 0.0),
                    device=self.device,
                    dtype=torch.float32
                )
                semantic_score = torch.tensor(
                    features.get('semantic_score', 0.0),
                    device=self.device,
                    dtype=torch.float32
                )

                # Compute importance weight (all tensors on same device)
                importance = self.alpha * rm_weight + self.beta * semantic_score

                # Weight the term embedding
                term_emb = term_embeddings[i]  # Already on correct device
                weighted_term_sum += importance * term_emb
                total_weight += importance.squeeze()

        # Combine with original query
        if total_weight > 1e-8:
            # FIX 6: Create mixing weight on correct device
            expansion_weight = torch.tensor(0.3, device=self.device, dtype=torch.float32)
            expansion_weight = torch.sigmoid(expansion_weight)

            enhanced_query = (1 - expansion_weight) * query_embedding + expansion_weight * (weighted_term_sum / total_weight)
        else:
            enhanced_query = query_embedding

        # Final validation
        assert enhanced_query.dim() == 1, f"Enhanced query should be 1D, got {enhanced_query.shape}"
        assert enhanced_query.device == self.device, f"Enhanced query on wrong device: {enhanced_query.device}"

        return enhanced_query

    def forward(self,
                query: str,
                expansion_features: Dict[str, Dict[str, float]],
                document: str) -> torch.Tensor:
        """
        Forward pass with comprehensive device checking.
        """
        try:
            # 1. Create enhanced query embedding
            enhanced_query_emb = self.create_enhanced_query_embedding(query, expansion_features)

            # Device check
            if enhanced_query_emb.device != self.device:
                logger.error(f"Enhanced query embedding on wrong device: {enhanced_query_emb.device}, expected: {self.device}")
                enhanced_query_emb = enhanced_query_emb.to(self.device)

            # 2. Get document embedding
            document_emb = self.encode_text(document)

            # Device check
            if document_emb.device != self.device:
                logger.error(f"Document embedding on wrong device: {document_emb.device}, expected: {self.device}")
                document_emb = document_emb.to(self.device)

            # 3. Validation
            assert enhanced_query_emb.dim() == 1 and enhanced_query_emb.size(0) == self.embedding_dim
            assert document_emb.dim() == 1 and document_emb.size(0) == self.embedding_dim
            assert enhanced_query_emb.device == self.device
            assert document_emb.device == self.device

            # 4. Concatenate embeddings
            combined_emb = torch.cat([enhanced_query_emb, document_emb], dim=0)

            # Final device check
            assert combined_emb.device == self.device, f"Combined embedding on wrong device: {combined_emb.device}"

            # 5. Score via neural network
            score = self.scoring_layers(combined_emb)

            return score.squeeze()

        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            logger.error(f"Model device: {self.device}")
            logger.error(f"Alpha device: {self.alpha.device}")
            logger.error(f"Beta device: {self.beta.device}")

            # Additional debugging
            try:
                test_query_emb = self.encode_text(query)
                test_doc_emb = self.encode_text(document)
                logger.error(f"Test query embedding device: {test_query_emb.device}")
                logger.error(f"Test document embedding device: {test_doc_emb.device}")
            except Exception as debug_e:
                logger.error(f"Even basic encoding failed: {debug_e}")

            raise

    def get_learned_weights(self) -> Tuple[float, float]:
        """Get current learned α and β values."""
        return self.alpha.item(), self.beta.item()

    def to(self, device):
        """Override to method to ensure sentence transformer is also moved."""
        super().to(device)

        # FIX 7: Also move the sentence transformer
        self.encoder = self.encoder.to(device)
        self.device = device

        logger.info(f"Model moved to device: {device}")
        return self

    def cuda(self, device=None):
        """Override cuda method."""
        if device is None:
            device = torch.cuda.current_device()
        return self.to(f'cuda:{device}')

    def cpu(self):
        """Override cpu method."""
        return self.to('cpu')

    def rerank_candidates(self,
                          query: str,
                          expansion_features: Dict[str, Dict[str, float]],
                          candidates: List[Tuple[str, float]],
                          document_texts: Dict[str, str],
                          top_k: int = 100) -> List[Tuple[str, float]]:
        """Rerank candidates with device safety."""
        if not candidates:
            return []

        self.eval()
        results = []

        with torch.no_grad():
            for doc_id, first_stage_score in candidates:
                if doc_id not in document_texts:
                    results.append((doc_id, first_stage_score))
                    continue

                try:
                    neural_score = self.forward(
                        query,
                        expansion_features,
                        document_texts[doc_id]
                    ).item()

                    results.append((doc_id, neural_score))

                except Exception as e:
                    logger.warning(f"Error scoring document {doc_id}: {e}")
                    results.append((doc_id, first_stage_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Factory function
def create_neural_reranker(model_name: str = 'all-MiniLM-L6-v2',
                           **kwargs) -> ImportanceWeightedNeuralReranker:
    """Factory function to create neural reranker."""
    return ImportanceWeightedNeuralReranker(model_name=model_name, **kwargs)


# Device debugging utility
def check_model_devices(model):
    """Debug function to check all model component devices."""
    print("=== MODEL DEVICE CHECK ===")
    print(f"Model device attribute: {model.device}")
    print(f"Encoder device: {model.encoder.device}")

    for name, param in model.named_parameters():
        print(f"Parameter {name}: {param.device}")

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"Module {name} weight: {module.weight.device}")

    print("=== END DEVICE CHECK ===")


if __name__ == "__main__":
    # Test device handling
    print("Testing device handling...")

    # Create model
    reranker = create_neural_reranker()
    print(f"Initial device: {reranker.device}")

    # Check devices
    check_model_devices(reranker)

    # Test data
    query = "machine learning algorithms"
    document = "Neural networks are machine learning algorithms."
    expansion_features = {
        "neural": {"rm_weight": 0.5, "semantic_score": 0.8},
        "algorithm": {"rm_weight": 0.7, "semantic_score": 0.9}
    }

    try:
        score = reranker.forward(query, expansion_features, document)
        print(f"Forward pass successful! Score: {score.item()}")
        print(f"Score device: {score.device}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()