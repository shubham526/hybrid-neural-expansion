"""
Enhanced Neural Reranker with Configurable Scoring Methods

Supports two scoring approaches:
1. Neural scoring layers (original)
2. Cosine similarity scoring
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from cross_encoder.src.core.semantic_similarity import SemanticSimilarity

logger = logging.getLogger(__name__)


class ConfigurableNeuralReranker(nn.Module):
    """
    Neural reranker with configurable scoring methods.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 max_expansion_terms: int = 15,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 scoring_method: str = "neural",  # "neural", "cosine", or "bilinear"
                 device: str = None,
                 force_hf: bool = False,  # NEW parameter
                 pooling_strategy: str = 'cls',
                 ablation_mode: str = "both"
    ):
        super().__init__()

        self.model_name = model_name
        self.max_expansion_terms = max_expansion_terms
        self.hidden_dim = hidden_dim
        self.scoring_method = scoring_method  # NEW
        self.force_hf = force_hf  # NEW
        self.pooling_strategy = pooling_strategy  # NEW

        self.ablation_mode = ablation_mode  # ADD THIS LINE after other assignments

        # Add validation
        if ablation_mode not in ["both", "rm3_only", "cosine_only"]:
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}. Must be 'both', 'rm3_only', or 'cosine_only'")

        logger.info(f"Ablation mode: {ablation_mode}")  # ADD THIS LINE

        # Determine device FIRST
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load sentence transformer with explicit device
        self.encoder = SemanticSimilarity(
            model_name=model_name,
            device=str(self.device),  # Pass device as string
            force_hf=force_hf,
            pooling_strategy=pooling_strategy
        )

        # Get embedding dimension after encoder is initialized
        if self.encoder.model_type == "sentence_transformer":
            self.embedding_dim = self.encoder.model.get_sentence_embedding_dimension()
            # Ensure encoder model is on correct device
            self.encoder.model = self.encoder.model.to(self.device)
        else:  # huggingface
            self.embedding_dim = self.encoder.embedding_dim
            # Ensure encoder model is on correct device
            self.encoder.model = self.encoder.model.to(self.device)

        # Update encoder's device reference
        self.encoder.device = self.device

        logger.info(f"Model device: {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Scoring method: {scoring_method}")

        # Learnable parameters (always needed for both methods)
        self.alpha = nn.Parameter(torch.tensor(0.5, device=self.device, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, device=self.device, dtype=torch.float32))
        self.expansion_weight = nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32))

        # Register parameters explicitly
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('beta', self.beta)
        self.register_parameter('expansion_weight', self.expansion_weight)

        # Setup scoring layers based on method
        if scoring_method == "neural":
            # Original neural scoring layers
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

            self.bilinear_layer = None

        elif scoring_method == "bilinear":
            # Bilinear scoring: learns interaction between query and doc embeddings
            self.bilinear_layer = nn.Bilinear(
                self.embedding_dim,  # query embedding size
                self.embedding_dim,  # document embedding size
                1,                   # output size (single score)
                bias=True
            ).to(self.device)

            self.scoring_layers = None
            logger.info("Using bilinear scoring")

        elif scoring_method == "cosine":
            # No additional layers needed for cosine similarity
            self.scoring_layers = None
            self.bilinear_layer = None
            logger.info("Using cosine similarity scoring - no neural layers")

        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}. Use 'neural', 'bilinear', or 'cosine'")

        # Initialize weights
        self._init_weights()

        logger.info(f"ConfigurableNeuralReranker initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Scoring method: {scoring_method}")
        logger.info(f"  Initial weights: α={self.alpha.item():.3f}, β={self.beta.item():.3f}")

    def _init_weights(self):
        """Initialize neural network weights."""
        if self.scoring_method == "neural" and self.scoring_layers is not None:
            for layer in self.scoring_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        elif self.scoring_method == "bilinear" and self.bilinear_layer is not None:
            # Initialize bilinear layer
            nn.init.xavier_uniform_(self.bilinear_layer.weight)
            if self.bilinear_layer.bias is not None:
                nn.init.zeros_(self.bilinear_layer.bias)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text ensuring output is on correct device."""
        # Use the enhanced SemanticSimilarity encode method
        embedding = self.encoder.encode(text)

        # Convert numpy to torch tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).requires_grad_(True)

        embedding = embedding.to(self.device, dtype=torch.float32)
        return embedding

    def encode_terms(self, terms: List[str]) -> torch.Tensor:
        """Encode multiple terms ensuring output is on correct device."""
        if not terms:
            return torch.zeros(1, self.embedding_dim, device=self.device, dtype=torch.float32)

        # Use the enhanced SemanticSimilarity encode method
        embeddings = self.encoder.encode(terms)

        # Convert numpy to torch tensor if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).requires_grad_(True)
        elif isinstance(embeddings, list):
            embeddings = torch.from_numpy(np.array(embeddings)).requires_grad_(True)

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

                # Compute importance weight based on ablation mode
                if self.ablation_mode == "rm3_only":
                    # Use only RM3 component
                    importance = self.alpha * rm_weight
                elif self.ablation_mode == "cosine_only":
                    # Use only cosine similarity component
                    importance = self.beta * semantic_score
                else:  # self.ablation_mode == "both"
                    # Use both components (original behavior)
                    importance = self.alpha * rm_weight + self.beta * semantic_score

                # In reranker.py, after computing importance:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Ablation: {self.ablation_mode}, RM3: {rm_weight.item():.4f}, "
                                 f"Semantic: {semantic_score.item():.4f}, Final: {importance.item():.4f}")

                # Weight the term embedding
                term_emb = term_embeddings[i]  # Already on correct device
                weighted_term_sum += importance * term_emb
                total_weight += importance.squeeze()

        # Combine with original query
        if total_weight > 1e-8:
            # FIX 6: Create mixing weight on correct device
            # expansion_weight = torch.tensor(0.3, device=self.device, dtype=torch.float32)
            # expansion_weight = torch.sigmoid(expansion_weight)
            expansion_weight = torch.sigmoid(self.expansion_weight)

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
        Forward pass with configurable scoring method.
        """
        try:
            # 1. Create enhanced query embedding
            enhanced_query_emb = self.create_enhanced_query_embedding(query, expansion_features)

            # 2. Get document embedding
            document_emb = self.encode_text(document)

            # 3. Validation
            assert enhanced_query_emb.dim() == 1 and enhanced_query_emb.size(0) == self.embedding_dim
            assert document_emb.dim() == 1 and document_emb.size(0) == self.embedding_dim
            assert enhanced_query_emb.device == self.device
            assert document_emb.device == self.device

            # 4. Score using selected method
            if self.scoring_method == "neural":
                return self._neural_scoring(enhanced_query_emb, document_emb)
            elif self.scoring_method == "bilinear":
                return self._bilinear_scoring(enhanced_query_emb, document_emb)
            elif self.scoring_method == "cosine":
                return self._cosine_scoring(enhanced_query_emb, document_emb)
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring_method}")

        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            logger.error(f"Model device: {self.device}")
            logger.error(f"Alpha device: {self.alpha.device}")
            logger.error(f"Beta device: {self.beta.device}")
            raise

    def _neural_scoring(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Score using neural layers (original method)."""
        # Concatenate embeddings
        combined_emb = torch.cat([query_emb, doc_emb], dim=0)

        # Final device check
        assert combined_emb.device == self.device, f"Combined embedding on wrong device: {combined_emb.device}"

        # Score via neural network
        score = self.scoring_layers(combined_emb)
        return score.squeeze()

    def _bilinear_scoring(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Score using bilinear layer."""
        # Add batch dimension for bilinear layer
        query_emb_batch = query_emb.unsqueeze(0)  # [1, embedding_dim]
        doc_emb_batch = doc_emb.unsqueeze(0)      # [1, embedding_dim]

        # Compute bilinear score: query^T W doc + b
        score = self.bilinear_layer(query_emb_batch, doc_emb_batch)  # [1, 1]

        # Return as scalar tensor
        return score.squeeze()

    def _cosine_scoring(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Score using cosine similarity."""
        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(query_emb.unsqueeze(0), doc_emb.unsqueeze(0), dim=1)

        # Return as scalar tensor
        return cosine_sim.squeeze()

    def get_learned_weights(self) -> Tuple[float, float, float]:
        """Get current learned α, β, and λ values."""
        # Get the value of lambda by applying the sigmoid
        expansion_weight = torch.sigmoid(self.expansion_weight).item()
        return self.alpha.item(), self.beta.item(), expansion_weight

    def get_scoring_method(self) -> str:
        """Get current scoring method."""
        return self.scoring_method

    def switch_scoring_method(self, new_method: str):
        """
        Switch scoring method (experimental - use with caution).

        Note: This recreates the scoring layers, so only use before training
        or be prepared to retrain.
        """
        if new_method not in ["neural", "bilinear", "cosine"]:
            raise ValueError(f"Unknown scoring method: {new_method}")

        old_method = self.scoring_method
        self.scoring_method = new_method

        # Clean up old layers
        self.scoring_layers = None
        self.bilinear_layer = None

        if new_method == "neural":
            # Add neural layers
            scoring_input_dim = 2 * self.embedding_dim
            self.scoring_layers = nn.Sequential(
                nn.Linear(scoring_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, 1)
            ).to(self.device)

        elif new_method == "bilinear":
            # Add bilinear layer
            self.bilinear_layer = nn.Bilinear(
                self.embedding_dim,
                self.embedding_dim,
                1,
                bias=True
            ).to(self.device)

        # cosine method needs no additional layers

        # Initialize new layers
        self._init_weights()
        logger.info(f"Switched scoring method from {old_method} to {new_method}")

    def to(self, device):
        """Override to method to ensure all components are moved to device."""
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)

        # Move the main module
        super().to(device)

        # Update device reference
        self.device = device

        # Move encoder components
        if hasattr(self.encoder, 'model'):
            if self.encoder.model_type == "sentence_transformer":
                self.encoder.model = self.encoder.model.to(device)
            else:  # huggingface
                self.encoder.model = self.encoder.model.to(device)
            self.encoder.device = device

        # Ensure parameters are on correct device
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)

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


# Factory function with scoring method option
def create_neural_reranker(model_name: str = 'all-MiniLM-L6-v2',
                           scoring_method: str = "neural",
                           force_hf: bool = False,  # NEW parameter
                           pooling_strategy: str = 'cls',
                           ablation_mode: str = "both", # NEW parameter
                           **kwargs) -> ConfigurableNeuralReranker:
    """
    Factory function to create neural reranker with configurable scoring.

    Args:
        model_name: Model name (SentenceTransformer or HuggingFace)
        scoring_method: "neural", "bilinear", or "cosine"
        force_hf: Force using HuggingFace transformers
        pooling_strategy: Pooling strategy for HF models
        **kwargs: Other model parameters

    Returns:
        Configured neural reranker
    """
    return ConfigurableNeuralReranker(
        model_name=model_name,
        scoring_method=scoring_method,
        force_hf=force_hf,
        pooling_strategy=pooling_strategy,
        ablation_mode=ablation_mode,
        **kwargs
    )


# Example usage and comparison
if __name__ == "__main__":
    print("Testing configurable neural reranker...")

    # Test all three scoring methods
    for method in ["neural", "bilinear", "cosine"]:
        print(f"\n--- Testing {method} scoring ---")

        reranker = create_neural_reranker(scoring_method=method)
        print(f"Scoring method: {reranker.get_scoring_method()}")

        # Count parameters
        total_params = sum(p.numel() for p in reranker.parameters())
        trainable_params = sum(p.numel() for p in reranker.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Test data
        query = "machine learning algorithms"
        document = "Neural networks are machine learning algorithms."
        expansion_features = {
            "neural": {"rm_weight": 0.5, "semantic_score": 0.8, "original_term": "neural"},
            "algorithm": {"rm_weight": 0.7, "semantic_score": 0.9, "original_term": "algorithms"}
        }

        try:
            score = reranker.forward(query, expansion_features, document)
            print(f"Score: {score.item():.4f}")
            print(f"Score device: {score.device}")

            alpha, beta = reranker.get_learned_weights()
            print(f"Learned weights: α={alpha:.3f}, β={beta:.3f}")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nConfigurable reranker test complete!")