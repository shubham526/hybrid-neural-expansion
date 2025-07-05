"""
Enhanced Semantic Similarity Module with Device Management Fixes

Key fixes:
1. Better device handling for both SentenceTransformer and HuggingFace models
2. Consistent tensor/numpy conversion with device awareness
3. Proper device parameter handling
4. Better error handling for device-related issues
"""

import logging
import numpy as np
import torch
from typing import List, Union, Dict, Optional
from functools import lru_cache
import threading

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """
    Semantic similarity computation using sentence transformers or HuggingFace models.
    Enhanced with better device management.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 cache_size: int = 1000,
                 force_hf: bool = False,
                 pooling_strategy: str = 'cls'):
        """
        Initialize semantic similarity computer.

        Args:
            model_name: Model name (SentenceTransformer or HuggingFace)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_size: Size of embedding cache (0 to disable)
            force_hf: Force using HuggingFace transformers even if SentenceTransformers available
            pooling_strategy: Pooling strategy for HF models ('cls', 'mean', 'max')
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.force_hf = force_hf

        # FIX 1: Better device handling
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            # Handle both string and torch.device inputs
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device

        # Initialize model
        self.model_type = self._initialize_model()

        # Thread-safe cache for embeddings
        self.cache_size = cache_size
        if cache_size > 0:
            self._embedding_cache = {}
            self._cache_lock = threading.Lock()
        else:
            self._embedding_cache = None
            self._cache_lock = None

        logger.info(f"Loaded model: {model_name} ({self.model_type}) on {self.device}")
        logger.debug(f"Embedding cache size: {cache_size}")

    def _initialize_model(self):
        """Initialize either SentenceTransformer or HuggingFace model with proper device handling."""

        # Try SentenceTransformers first (unless forced to use HF)
        if not self.force_hf and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # FIX 2: Pass device as string to SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                self.tokenizer = None

                # Verify model is on correct device
                if hasattr(self.model, '_modules'):
                    for module in self.model._modules.values():
                        if hasattr(module, 'to'):
                            module.to(self.device)

                logger.info(f"SentenceTransformer model loaded on device: {self.device}")
                return "sentence_transformer"
            except Exception as e:
                logger.warning(f"Failed to load as SentenceTransformer: {e}")
                logger.info("Falling back to HuggingFace transformers...")

        # Try HuggingFace transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)

                # FIX 3: Ensure model is moved to correct device
                self.model = self.model.to(self.device)
                self.model.eval()

                # Get embedding dimension
                config = AutoConfig.from_pretrained(self.model_name)
                self.embedding_dim = config.hidden_size

                logger.info(f"HuggingFace model loaded on device: {self.device}")
                return "huggingface"
            except Exception as e:
                logger.error(f"Failed to load as HuggingFace model: {e}")
                raise

        # No suitable library available
        missing_libs = []
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            missing_libs.append("sentence-transformers")
        if not TRANSFORMERS_AVAILABLE:
            missing_libs.append("transformers")

        raise ImportError(f"Required libraries not available: {missing_libs}")

    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to hidden states."""
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]

        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling_strategy == 'max':
            # Max pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            return torch.max(hidden_states, 1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

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
            if self.model_type == "sentence_transformer":
                with torch.no_grad():
                    # FIX 4: Ensure output is properly converted to numpy
                    embedding = self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]

                    # Ensure it's numpy array
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    elif not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)

                return embedding

            else:  # huggingface
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                # FIX 5: Ensure inputs are moved to correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']

                    # Apply pooling
                    pooled = self._pool_embeddings(hidden_states, attention_mask)
                    # FIX 6: Ensure proper device -> CPU -> numpy conversion
                    embedding = pooled.squeeze().cpu().numpy()

                return embedding

        except Exception as e:
            logger.error(f"Failed to encode text '{text[:50]}...': {e}")
            logger.error(f"Model device: {self.device}")
            # Return zero embedding as fallback
            if self.model_type == "sentence_transformer":
                dim = self.model.get_sentence_embedding_dimension()
            else:
                dim = self.embedding_dim
            return np.zeros(dim, dtype=np.float32)

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

        # Handle batch encoding
        try:
            if self.model_type == "sentence_transformer":
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        convert_to_tensor=False,
                        show_progress_bar=False
                    )

                    # FIX 7: Ensure batch outputs are properly converted
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.cpu().numpy()
                    elif not isinstance(embeddings, np.ndarray):
                        embeddings = np.array(embeddings)

                return embeddings

            else:  # huggingface - process in batches
                all_embeddings = []

                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=512
                    )

                    # FIX 8: Ensure batch inputs are moved to correct device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        hidden_states = outputs.last_hidden_state
                        attention_mask = inputs['attention_mask']

                        # Apply pooling
                        pooled = self._pool_embeddings(hidden_states, attention_mask)
                        # FIX 9: Proper device -> CPU -> numpy conversion for batch
                        batch_embeddings = pooled.cpu().numpy()

                    all_embeddings.extend(batch_embeddings)

                return all_embeddings

        except Exception as e:
            logger.error(f"Failed to encode {len(texts)} texts: {e}")
            logger.error(f"Model device: {self.device}")
            # Return zero embeddings as fallback
            if self.model_type == "sentence_transformer":
                dim = self.model.get_sentence_embedding_dimension()
            else:
                dim = self.embedding_dim
            return [np.zeros(dim, dtype=np.float32) for _ in texts]

    def compute_similarity(self,
                          text1: Union[str, np.ndarray],
                          text2: Union[str, np.ndarray]) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text (string or pre-computed embedding)
            text2: Second text (string or pre-computed embedding)

        Returns:
            Cosine similarity score (-1 to 1, but clamped to 0-1)
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

            # FIX 10: Ensure embeddings are numpy arrays
            if isinstance(emb1, torch.Tensor):
                emb1 = emb1.cpu().numpy()
            if isinstance(emb2, torch.Tensor):
                emb2 = emb2.cpu().numpy()

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

    def get_cache_info(self) -> Dict[str, Union[int, bool, str]]:
        """Get information about the cache and model."""
        cache_info = {"cache_enabled": False}

        if self._embedding_cache is not None:
            with self._cache_lock:
                cache_info = {
                    "cache_enabled": True,
                    "cache_size": len(self._embedding_cache),
                    "max_cache_size": self.cache_size
                }

        cache_info.update({
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": str(self.device),
            "pooling_strategy": self.pooling_strategy if self.model_type == "huggingface" else "N/A"
        })

        return cache_info

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

    # FIX 11: Add method to move model to different device
    def to(self, device):
        """Move model to specified device."""
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        if self.model_type == "sentence_transformer":
            # Move SentenceTransformer model
            if hasattr(self.model, 'to'):
                self.model.to(device)
            elif hasattr(self.model, '_modules'):
                for module in self.model._modules.values():
                    if hasattr(module, 'to'):
                        module.to(device)
        else:  # huggingface
            self.model.to(device)

        logger.info(f"SemanticSimilarity model moved to device: {device}")
        return self


# Example usage remains the same...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Enhanced Semantic Similarity Module Test")
    print("=" * 42)

    # Test device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    try:
        sim_computer = SemanticSimilarity(
            model_name='all-MiniLM-L6-v2',
            device=device,
            force_hf=False
        )

        query = "machine learning algorithms"
        expansion_terms = ["neural", "networks", "supervised", "classification"]

        print(f"Query: '{query}'")
        print(f"Expansion terms: {expansion_terms}")

        # Test encoding
        query_emb = sim_computer.encode(query)
        print(f"Query embedding shape: {query_emb.shape}")
        print(f"Query embedding type: {type(query_emb)}")

        # Test similarities
        similarities = sim_computer.compute_query_expansion_similarities(query, expansion_terms)

        print("\nExpansion term similarities:")
        for term, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {term:<15} {sim:.4f}")

        print("\nDevice test complete!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()