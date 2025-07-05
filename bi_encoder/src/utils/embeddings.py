"""
Embedding Utilities for Bi-Encoder Systems

This module provides utilities for efficient embedding computation, storage,
and manipulation in dense retrieval systems.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import pickle
import json
import h5py
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding computation, storage, and retrieval for bi-encoder systems.

    Handles both in-memory and disk-based storage with efficient batch processing.
    """

    def __init__(self,
                 embedding_dim: int,
                 storage_format: str = 'hdf5',
                 normalize_embeddings: bool = True,
                 device: str = None):
        """
        Initialize embedding manager.

        Args:
            embedding_dim: Dimension of embeddings
            storage_format: Storage format ('hdf5', 'numpy', 'pickle')
            normalize_embeddings: Whether to L2 normalize embeddings
            device: Device for tensor operations
        """
        self.embedding_dim = embedding_dim
        self.storage_format = storage_format
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # In-memory storage
        self.embeddings = {}
        self.metadata = {}

        logger.info(f"EmbeddingManager initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Storage format: {storage_format}")
        logger.info(f"  Normalize: {normalize_embeddings}")
        logger.info(f"  Device: {self.device}")

    def compute_embeddings(self,
                           texts: List[str],
                           model,
                           batch_size: int = 32,
                           show_progress: bool = True,
                           ids: Optional[List[str]] = None) -> torch.Tensor:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of text strings
            model: Bi-encoder model with encode_documents method
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            ids: Optional list of IDs corresponding to texts

        Returns:
            Tensor of embeddings [num_texts, embedding_dim]
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)

        model.eval()
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                # Encode batch
                if hasattr(model, 'encode_documents'):
                    batch_embeddings = model.encode_documents(batch_texts)
                elif hasattr(model, 'encode'):
                    batch_embeddings = model.encode(batch_texts)
                else:
                    raise ValueError("Model must have 'encode_documents' or 'encode' method")

                # Move to CPU and normalize if requested
                batch_embeddings = batch_embeddings.cpu()
                if self.normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)

        # Store in memory if IDs provided
        if ids is not None:
            for i, (text_id, embedding) in enumerate(zip(ids, embeddings)):
                self.embeddings[text_id] = embedding
                self.metadata[text_id] = {
                    'text': texts[i] if i < len(texts) else '',
                    'embedding_dim': self.embedding_dim,
                    'normalized': self.normalize_embeddings
                }

        logger.info(f"Computed {len(embeddings)} embeddings")
        return embeddings

    def save_embeddings(self,
                        embeddings: torch.Tensor,
                        ids: List[str],
                        filepath: Path,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: Tensor of embeddings
            ids: List of IDs corresponding to embeddings
            filepath: Path to save embeddings
            metadata: Optional metadata to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.storage_format == 'hdf5':
            self._save_hdf5(embeddings, ids, filepath, metadata)
        elif self.storage_format == 'numpy':
            self._save_numpy(embeddings, ids, filepath, metadata)
        elif self.storage_format == 'pickle':
            self._save_pickle(embeddings, ids, filepath, metadata)
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")

        logger.info(f"Saved {len(embeddings)} embeddings to: {filepath}")

    def load_embeddings(self, filepath: Path) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        """
        Load embeddings from disk.

        Args:
            filepath: Path to load embeddings from

        Returns:
            Tuple of (embeddings, ids, metadata)
        """
        filepath = Path(filepath)

        if self.storage_format == 'hdf5':
            return self._load_hdf5(filepath)
        elif self.storage_format == 'numpy':
            return self._load_numpy(filepath)
        elif self.storage_format == 'pickle':
            return self._load_pickle(filepath)
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")

    def _save_hdf5(self, embeddings: torch.Tensor, ids: List[str],
                   filepath: Path, metadata: Optional[Dict[str, Any]]):
        """Save embeddings in HDF5 format (recommended for large datasets)."""
        embeddings_np = embeddings.numpy()

        with h5py.File(filepath, 'w') as f:
            # Save embeddings
            f.create_dataset('embeddings', data=embeddings_np, compression='gzip')

            # Save IDs (as string dataset)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ids', data=ids, dtype=dt)

            # Save metadata
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_group.attrs[key] = value
                    else:
                        # For complex objects, save as JSON string
                        metadata_group.attrs[key] = json.dumps(value)

            # Save embedding info
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['normalized'] = self.normalize_embeddings
            f.attrs['num_embeddings'] = len(embeddings)

    def _load_hdf5(self, filepath: Path) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        """Load embeddings from HDF5 format."""
        with h5py.File(filepath, 'r') as f:
            # Load embeddings
            embeddings_np = f['embeddings'][:]
            embeddings = torch.from_numpy(embeddings_np)

            # Load IDs
            ids = [id_bytes.decode('utf-8') if isinstance(id_bytes, bytes) else str(id_bytes)
                   for id_bytes in f['ids'][:]]

            # Load metadata
            metadata = {}
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    value = metadata_group.attrs[key]
                    if isinstance(value, str) and value.startswith('{'):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            metadata[key] = value
                    else:
                        metadata[key] = value

            # Add embedding info
            metadata['embedding_dim'] = f.attrs.get('embedding_dim', self.embedding_dim)
            metadata['normalized'] = f.attrs.get('normalized', self.normalize_embeddings)
            metadata['num_embeddings'] = f.attrs.get('num_embeddings', len(embeddings))

        logger.info(f"Loaded {len(embeddings)} embeddings from HDF5: {filepath}")
        return embeddings, ids, metadata

    def _save_numpy(self, embeddings: torch.Tensor, ids: List[str],
                    filepath: Path, metadata: Optional[Dict[str, Any]]):
        """Save embeddings in NumPy format."""
        embeddings_np = embeddings.numpy()

        # Save embeddings
        np.save(filepath.with_suffix('.npy'), embeddings_np)

        # Save IDs and metadata
        data = {
            'ids': ids,
            'metadata': metadata or {},
            'embedding_dim': self.embedding_dim,
            'normalized': self.normalize_embeddings,
            'num_embeddings': len(embeddings)
        }

        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def _load_numpy(self, filepath: Path) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        """Load embeddings from NumPy format."""
        # Load embeddings
        embeddings_np = np.load(filepath.with_suffix('.npy'))
        embeddings = torch.from_numpy(embeddings_np)

        # Load IDs and metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            data = json.load(f)

        ids = data['ids']
        metadata = data.get('metadata', {})
        metadata.update({
            'embedding_dim': data.get('embedding_dim', self.embedding_dim),
            'normalized': data.get('normalized', self.normalize_embeddings),
            'num_embeddings': data.get('num_embeddings', len(embeddings))
        })

        logger.info(f"Loaded {len(embeddings)} embeddings from NumPy: {filepath}")
        return embeddings, ids, metadata

    def _save_pickle(self, embeddings: torch.Tensor, ids: List[str],
                     filepath: Path, metadata: Optional[Dict[str, Any]]):
        """Save embeddings in Pickle format (not recommended for large datasets)."""
        data = {
            'embeddings': embeddings,
            'ids': ids,
            'metadata': metadata or {},
            'embedding_dim': self.embedding_dim,
            'normalized': self.normalize_embeddings,
            'num_embeddings': len(embeddings)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, filepath: Path) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
        """Load embeddings from Pickle format."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        ids = data['ids']
        metadata = data.get('metadata', {})
        metadata.update({
            'embedding_dim': data.get('embedding_dim', self.embedding_dim),
            'normalized': data.get('normalized', self.normalize_embeddings),
            'num_embeddings': data.get('num_embeddings', len(embeddings))
        })

        logger.info(f"Loaded {len(embeddings)} embeddings from Pickle: {filepath}")
        return embeddings, ids, metadata

    def get_embedding(self, text_id: str) -> Optional[torch.Tensor]:
        """Get embedding for a specific ID from memory."""
        return self.embeddings.get(text_id)

    def add_embedding(self, text_id: str, embedding: torch.Tensor, text: str = ""):
        """Add embedding to memory storage."""
        if self.normalize_embeddings:
            embedding = F.normalize(embedding, p=2, dim=0)

        self.embeddings[text_id] = embedding
        self.metadata[text_id] = {
            'text': text,
            'embedding_dim': embedding.size(0),
            'normalized': self.normalize_embeddings
        }

    def clear_memory(self):
        """Clear in-memory embeddings."""
        self.embeddings.clear()
        self.metadata.clear()
        gc.collect()
        logger.info("Cleared embedding memory")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.embeddings:
            return {'num_embeddings': 0, 'memory_mb': 0}

        total_elements = sum(emb.numel() for emb in self.embeddings.values())
        memory_bytes = total_elements * 4  # Assuming float32
        memory_mb = memory_bytes / (1024 * 1024)

        return {
            'num_embeddings': len(self.embeddings),
            'total_elements': total_elements,
            'memory_mb': memory_mb,
            'avg_embedding_dim': total_elements / len(self.embeddings) if self.embeddings else 0
        }


class EmbeddingCache:
    """
    LRU cache for embeddings to manage memory usage.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []

        logger.info(f"EmbeddingCache initialized with max_size={max_size}")

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, embedding: torch.Tensor):
        """Put embedding in cache."""
        if key in self.cache:
            # Update existing
            self.cache[key] = embedding
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = embedding
            self.access_order.append(key)

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def normalize_embeddings(embeddings: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Input embeddings tensor
        dim: Dimension to normalize along

    Returns:
        Normalized embeddings
    """
    return F.normalize(embeddings, p=2, dim=dim)


def compute_similarity_matrix(query_embeddings: torch.Tensor,
                              doc_embeddings: torch.Tensor,
                              similarity_type: str = 'cosine') -> torch.Tensor:
    """
    Compute similarity matrix between queries and documents.

    Args:
        query_embeddings: [num_queries, embedding_dim]
        doc_embeddings: [num_docs, embedding_dim]
        similarity_type: Type of similarity ('cosine', 'dot', 'euclidean')

    Returns:
        Similarity matrix [num_queries, num_docs]
    """
    if similarity_type == 'cosine':
        # Normalize and compute dot product
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=1)
        return torch.matmul(query_norm, doc_norm.t())

    elif similarity_type == 'dot':
        return torch.matmul(query_embeddings, doc_embeddings.t())

    elif similarity_type == 'euclidean':
        # Negative euclidean distance (higher = more similar)
        distances = torch.cdist(query_embeddings, doc_embeddings, p=2)
        return -distances

    else:
        raise ValueError(f"Unknown similarity type: {similarity_type}")


def batch_encode_texts(texts: List[str],
                       model,
                       batch_size: int = 32,
                       device: str = None,
                       normalize: bool = True,
                       show_progress: bool = True) -> torch.Tensor:
    """
    Convenience function for batch encoding texts.

    Args:
        texts: List of texts to encode
        model: Model with encode method
        batch_size: Batch size
        device: Device to use
        normalize: Whether to normalize embeddings
        show_progress: Whether to show progress

    Returns:
        Tensor of embeddings
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    all_embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding")

    with torch.no_grad():
        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            if hasattr(model, 'encode_documents'):
                batch_embs = model.encode_documents(batch_texts)
            elif hasattr(model, 'encode'):
                batch_embs = model.encode(batch_texts)
            else:
                raise ValueError("Model must have encode method")

            if normalize:
                batch_embs = F.normalize(batch_embs, p=2, dim=1)

            all_embeddings.append(batch_embs.cpu())

    return torch.cat(all_embeddings, dim=0)


def create_embedding_manager(embedding_dim: int,
                             storage_format: str = 'hdf5',
                             normalize: bool = True,
                             device: str = None) -> EmbeddingManager:
    """Factory function to create embedding manager."""
    return EmbeddingManager(
        embedding_dim=embedding_dim,
        storage_format=storage_format,
        normalize_embeddings=normalize,
        device=device
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing embedding utilities...")

    # Test embedding manager
    manager = create_embedding_manager(embedding_dim=384)

    # Create dummy embeddings
    dummy_embeddings = torch.randn(100, 384)
    dummy_ids = [f"doc_{i}" for i in range(100)]

    # Test saving/loading
    test_path = Path("test_embeddings.hdf5")
    manager.save_embeddings(dummy_embeddings, dummy_ids, test_path)

    loaded_embs, loaded_ids, metadata = manager.load_embeddings(test_path)

    print(f"Original shape: {dummy_embeddings.shape}")
    print(f"Loaded shape: {loaded_embs.shape}")
    print(f"IDs match: {dummy_ids == loaded_ids}")
    print(f"Metadata: {metadata}")

    # Cleanup
    test_path.unlink()

    # Test similarity computation
    query_embs = torch.randn(5, 384)
    doc_embs = torch.randn(10, 384)

    sim_matrix = compute_similarity_matrix(query_embs, doc_embs, 'cosine')
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")

    print("Embedding utilities test completed!")