"""
Dense Indexing Utilities for Bi-Encoder Systems

This module provides utilities for building and managing FAISS indices
for efficient dense retrieval, along with fallback exact search methods.
"""

import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Only exact search will be supported.")

logger = logging.getLogger(__name__)


class DenseIndex:
    """
    Dense index supporting both FAISS and exact search.

    Provides a unified interface for dense retrieval with automatic
    fallback to exact search when FAISS is not available.
    """

    def __init__(self,
                 embedding_dim: int,
                 index_type: str = "IndexFlatIP",
                 use_gpu: bool = False,
                 normalize_embeddings: bool = True,
                 exact_search_fallback: bool = True):
        """
        Initialize dense index.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: FAISS index type ("IndexFlatIP", "IndexIVFFlat", etc.)
            use_gpu: Whether to use GPU for FAISS (if available)
            normalize_embeddings: Whether to normalize embeddings
            exact_search_fallback: Whether to use exact search if FAISS fails
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.normalize_embeddings = normalize_embeddings
        self.exact_search_fallback = exact_search_fallback

        # Index components
        self.faiss_index = None
        self.embeddings = None  # For exact search fallback
        self.document_ids = None
        self.gpu_resource = None

        # Statistics
        self.num_vectors = 0
        self.is_trained = False
        self.build_time = 0.0

        logger.info(f"DenseIndex initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Index type: {index_type}")
        logger.info(f"  Use GPU: {self.use_gpu}")
        logger.info(f"  Normalize: {normalize_embeddings}")
        logger.info(f"  FAISS available: {FAISS_AVAILABLE}")

    def build_index(self,
                    embeddings: Union[torch.Tensor, np.ndarray],
                    document_ids: List[str],
                    train_size: Optional[int] = None) -> None:
        """
        Build the dense index from embeddings.

        Args:
            embeddings: Document embeddings [num_docs, embedding_dim]
            document_ids: List of document IDs
            train_size: Number of vectors to use for training (for IVF indices)
        """
        start_time = time.time()

        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().numpy().astype('float32')
        else:
            embeddings_np = embeddings.astype('float32')

        # Normalize if requested
        if self.normalize_embeddings:
            embeddings_np = self._normalize_numpy(embeddings_np)

        self.embeddings = embeddings_np
        self.document_ids = document_ids
        self.num_vectors = len(embeddings_np)

        logger.info(f"Building index for {self.num_vectors} vectors...")

        # Try to build FAISS index
        if FAISS_AVAILABLE:
            try:
                self._build_faiss_index(embeddings_np, train_size)
                logger.info("FAISS index built successfully")
            except Exception as e:
                logger.warning(f"FAISS index building failed: {e}")
                if not self.exact_search_fallback:
                    raise
                logger.info("Falling back to exact search")
        else:
            if not self.exact_search_fallback:
                raise RuntimeError("FAISS not available and exact search fallback disabled")
            logger.info("Using exact search (FAISS not available)")

        self.build_time = time.time() - start_time
        logger.info(f"Index building completed in {self.build_time:.2f}s")

    def _build_faiss_index(self, embeddings_np: np.ndarray, train_size: Optional[int]):
        """Build FAISS index."""
        # Create FAISS index
        if self.index_type == "IndexFlatIP":
            # Flat inner product index (exact search)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == "IndexFlatL2":
            # Flat L2 distance index (exact search)
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        elif self.index_type == "IndexIVFFlat":
            # IVF with flat quantizer (approximate search)
            nlist = min(4096, max(1, self.num_vectors // 39))  # Common heuristic
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

        elif self.index_type == "IndexIVFPQ":
            # IVF with product quantization (memory efficient)
            nlist = min(4096, max(1, self.num_vectors // 39))
            m = 8  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)

        elif self.index_type == "IndexHNSW":
            # Hierarchical Navigable Small World (good for high-dimensional data)
            M = 32  # Number of connections per node
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.hnsw.efSearch = 128

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # Train index if needed
        if not self.faiss_index.is_trained:
            logger.info("Training FAISS index...")
            if train_size and train_size < len(embeddings_np):
                # Use subset for training
                train_indices = np.random.choice(len(embeddings_np), train_size, replace=False)
                train_vectors = embeddings_np[train_indices]
            else:
                train_vectors = embeddings_np

            self.faiss_index.train(train_vectors)
            self.is_trained = True

        # Move to GPU if requested
        if self.use_gpu:
            try:
                self.gpu_resource = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.faiss_index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}")

        # Add vectors to index
        logger.info("Adding vectors to FAISS index...")
        self.faiss_index.add(embeddings_np)

        # Set search parameters for IVF indices
        if "IVF" in self.index_type and hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = min(128, max(1, self.faiss_index.nlist // 4))

    def search(self,
               query_embeddings: Union[torch.Tensor, np.ndarray],
               top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for top-k nearest neighbors.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            top_k: Number of top results to return

        Returns:
            Tuple of (scores, indices) arrays
        """
        # Convert to numpy if needed
        if isinstance(query_embeddings, torch.Tensor):
            query_np = query_embeddings.cpu().numpy().astype('float32')
        else:
            query_np = query_embeddings.astype('float32')

        # Normalize if requested
        if self.normalize_embeddings:
            query_np = self._normalize_numpy(query_np)

        # Ensure 2D shape
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        # Use FAISS if available
        if self.faiss_index is not None:
            return self._faiss_search(query_np, top_k)
        else:
            return self._exact_search(query_np, top_k)

    def _faiss_search(self, query_np: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        scores, indices = self.faiss_index.search(query_np, top_k)
        return scores, indices

    def _exact_search(self, query_np: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Exact search using numpy operations."""
        if self.embeddings is None:
            raise ValueError("No embeddings available for exact search")

        # Compute similarities
        if self.normalize_embeddings:
            # Cosine similarity (embeddings already normalized)
            similarities = np.dot(query_np, self.embeddings.T)
        else:
            # Dot product similarity
            similarities = np.dot(query_np, self.embeddings.T)

        # Get top-k indices
        top_k = min(top_k, similarities.shape[1])
        top_indices = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]

        # Sort within top-k
        sorted_indices = np.argsort(similarities[np.arange(similarities.shape[0])[:, None], top_indices], axis=1)[:,
                         ::-1]
        indices = top_indices[np.arange(similarities.shape[0])[:, None], sorted_indices]

        # Get corresponding scores
        scores = similarities[np.arange(similarities.shape[0])[:, None], indices]

        return scores, indices

    def _normalize_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize numpy embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def search_batch(self,
                     query_embeddings: Union[torch.Tensor, np.ndarray],
                     top_k: int = 100,
                     batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search in batches for memory efficiency.

        Args:
            query_embeddings: Query embeddings
            top_k: Number of top results per query
            batch_size: Batch size for processing

        Returns:
            Tuple of (scores, indices) arrays
        """
        # Convert to numpy if needed
        if isinstance(query_embeddings, torch.Tensor):
            query_np = query_embeddings.cpu().numpy().astype('float32')
        else:
            query_np = query_embeddings.astype('float32')

        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        num_queries = query_np.shape[0]
        all_scores = []
        all_indices = []

        for i in tqdm(range(0, num_queries, batch_size), desc="Batch search"):
            batch_queries = query_np[i:i + batch_size]
            batch_scores, batch_indices = self.search(batch_queries, top_k)
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)

        return np.vstack(all_scores), np.vstack(all_indices)

    def get_documents_by_indices(self, indices: np.ndarray) -> List[List[str]]:
        """
        Get document IDs corresponding to indices.

        Args:
            indices: Array of document indices [num_queries, top_k]

        Returns:
            List of lists of document IDs
        """
        if self.document_ids is None:
            raise ValueError("No document IDs available")

        results = []
        for query_indices in indices:
            query_docs = []
            for idx in query_indices:
                if 0 <= idx < len(self.document_ids):
                    query_docs.append(self.document_ids[idx])
                else:
                    query_docs.append("")  # Invalid index
            results.append(query_docs)

        return results

    def save_index(self, save_path: Path, save_embeddings: bool = True) -> None:
        """
        Save the index to disk.

        Args:
            save_path: Directory to save index files
            save_embeddings: Whether to save embeddings for exact search fallback
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index if available
        if self.faiss_index is not None:
            faiss_path = save_path / "faiss_index.bin"
            # Move to CPU before saving if on GPU
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
                faiss.write_index(cpu_index, str(faiss_path))
            else:
                faiss.write_index(self.faiss_index, str(faiss_path))
            logger.info(f"FAISS index saved to: {faiss_path}")

        # Save embeddings if requested
        if save_embeddings and self.embeddings is not None:
            embeddings_path = save_path / "embeddings.npy"
            np.save(embeddings_path, self.embeddings)
            logger.info(f"Embeddings saved to: {embeddings_path}")

        # Save document IDs
        if self.document_ids is not None:
            ids_path = save_path / "document_ids.json"
            with open(ids_path, 'w') as f:
                json.dump(self.document_ids, f)
            logger.info(f"Document IDs saved to: {ids_path}")

        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'normalize_embeddings': self.normalize_embeddings,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained,
            'build_time': self.build_time,
            'faiss_available': self.faiss_index is not None,
            'exact_search_available': self.embeddings is not None
        }

        metadata_path = save_path / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Index saved to: {save_path}")

    def load_index(self, load_path: Path, load_embeddings: bool = True) -> None:
        """
        Load the index from disk.

        Args:
            load_path: Directory containing index files
            load_embeddings: Whether to load embeddings for exact search
        """
        load_path = Path(load_path)

        # Load metadata
        metadata_path = load_path / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata['index_type']
            self.normalize_embeddings = metadata['normalize_embeddings']
            self.num_vectors = metadata['num_vectors']
            self.is_trained = metadata['is_trained']
            self.build_time = metadata.get('build_time', 0.0)

        # Load FAISS index if available
        faiss_path = load_path / "faiss_index.bin"
        if FAISS_AVAILABLE and faiss_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))

                # Move to GPU if requested
                if self.use_gpu:
                    self.gpu_resource = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, self.faiss_index)

                logger.info(f"FAISS index loaded from: {faiss_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self.faiss_index = None

        # Load embeddings if requested
        embeddings_path = load_path / "embeddings.npy"
        if load_embeddings and embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Embeddings loaded from: {embeddings_path}")

        # Load document IDs
        ids_path = load_path / "document_ids.json"
        if ids_path.exists():
            with open(ids_path, 'r') as f:
                self.document_ids = json.load(f)
            logger.info(f"Document IDs loaded from: {ids_path}")

        logger.info(f"Index loaded from: {load_path}")

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the index."""
        info = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'num_vectors': self.num_vectors,
            'normalize_embeddings': self.normalize_embeddings,
            'is_trained': self.is_trained,
            'build_time': self.build_time,
            'faiss_available': self.faiss_index is not None,
            'exact_search_available': self.embeddings is not None,
            'use_gpu': self.use_gpu
        }

        # Add FAISS-specific info
        if self.faiss_index is not None:
            info['faiss_ntotal'] = self.faiss_index.ntotal
            if hasattr(self.faiss_index, 'nprobe'):
                info['faiss_nprobe'] = self.faiss_index.nprobe
            if hasattr(self.faiss_index, 'nlist'):
                info['faiss_nlist'] = self.faiss_index.nlist

        return info


class IndexManager:
    """
    Manager for multiple dense indices.

    Useful for managing different index types or versions.
    """

    def __init__(self):
        """Initialize index manager."""
        self.indices = {}
        logger.info("IndexManager initialized")

    def add_index(self, name: str, index: DenseIndex) -> None:
        """Add an index to the manager."""
        self.indices[name] = index
        logger.info(f"Added index '{name}' to manager")

    def get_index(self, name: str) -> Optional[DenseIndex]:
        """Get an index by name."""
        return self.indices.get(name)

    def remove_index(self, name: str) -> bool:
        """Remove an index from the manager."""
        if name in self.indices:
            del self.indices[name]
            logger.info(f"Removed index '{name}' from manager")
            return True
        return False

    def list_indices(self) -> List[str]:
        """List all index names."""
        return list(self.indices.keys())

    def get_index_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific index."""
        index = self.indices.get(name)
        if index:
            return index.get_index_info()
        return None

    def save_all_indices(self, base_path: Path) -> None:
        """Save all indices to disk."""
        base_path = Path(base_path)
        for name, index in self.indices.items():
            index_path = base_path / name
            index.save_index(index_path)
        logger.info(f"Saved {len(self.indices)} indices to: {base_path}")

    def load_indices_from_directory(self, base_path: Path) -> None:
        """Load all indices from a directory."""
        base_path = Path(base_path)
        if not base_path.exists():
            logger.warning(f"Directory does not exist: {base_path}")
            return

        for index_dir in base_path.iterdir():
            if index_dir.is_dir():
                try:
                    index = DenseIndex(embedding_dim=384)  # Will be updated from metadata
                    index.load_index(index_dir)
                    self.indices[index_dir.name] = index
                    logger.info(f"Loaded index '{index_dir.name}'")
                except Exception as e:
                    logger.warning(f"Failed to load index from {index_dir}: {e}")


def create_dense_index(embedding_dim: int,
                       index_type: str = "IndexFlatIP",
                       use_gpu: bool = False,
                       normalize: bool = True) -> DenseIndex:
    """
    Factory function to create a dense index.

    Args:
        embedding_dim: Dimension of embeddings
        index_type: FAISS index type
        use_gpu: Whether to use GPU
        normalize: Whether to normalize embeddings

    Returns:
        Configured DenseIndex
    """
    return DenseIndex(
        embedding_dim=embedding_dim,
        index_type=index_type,
        use_gpu=use_gpu,
        normalize_embeddings=normalize
    )


def recommend_index_type(num_vectors: int,
                         embedding_dim: int,
                         memory_constraint: bool = False) -> str:
    """
    Recommend FAISS index type based on data characteristics.

    Args:
        num_vectors: Number of vectors to index
        embedding_dim: Embedding dimension
        memory_constraint: Whether memory usage is a constraint

    Returns:
        Recommended index type
    """
    if num_vectors < 10000:
        # Small dataset - use exact search
        return "IndexFlatIP"
    elif num_vectors < 100000:
        # Medium dataset - use IVF
        return "IndexIVFFlat"
    elif memory_constraint:
        # Large dataset with memory constraints - use PQ compression
        return "IndexIVFPQ"
    elif embedding_dim > 512:
        # High dimensional data - use HNSW
        return "IndexHNSW"
    else:
        # Large dataset - use IVF
        return "IndexIVFFlat"


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing dense indexing utilities...")

    # Create test data
    embedding_dim = 384
    num_docs = 1000

    # Generate random embeddings
    embeddings = np.random.randn(num_docs, embedding_dim).astype('float32')
    document_ids = [f"doc_{i}" for i in range(num_docs)]

    # Test index creation
    index = create_dense_index(
        embedding_dim=embedding_dim,
        index_type="IndexFlatIP",
        normalize=True
    )

    # Build index
    index.build_index(embeddings, document_ids)

    # Test search
    query_embeddings = np.random.randn(5, embedding_dim).astype('float32')
    scores, indices = index.search(query_embeddings, top_k=10)

    print(f"Search results shape: {scores.shape}, {indices.shape}")
    print(f"Top scores for first query: {scores[0][:5]}")

    # Test getting document IDs
    doc_results = index.get_documents_by_indices(indices)
    print(f"First query results: {doc_results[0][:3]}")

    # Test index info
    info = index.get_index_info()
    print(f"Index info: {info}")

    # Test index manager
    manager = IndexManager()
    manager.add_index("test_index", index)

    print(f"Index manager indices: {manager.list_indices()}")

    # Test recommendation
    recommended = recommend_index_type(num_vectors=50000, embedding_dim=384)
    print(f"Recommended index type for 50k vectors: {recommended}")

    print("Dense indexing utilities test completed!")