#!/usr/bin/env python3
"""
Vector-based Pseudo Relevance Feedback (PRF) Reranker

A modular system to rerank TREC run files using vector-based PRF.
Uses ir_datasets for document/query handling and dense models for embeddings.
"""

import os
import json
import pickle
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import torch
import ir_datasets
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseModelHandler:
    """Handles loading and encoding with dense retrieval models."""

    def __init__(self, model_name: str, device: str = 'cuda', max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading dense model: {model_name}")

        # Check if it's a sentence-transformers model
        if self._is_sentence_transformer(model_name):
            self._load_sentence_transformer(model_name)
        else:
            self._load_huggingface_model(model_name)

        logger.info(f"Model loaded on {self.device}")

    def _is_sentence_transformer(self, model_name: str) -> bool:
        """Check if model is a sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            # Try to load as sentence transformer
            temp_model = SentenceTransformer(model_name, device='cpu')
            del temp_model
            return True
        except:
            return False

    def _load_sentence_transformer(self, model_name: str):
        """Load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=self.device)
            self.tokenizer = None  # Not needed for sentence-transformers
            self.model_type = 'sentence_transformer'
            logger.info(f"Loaded as sentence-transformers model")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to HuggingFace")
            self._load_huggingface_model(model_name)

    def _load_huggingface_model(self, model_name: str):
        """Load standard HuggingFace model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = 'huggingface'
        logger.info(f"Loaded as HuggingFace transformer model")

    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into dense vectors."""
        if not texts:
            return np.array([]).reshape(0, -1)

        if self.model_type == 'sentence_transformer':
            return self._encode_with_sentence_transformer(texts, batch_size, show_progress)
        else:
            return self._encode_with_huggingface(texts, batch_size, show_progress)

    def _encode_with_sentence_transformer(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Encode using sentence-transformers (optimized)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def _encode_with_huggingface(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Encode using standard HuggingFace transformers."""
        embeddings = []

        progress_bar = tqdm(range(0, len(texts), batch_size),
                            desc="Encoding texts", disable=not show_progress)

        with torch.no_grad():
            for i in progress_bar:
                batch_texts = texts[i:i + batch_size]

                try:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get embeddings
                    outputs = self.model(**inputs)

                    # Try multiple pooling strategies
                    batch_embeddings = self._extract_embeddings(outputs, inputs.get('attention_mask'))
                    embeddings.append(batch_embeddings.cpu().numpy())

                except Exception as e:
                    logger.warning(f"Failed to encode batch {i // batch_size}: {e}")
                    # Create zero embeddings as fallback
                    fallback_emb = torch.zeros(len(batch_texts), self.model.config.hidden_size)
                    embeddings.append(fallback_emb.numpy())

        return np.vstack(embeddings) if embeddings else np.array([]).reshape(0, -1)

    def _extract_embeddings(self, outputs, attention_mask):
        """Extract embeddings using the best available method."""
        # Strategy 1: Use pooler output if available
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output

        # Strategy 2: Use [CLS] token
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Strategy 3: Mean pooling
        if hasattr(outputs, 'last_hidden_state') and attention_mask is not None:
            return self._mean_pooling(outputs.last_hidden_state, attention_mask)

        # Fallback: just return first hidden state
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)

        raise ValueError("Could not extract embeddings from model outputs")

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings."""
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Sum embeddings and divide by actual length
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode_texts([query_text], batch_size=1, show_progress=False)[0]

    def encode_documents(self, doc_texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple documents."""
        return self.encode_texts(doc_texts, batch_size=batch_size, show_progress=True)


class VectorFusion:
    """Implements vector fusion methods for PRF."""

    @staticmethod
    def average_fusion(query_embedding: np.ndarray, feedback_embeddings: np.ndarray) -> np.ndarray:
        """
        Average fusion: new_query = avg(query, feedback_docs)

        Args:
            query_embedding: Query vector (1D array)
            feedback_embeddings: Feedback document vectors (2D array: num_docs x embedding_dim)
        """
        if len(feedback_embeddings) == 0:
            return query_embedding

        # Combine query with feedback documents
        all_embeddings = np.vstack([query_embedding.reshape(1, -1), feedback_embeddings])
        return np.mean(all_embeddings, axis=0)

    @staticmethod
    def rocchio_fusion(query_embedding: np.ndarray,
                       feedback_embeddings: np.ndarray,
                       alpha: float = 1.0,
                       beta: float = 0.5) -> np.ndarray:
        """
        Rocchio fusion: new_query = α * query + β * avg(feedback_docs)

        Args:
            query_embedding: Query vector (1D array)
            feedback_embeddings: Feedback document vectors (2D array)
            alpha: Weight for original query
            beta: Weight for feedback documents
        """
        if len(feedback_embeddings) == 0:
            return query_embedding

        feedback_centroid = np.mean(feedback_embeddings, axis=0)
        return alpha * query_embedding + beta * feedback_centroid


class SimilarityComputer:
    """Computes similarities between query and document vectors."""

    @staticmethod
    def compute_similarities(query_embedding: np.ndarray,
                             doc_embeddings: np.ndarray,
                             metric: str = "dot_product") -> np.ndarray:
        """
        Compute similarities between query and documents.

        Args:
            query_embedding: Query vector (1D array)
            doc_embeddings: Document vectors (2D array: num_docs x embedding_dim)
            metric: Similarity metric ('dot_product', 'cosine')
        """
        if len(doc_embeddings) == 0:
            return np.array([])

        if metric == "dot_product":
            similarities = np.dot(doc_embeddings, query_embedding)
        elif metric == "cosine":
            # Normalize vectors
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(doc_norms, query_norm)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return similarities


class TRECRunLoader:
    """Loads and processes TREC run files."""

    def __init__(self, run_path: str):
        self.run_path = run_path
        self.run_data = self._load_run()

    def _load_run(self) -> Dict[str, List[Tuple[str, float]]]:
        """Load TREC run file into query_id -> [(doc_id, score), ...] format."""
        run_data = defaultdict(list)

        with open(self.run_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, rank, score, _ = parts[:6]
                    run_data[qid].append((docid, float(score)))
                elif len(parts) >= 3:
                    qid, docid, rank = parts[:3]
                    run_data[qid].append((docid, float(rank)))

        # Sort by score descending for each query
        for qid in run_data:
            run_data[qid].sort(key=lambda x: x[1], reverse=True)

        return dict(run_data)

    def get_query_docs(self, query_id: str, top_k: int = None) -> List[str]:
        """Get document IDs for a query, optionally limited to top_k."""
        if query_id not in self.run_data:
            return []

        docs = [doc_id for doc_id, _ in self.run_data[query_id]]
        return docs[:top_k] if top_k else docs


class IRDatasetHandler:
    """Handles ir_datasets integration for any dataset."""

    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

        try:
            self.dataset = ir_datasets.load(dataset_name)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        # Initialize caches
        self.doc_cache = {}
        self.query_cache = {}
        self._build_caches()

    def _build_caches(self):
        """Build document and query caches."""
        self._build_doc_cache()
        self._build_query_cache()

    def _build_doc_cache(self):
        """Build document ID to text cache."""
        logger.info("Building document cache...")

        if not hasattr(self.dataset, 'docs_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have documents")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_docs.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.doc_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.doc_cache)} documents from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load doc cache: {e}")

        # Build cache from dataset
        doc_count = 0
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents"):
            doc_text = self._extract_document_text(doc)
            if doc_text.strip():
                self.doc_cache[doc.doc_id] = doc_text
                doc_count += 1

        logger.info(f"Loaded {doc_count} documents")

        # Save to cache
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.doc_cache, f)
                logger.info(f"Saved document cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save doc cache: {e}")

    def _build_query_cache(self):
        """Build query ID to text cache."""
        logger.info("Building query cache...")

        if not hasattr(self.dataset, 'queries_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have queries")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_queries.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.query_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.query_cache)} queries from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load query cache: {e}")

        # Build cache from dataset
        query_count = 0
        for query in self.dataset.queries_iter():
            query_text = self._extract_query_text(query)
            if query_text.strip():
                self.query_cache[query.query_id] = query_text
                query_count += 1

        logger.info(f"Loaded {query_count} queries")

        # Save to cache
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.query_cache, f)
                logger.info(f"Saved query cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save query cache: {e}")

    def _extract_document_text(self, doc) -> str:
        """Extract text from document."""
        text_parts = []
        text_fields = ['title', 'text', 'body', 'content', 'abstract', 'summary']

        for field in text_fields:
            if hasattr(doc, field):
                field_value = getattr(doc, field)
                if field_value and str(field_value).strip():
                    text_parts.append(str(field_value).strip())

        return " ".join(text_parts) if text_parts else ""

    def _extract_query_text(self, query) -> str:
        """Extract text from query."""
        text_fields = ['text', 'title', 'query', 'description', 'narrative']

        for field in text_fields:
            if hasattr(query, field):
                field_value = getattr(query, field)
                if field_value and str(field_value).strip():
                    return str(field_value).strip()

        return ""

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get document text by ID."""
        return self.doc_cache.get(doc_id)

    def get_query_text(self, query_id: str) -> Optional[str]:
        """Get query text by ID."""
        return self.query_cache.get(query_id)

    def get_documents_text(self, doc_ids: List[str]) -> List[str]:
        """Get multiple document texts."""
        return [self.doc_cache.get(doc_id, "") for doc_id in doc_ids]


class VectorPRFReranker:
    """Main Vector PRF reranking system."""

    def __init__(self,
                 model_name: str,
                 dataset_name: str,
                 device: str = 'cuda',
                 fusion_method: str = 'rocchio',
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 similarity_metric: str = 'dot_product',
                 cache_dir: Optional[str] = None):

        self.fusion_method = fusion_method
        self.alpha = alpha
        self.beta = beta
        self.similarity_metric = similarity_metric

        # Initialize components
        self.dense_model = DenseModelHandler(model_name, device)
        self.dataset_handler = IRDatasetHandler(dataset_name, cache_dir)
        self.vector_fusion = VectorFusion()
        self.similarity_computer = SimilarityComputer()

        logger.info(f"VectorPRFReranker initialized with {model_name}")

    def rerank_run(self,
                   run_path: str,
                   output_path: str,
                   num_feedback: int = 3,
                   rerank_depth: int = 1000) -> None:
        """Rerank a TREC run file using vector-based PRF."""

        # Load run file
        run_loader = TRECRunLoader(run_path)
        logger.info(f"Loaded run file with {len(run_loader.run_data)} queries")

        # Process each query
        reranked_results = {}

        for qid in tqdm(run_loader.run_data.keys(), desc="Processing queries"):
            # Get query text
            query_text = self.dataset_handler.get_query_text(qid)
            if not query_text:
                logger.warning(f"Query {qid} not found in dataset")
                continue

            # Get candidate documents
            candidate_doc_ids = run_loader.get_query_docs(qid, rerank_depth)
            if not candidate_doc_ids:
                logger.warning(f"No candidate documents for query {qid}")
                continue

            # Get document texts
            doc_texts = self.dataset_handler.get_documents_text(candidate_doc_ids)

            # Filter out empty documents
            valid_docs = []
            valid_doc_ids = []
            for doc_id, doc_text in zip(candidate_doc_ids, doc_texts):
                if doc_text.strip():
                    valid_docs.append(doc_text)
                    valid_doc_ids.append(doc_id)

            if not valid_docs:
                logger.warning(f"No valid documents for query {qid}")
                continue

            # Encode query and documents
            query_embedding = self.dense_model.encode_query(query_text)
            doc_embeddings = self.dense_model.encode_documents(valid_docs)

            # Get PRF documents (top-k from original ranking)
            prf_doc_count = min(num_feedback, len(valid_docs))
            prf_embeddings = doc_embeddings[:prf_doc_count]

            # Apply vector fusion
            if self.fusion_method == 'average':
                new_query_embedding = self.vector_fusion.average_fusion(
                    query_embedding, prf_embeddings
                )
            elif self.fusion_method == 'rocchio':
                new_query_embedding = self.vector_fusion.rocchio_fusion(
                    query_embedding, prf_embeddings, self.alpha, self.beta
                )
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            # Compute similarities with new query
            similarities = self.similarity_computer.compute_similarities(
                new_query_embedding, doc_embeddings, self.similarity_metric
            )

            # Rank by similarity (highest first)
            ranked_indices = np.argsort(similarities)[::-1]

            # Store results
            reranked_results[qid] = [
                (valid_doc_ids[idx], similarities[idx])
                for idx in ranked_indices
            ]

        # Save results
        self._save_results(reranked_results, output_path)
        logger.info(f"Reranked results saved to {output_path}")

    def _save_results(self, results: Dict[str, List[Tuple[str, float]]], output_path: str):
        """Save results in TREC format."""
        with open(output_path, 'w') as f:
            for qid, doc_scores in results.items():
                for rank, (doc_id, score) in enumerate(doc_scores, 1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} VectorPRF\n")


def main():
    parser = argparse.ArgumentParser(description="Vector-based PRF Reranker")
    parser.add_argument("--model", required=True,
                        help="Dense model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--dataset", required=True,
                        help="IR dataset name (e.g., 'msmarco-passage/dev')")
    parser.add_argument("--run", required=True,
                        help="Path to input TREC run file")
    parser.add_argument("--output", required=True,
                        help="Path to output reranked file")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for encoding")
    parser.add_argument("--fusion-method", default="rocchio", choices=["average", "rocchio"],
                        help="Vector fusion method")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Rocchio alpha parameter (query weight)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Rocchio beta parameter (feedback weight)")
    parser.add_argument("--similarity", default="dot_product", choices=["dot_product", "cosine"],
                        help="Similarity metric")
    parser.add_argument("--num-feedback", type=int, default=3,
                        help="Number of feedback documents for PRF")
    parser.add_argument("--rerank-depth", type=int, default=1000,
                        help="Number of top docs to rerank")
    parser.add_argument("--cache-dir",
                        help="Directory to cache dataset files")

    args = parser.parse_args()

    # Initialize reranker
    reranker = VectorPRFReranker(
        model_name=args.model,
        dataset_name=args.dataset,
        device=args.device,
        fusion_method=args.fusion_method,
        alpha=args.alpha,
        beta=args.beta,
        similarity_metric=args.similarity,
        cache_dir=args.cache_dir
    )

    # Perform reranking
    reranker.rerank_run(
        run_path=args.run,
        output_path=args.output,
        num_feedback=args.num_feedback,
        rerank_depth=args.rerank_depth
    )

    print(f"Vector PRF reranking completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()