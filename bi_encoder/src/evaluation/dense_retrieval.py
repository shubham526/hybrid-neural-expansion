"""
Dense Retrieval Evaluation for Bi-Encoder Models

This module provides comprehensive evaluation capabilities for dense retrieval
using bi-encoder models, including end-to-end retrieval evaluation and
comparison with BM25 baselines.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from collections import defaultdict
from tqdm import tqdm
import tempfile
import os

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Dense indexing will use slower exact search.")

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retrieval engine using bi-encoder models.

    Supports both exact search and approximate search with FAISS.
    """

    def __init__(self,
                 bi_encoder_model,
                 use_faiss: bool = True,
                 faiss_index_type: str = "IndexFlatIP",
                 normalize_embeddings: bool = True):
        """
        Initialize dense retriever.

        Args:
            bi_encoder_model: Trained bi-encoder model
            use_faiss: Whether to use FAISS for approximate search
            faiss_index_type: Type of FAISS index ("IndexFlatIP", "IndexIVFFlat", etc.)
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
        """
        self.model = bi_encoder_model
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_type = faiss_index_type
        self.normalize_embeddings = normalize_embeddings

        # Index storage
        self.document_embeddings = None
        self.document_ids = None
        self.faiss_index = None

        logger.info(f"DenseRetriever initialized:")
        logger.info(f"  FAISS enabled: {self.use_faiss}")
        logger.info(f"  Index type: {faiss_index_type}")
        logger.info(f"  Normalize embeddings: {normalize_embeddings}")

    def index_documents(self,
                        documents: List[str],
                        document_ids: List[str],
                        batch_size: int = 32,
                        show_progress: bool = True) -> None:
        """
        Index a collection of documents.

        Args:
            documents: List of document texts
            document_ids: List of document IDs
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        logger.info(f"Indexing {len(documents)} documents...")

        # Encode documents in batches
        all_embeddings = []

        iterator = range(0, len(documents), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents")

        self.model.eval()
        with torch.no_grad():
            for i in iterator:
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = self.model.encode_documents(batch_docs)

                # Move to CPU and normalize if requested
                batch_embeddings = batch_embeddings.cpu()
                if self.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        self.document_embeddings = torch.cat(all_embeddings, dim=0)
        self.document_ids = document_ids

        logger.info(f"Document encoding completed. Shape: {self.document_embeddings.shape}")

        # Build FAISS index if enabled
        if self.use_faiss:
            self._build_faiss_index()

        logger.info("Document indexing completed")

    def _build_faiss_index(self):
        """Build FAISS index from document embeddings."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return

        logger.info(f"Building FAISS index: {self.faiss_index_type}")

        # Convert to numpy
        embeddings_np = self.document_embeddings.numpy().astype('float32')
        embedding_dim = embeddings_np.shape[1]

        # Create FAISS index
        if self.faiss_index_type == "IndexFlatIP":
            # Inner product (cosine similarity for normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        elif self.faiss_index_type == "IndexFlatL2":
            # L2 distance
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        elif self.faiss_index_type.startswith("IndexIVF"):
            # Inverted file index for approximate search
            nlist = min(4096, len(self.document_ids) // 39)  # Common heuristic
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)

            # Train the index
            logger.info("Training FAISS index...")
            self.faiss_index.train(embeddings_np)
            self.faiss_index.nprobe = min(128, nlist // 4)  # Search parameter
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")

        # Add embeddings to index
        logger.info("Adding embeddings to FAISS index...")
        self.faiss_index.add(embeddings_np)

        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")

    def search(self,
               query: str,
               expansion_features: Dict[str, Dict[str, float]] = None,
               top_k: int = 100,
               return_scores: bool = True) -> List[Tuple[str, float]]:
        """
        Search for top-k documents for a query.

        Args:
            query: Query text
            expansion_features: Optional expansion features
            top_k: Number of results to return
            return_scores: Whether to return similarity scores

        Returns:
            List of (doc_id, score) tuples
        """
        if self.document_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Encode query
        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model.encode_query(query, expansion_features)

            # Normalize if needed
            if self.normalize_embeddings:
                query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

        # Search using FAISS or exact search
        if self.use_faiss and self.faiss_index is not None:
            return self._faiss_search(query_embedding, top_k, return_scores)
        else:
            return self._exact_search(query_embedding, top_k, return_scores)

    def _faiss_search(self, query_embedding: torch.Tensor, top_k: int,
                      return_scores: bool) -> List[Tuple[str, float]]:
        """Search using FAISS index."""
        # Convert to numpy
        query_np = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)

        # Search
        scores, indices = self.faiss_index.search(query_np, top_k)
        scores = scores[0]  # Remove batch dimension
        indices = indices[0]

        # Convert to results
        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            if idx < len(self.document_ids):  # Valid index
                doc_id = self.document_ids[idx]
                if return_scores:
                    results.append((doc_id, float(score)))
                else:
                    results.append((doc_id, 1.0 - i / top_k))  # Rank-based score

        return results

    def _exact_search(self, query_embedding: torch.Tensor, top_k: int,
                      return_scores: bool) -> List[Tuple[str, float]]:
        """Search using exact similarity computation."""
        # Compute similarities
        similarities = self.model.similarity_computer.compute_similarity(
            query_embedding, self.document_embeddings
        )

        # Get top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.document_ids)))

        # Convert to results
        results = []
        for score, idx in zip(top_scores, top_indices):
            doc_id = self.document_ids[idx.item()]
            if return_scores:
                results.append((doc_id, score.item()))
            else:
                results.append((doc_id, 1.0))

        return results

    def batch_search(self,
                     queries: List[str],
                     expansion_features_list: List[Dict[str, Dict[str, float]]] = None,
                     top_k: int = 100) -> Dict[int, List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.

        Args:
            queries: List of query texts
            expansion_features_list: List of expansion features (one per query)
            top_k: Number of results per query

        Returns:
            Dictionary mapping query index to results
        """
        results = {}

        for i, query in enumerate(tqdm(queries, desc="Searching queries")):
            expansion_features = None
            if expansion_features_list and i < len(expansion_features_list):
                expansion_features = expansion_features_list[i]

            query_results = self.search(query, expansion_features, top_k)
            results[i] = query_results

        return results

    def save_index(self, index_path: Path):
        """Save the document index to disk."""
        index_path = Path(index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings and metadata
        torch.save({
            'document_embeddings': self.document_embeddings,
            'document_ids': self.document_ids,
            'normalize_embeddings': self.normalize_embeddings,
            'faiss_index_type': self.faiss_index_type
        }, index_path / 'index_data.pt')

        # Save FAISS index if available
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_path / 'faiss_index.bin'))

        logger.info(f"Index saved to: {index_path}")

    def load_index(self, index_path: Path):
        """Load document index from disk."""
        index_path = Path(index_path)

        # Load embeddings and metadata
        data = torch.load(index_path / 'index_data.pt', map_location='cpu')
        self.document_embeddings = data['document_embeddings']
        self.document_ids = data['document_ids']
        self.normalize_embeddings = data['normalize_embeddings']
        self.faiss_index_type = data['faiss_index_type']

        # Load FAISS index if available
        faiss_index_path = index_path / 'faiss_index.bin'
        if self.use_faiss and faiss_index_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_index_path))

        logger.info(f"Index loaded from: {index_path}")
        logger.info(f"Loaded {len(self.document_ids)} documents")


class DenseRetrievalEvaluator:
    """
    Comprehensive evaluator for dense retrieval systems.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator.

        Args:
            metrics: List of metrics to compute
        """
        if metrics is None:
            self.metrics = ['recall_10', 'recall_100', 'recall_1000',
                            'ndcg_cut_10', 'ndcg_cut_100', 'map']
        else:
            self.metrics = metrics

        logger.info(f"DenseRetrievalEvaluator initialized with metrics: {self.metrics}")

    def evaluate_retrieval(self,
                           retrieval_results: Dict[str, List[Tuple[str, float]]],
                           qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Evaluate retrieval results against relevance judgments.

        Args:
            retrieval_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            Dictionary of metric scores
        """
        from cross_encoder.src.evaluation.metrics import get_metric

        # Create temporary files for evaluation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.qrel') as qrel_file, \
                tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as run_file:

            try:
                # Write qrels
                self._write_qrels(qrels, qrel_file.name)

                # Write run
                self._write_run(retrieval_results, run_file.name)

                # Evaluate using existing metric function
                results = {}
                for metric in self.metrics:
                    try:
                        results[metric] = get_metric(qrel_file.name, run_file.name, metric)
                    except Exception as e:
                        logger.warning(f"Failed to compute {metric}: {e}")
                        results[metric] = 0.0

                return results

            finally:
                # Cleanup temporary files
                try:
                    os.unlink(qrel_file.name)
                    os.unlink(run_file.name)
                except OSError:
                    pass

    def _write_qrels(self, qrels: Dict[str, Dict[str, int]], filename: str):
        """Write qrels in TREC format."""
        with open(filename, 'w') as f:
            for query_id, docs in qrels.items():
                for doc_id, relevance in docs.items():
                    f.write(f"{query_id} 0 {doc_id} {relevance}\n")

    def _write_run(self, results: Dict[str, List[Tuple[str, float]]], filename: str):
        """Write results in TREC run format."""
        with open(filename, 'w') as f:
            for query_id, docs in results.items():
                for rank, (doc_id, score) in enumerate(docs, 1):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} dense_retrieval\n")

    def compare_with_baseline(self,
                              dense_results: Dict[str, List[Tuple[str, float]]],
                              baseline_results: Dict[str, List[Tuple[str, float]]],
                              qrels: Dict[str, Dict[str, int]],
                              baseline_name: str = "BM25") -> Dict[str, Any]:
        """
        Compare dense retrieval with baseline (e.g., BM25).

        Args:
            dense_results: Dense retrieval results
            baseline_results: Baseline retrieval results
            qrels: Relevance judgments
            baseline_name: Name of baseline system

        Returns:
            Comparison results
        """
        # Evaluate both systems
        dense_metrics = self.evaluate_retrieval(dense_results, qrels)
        baseline_metrics = self.evaluate_retrieval(baseline_results, qrels)

        # Compute improvements
        improvements = {}
        for metric in self.metrics:
            if metric in dense_metrics and metric in baseline_metrics:
                dense_score = dense_metrics[metric]
                baseline_score = baseline_metrics[metric]

                if baseline_score > 0:
                    improvement_pct = (dense_score - baseline_score) / baseline_score * 100
                    improvements[f"{metric}_improvement_pct"] = improvement_pct
                    improvements[f"{metric}_improvement_abs"] = dense_score - baseline_score
                else:
                    improvements[f"{metric}_improvement_pct"] = 0.0
                    improvements[f"{metric}_improvement_abs"] = dense_score

        return {
            'dense_retrieval': dense_metrics,
            baseline_name: baseline_metrics,
            'improvements': improvements,
            'comparison_summary': self._create_comparison_summary(dense_metrics, baseline_metrics, baseline_name)
        }

    def _create_comparison_summary(self, dense_metrics: Dict[str, float],
                                   baseline_metrics: Dict[str, float],
                                   baseline_name: str) -> str:
        """Create a formatted comparison summary."""
        summary = f"\nDense Retrieval vs {baseline_name} Comparison:\n"
        summary += "=" * 50 + "\n"

        for metric in self.metrics:
            if metric in dense_metrics and metric in baseline_metrics:
                dense_score = dense_metrics[metric]
                baseline_score = baseline_metrics[metric]
                improvement = dense_score - baseline_score
                improvement_pct = improvement / baseline_score * 100 if baseline_score > 0 else 0

                summary += f"{metric.upper():>12}: "
                summary += f"Dense={dense_score:.4f}, {baseline_name}={baseline_score:.4f}, "
                summary += f"Δ={improvement:+.4f} ({improvement_pct:+.1f}%)\n"

        return summary

    def evaluate_expansion_effectiveness(self,
                                         results_without_expansion: Dict[str, List[Tuple[str, float]]],
                                         results_with_expansion: Dict[str, List[Tuple[str, float]]],
                                         qrels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of query expansion.

        Args:
            results_without_expansion: Results using original queries only
            results_with_expansion: Results using expanded queries
            qrels: Relevance judgments

        Returns:
            Expansion effectiveness analysis
        """
        # Evaluate both approaches
        without_expansion = self.evaluate_retrieval(results_without_expansion, qrels)
        with_expansion = self.evaluate_retrieval(results_with_expansion, qrels)

        # Compute expansion improvements
        expansion_improvements = {}
        for metric in self.metrics:
            if metric in without_expansion and metric in with_expansion:
                baseline_score = without_expansion[metric]
                expanded_score = with_expansion[metric]

                if baseline_score > 0:
                    improvement_pct = (expanded_score - baseline_score) / baseline_score * 100
                    expansion_improvements[f"{metric}_improvement_pct"] = improvement_pct
                    expansion_improvements[f"{metric}_improvement_abs"] = expanded_score - baseline_score
                else:
                    expansion_improvements[f"{metric}_improvement_pct"] = 0.0
                    expansion_improvements[f"{metric}_improvement_abs"] = expanded_score

        return {
            'without_expansion': without_expansion,
            'with_expansion': with_expansion,
            'expansion_improvements': expansion_improvements,
            'expansion_summary': self._create_expansion_summary(without_expansion, with_expansion)
        }

    def _create_expansion_summary(self, without_expansion: Dict[str, float],
                                  with_expansion: Dict[str, float]) -> str:
        """Create expansion effectiveness summary."""
        summary = "\nQuery Expansion Effectiveness:\n"
        summary += "=" * 40 + "\n"

        for metric in self.metrics:
            if metric in without_expansion and metric in with_expansion:
                baseline_score = without_expansion[metric]
                expanded_score = with_expansion[metric]
                improvement = expanded_score - baseline_score
                improvement_pct = improvement / baseline_score * 100 if baseline_score > 0 else 0

                summary += f"{metric.upper():>12}: "
                summary += f"Baseline={baseline_score:.4f}, Expanded={expanded_score:.4f}, "
                summary += f"Δ={improvement:+.4f} ({improvement_pct:+.1f}%)\n"

        return summary


class EndToEndRetrievalEvaluator:
    """
    End-to-end evaluation pipeline for dense retrieval systems.
    """

    def __init__(self, bi_encoder_model, evaluator: DenseRetrievalEvaluator):
        """
        Initialize end-to-end evaluator.

        Args:
            bi_encoder_model: Trained bi-encoder model
            evaluator: Dense retrieval evaluator
        """
        self.model = bi_encoder_model
        self.evaluator = evaluator
        self.retriever = None

    def setup_retrieval_index(self,
                              documents: List[str],
                              document_ids: List[str],
                              use_faiss: bool = True,
                              batch_size: int = 32):
        """
        Setup retrieval index with documents.

        Args:
            documents: List of document texts
            document_ids: List of document IDs
            use_faiss: Whether to use FAISS indexing
            batch_size: Batch size for encoding
        """
        self.retriever = DenseRetriever(
            self.model,
            use_faiss=use_faiss
        )

        logger.info("Setting up retrieval index...")
        self.retriever.index_documents(documents, document_ids, batch_size)
        logger.info("Retrieval index setup completed")

    def run_retrieval_evaluation(self,
                                 queries: Dict[str, str],
                                 expansion_features: Dict[str, Dict] = None,
                                 qrels: Dict[str, Dict[str, int]] = None,
                                 top_k: int = 1000) -> Dict[str, Any]:
        """
        Run complete retrieval evaluation.

        Args:
            queries: {query_id: query_text}
            expansion_features: {query_id: expansion_features}
            qrels: {query_id: {doc_id: relevance}}
            top_k: Number of documents to retrieve per query

        Returns:
            Complete evaluation results
        """
        if self.retriever is None:
            raise ValueError("Retrieval index not setup. Call setup_retrieval_index() first.")

        logger.info(f"Running retrieval evaluation for {len(queries)} queries...")

        # Run retrieval
        retrieval_results = {}

        for query_id, query_text in tqdm(queries.items(), desc="Retrieving"):
            expansion_feat = expansion_features.get(query_id) if expansion_features else None
            results = self.retriever.search(query_text, expansion_feat, top_k)
            retrieval_results[query_id] = results

        # Evaluate if qrels provided
        evaluation_results = {}
        if qrels:
            logger.info("Evaluating retrieval results...")
            evaluation_results = self.evaluator.evaluate_retrieval(retrieval_results, qrels)

            logger.info("Evaluation Results:")
            for metric, score in evaluation_results.items():
                logger.info(f"  {metric}: {score:.4f}")

        return {
            'retrieval_results': retrieval_results,
            'evaluation_metrics': evaluation_results,
            'num_queries': len(queries),
            'top_k': top_k
        }

    def compare_expansion_strategies(self,
                                     queries: Dict[str, str],
                                     expansion_features: Dict[str, Dict],
                                     qrels: Dict[str, Dict[str, int]],
                                     top_k: int = 1000) -> Dict[str, Any]:
        """
        Compare retrieval with and without expansion.

        Args:
            queries: {query_id: query_text}
            expansion_features: {query_id: expansion_features}
            qrels: {query_id: {doc_id: relevance}}
            top_k: Number of documents to retrieve

        Returns:
            Comparison results
        """
        logger.info("Comparing expansion strategies...")

        # Retrieval without expansion
        logger.info("Running retrieval without expansion...")
        results_without = {}
        for query_id, query_text in tqdm(queries.items(), desc="No expansion"):
            results = self.retriever.search(query_text, None, top_k)
            results_without[query_id] = results

        # Retrieval with expansion
        logger.info("Running retrieval with expansion...")
        results_with = {}
        for query_id, query_text in tqdm(queries.items(), desc="With expansion"):
            expansion_feat = expansion_features.get(query_id)
            results = self.retriever.search(query_text, expansion_feat, top_k)
            results_with[query_id] = results

        # Compare results
        comparison = self.evaluator.evaluate_expansion_effectiveness(
            results_without, results_with, qrels
        )

        logger.info(comparison['expansion_summary'])

        return comparison


# Factory functions
def create_dense_retriever(bi_encoder_model,
                           use_faiss: bool = True,
                           faiss_index_type: str = "IndexFlatIP") -> DenseRetriever:
    """Create dense retriever with bi-encoder model."""
    return DenseRetriever(
        bi_encoder_model=bi_encoder_model,
        use_faiss=use_faiss,
        faiss_index_type=faiss_index_type
    )


def create_dense_retrieval_evaluator(metrics: List[str] = None) -> DenseRetrievalEvaluator:
    """Create dense retrieval evaluator."""
    return DenseRetrievalEvaluator(metrics=metrics)


def create_end_to_end_evaluator(bi_encoder_model,
                                metrics: List[str] = None) -> EndToEndRetrievalEvaluator:
    """Create end-to-end retrieval evaluator."""
    evaluator = create_dense_retrieval_evaluator(metrics)
    return EndToEndRetrievalEvaluator(bi_encoder_model, evaluator)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing dense retrieval evaluation...")

    # Mock data for testing
    mock_results = {
        'q1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
        'q2': [('doc4', 0.95), ('doc1', 0.85), ('doc5', 0.75)]
    }

    mock_qrels = {
        'q1': {'doc1': 2, 'doc2': 1, 'doc3': 0},
        'q2': {'doc4': 2, 'doc1': 0, 'doc5': 1}
    }

    # Test evaluator
    evaluator = create_dense_retrieval_evaluator(['recall_10', 'ndcg_cut_10'])

    try:
        results = evaluator.evaluate_retrieval(mock_results, mock_qrels)
        print(f"Evaluation results: {results}")
    except Exception as e:
        print(f"Evaluation test failed (expected without proper setup): {e}")

    print("Dense retrieval evaluation module test completed!")