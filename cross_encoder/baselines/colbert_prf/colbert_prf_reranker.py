#!/usr/bin/env python3
"""
Simplified ColBERT-PRF Reranking System

Uses existing ColBERT index and PyTerrier's built-in PRF functionality.
No per-query indexing needed!
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyterrier as pt
import ir_datasets
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def to_pyterrier_format(self, rerank_depth: int = None) -> pd.DataFrame:
        """Convert run data to PyTerrier DataFrame format."""
        results = []

        for qid, doc_scores in self.run_data.items():
            # Limit to rerank_depth if specified
            doc_scores_limited = doc_scores[:rerank_depth] if rerank_depth else doc_scores

            for rank, (docid, score) in enumerate(doc_scores_limited, 1):
                results.append({
                    'qid': qid,
                    'docno': docid,
                    'score': score,
                    'rank': rank
                })

        return pd.DataFrame(results)


class IRDatasetHandler:
    """Handles ir_datasets integration for queries only (documents come from ColBERT index)."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        try:
            self.dataset = ir_datasets.load(dataset_name)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

        # Build query cache
        self.query_cache = {}
        self._build_query_cache()

    def _build_query_cache(self):
        """Build query ID to text cache."""
        logger.info("Building query cache...")

        if not hasattr(self.dataset, 'queries_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have queries")
            return

        query_count = 0
        for query in self.dataset.queries_iter():
            query_text = self._extract_query_text(query)
            if query_text.strip():
                self.query_cache[query.query_id] = query_text
                query_count += 1

        logger.info(f"Loaded {query_count} queries")

    def _extract_query_text(self, query) -> str:
        """Extract text from query with flexible field handling."""
        text_fields = ['text', 'title', 'query', 'description', 'narrative']

        for field in text_fields:
            if hasattr(query, field):
                field_value = getattr(query, field)
                if field_value and str(field_value).strip():
                    return str(field_value).strip()

        return ""

    def get_queries_dataframe(self, query_ids: List[str]) -> pd.DataFrame:
        """Get queries in PyTerrier DataFrame format."""
        queries = []

        for qid in query_ids:
            if qid in self.query_cache:
                queries.append({
                    'qid': qid,
                    'query': self.query_cache[qid]
                })
            else:
                logger.warning(f"Query {qid} not found in dataset")

        return pd.DataFrame(queries)


class ColBERTPRFReranker:
    """Simplified ColBERT-PRF reranker using existing index."""

    def __init__(self,
                 checkpoint_path: str,
                 index_root: str,
                 index_name: str,
                 dataset_name: str,
                 fb_docs: int = 3,
                 k: int = 24,
                 fb_embs: int = 10,
                 beta: float = 1.0):

        self.checkpoint_path = checkpoint_path
        self.index_root = index_root
        self.index_name = index_name
        self.dataset_name = dataset_name
        self.fb_docs = fb_docs
        self.k = k
        self.fb_embs = fb_embs
        self.beta = beta

        # Initialize PyTerrier
        if not pt.started():
            pt.java.init()

        # Initialize dataset handler (for queries only)
        self.dataset_handler = IRDatasetHandler(dataset_name)

        # Initialize ColBERT
        self._setup_colbert()

    def _setup_colbert(self):
        """Setup ColBERT factory and PRF pipeline."""
        try:
            from pyterrier_colbert.ranking import ColBERTFactory

            logger.info(f"Loading ColBERT from checkpoint: {self.checkpoint_path}")
            logger.info(f"Using index: {self.index_root}/{self.index_name}")

            # Create ColBERT factory
            self.pytcolbert = ColBERTFactory(
                self.checkpoint_path,
                self.index_root,
                self.index_name
            )

            # Create PRF reranker pipeline
            logger.info(f"Creating ColBERT PRF pipeline...")
            logger.info(f"  fb_docs: {self.fb_docs}")
            logger.info(f"  k: {self.k}")
            logger.info(f"  fb_embs: {self.fb_embs}")
            logger.info(f"  beta: {self.beta}")

            self.prf_reranker = self.pytcolbert.prf(
                rerank=True,  # Rerank initial documents
                fb_docs=self.fb_docs,  # Number of feedback documents
                k=self.k,  # Number of clusters
                fb_embs=self.fb_embs,  # Number of expansion embeddings
                beta=self.beta  # Weight of expansion embeddings
            )

            logger.info("ColBERT PRF pipeline ready!")

        except ImportError as e:
            logger.error(f"Failed to import PyTerrier ColBERT: {e}")
            logger.error("Please install with: pip install git+https://github.com/terrierteam/pyterrier_colbert.git")
            raise
        except Exception as e:
            logger.error(f"Failed to setup ColBERT: {e}")
            raise

    def rerank_run(self,
                   run_path: str,
                   output_path: str,
                   rerank_depth: int = 100) -> None:
        """Rerank a TREC run file and save results."""

        logger.info(f"Loading run file: {run_path}")

        # Load run file
        run_loader = TRECRunLoader(run_path)
        logger.info(f"Loaded {len(run_loader.run_data)} queries")

        # Convert run to PyTerrier format
        logger.info(f"Converting to PyTerrier format (rerank_depth: {rerank_depth})")
        initial_results = run_loader.to_pyterrier_format(rerank_depth)
        logger.info(f"Initial results: {len(initial_results)} query-document pairs")

        # Get queries
        query_ids = list(run_loader.run_data.keys())
        queries_df = self.dataset_handler.get_queries_dataframe(query_ids)
        logger.info(f"Found {len(queries_df)} queries in dataset")

        if len(queries_df) == 0:
            logger.error("No queries found in dataset!")
            return

        # Create initial retriever from run file
        logger.info("Creating initial retriever from run file...")
        initial_retriever = pt.Transformer.from_df(initial_results, uniform=True)

        # Create full pipeline: initial results -> ColBERT PRF reranking
        logger.info("Starting ColBERT PRF reranking...")
        pipeline = initial_retriever >> self.prf_reranker

        # Run the pipeline
        logger.info("Running pipeline...")
        start_time = time.time()

        try:
            results = pipeline(queries_df)
            end_time = time.time()

            logger.info(f"Reranking completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Results: {len(results)} query-document pairs")

            # Save results
            self._save_results(results, output_path)
            logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_results(self, results: pd.DataFrame, output_path: str):
        """Save results in TREC format."""
        logger.info(f"Saving {len(results)} results to {output_path}")

        with open(output_path, 'w') as f:
            for _, row in results.iterrows():
                f.write(f"{row['qid']} Q0 {row['docno']} "
                        f"{row['rank']} {row['score']:.6f} ColBERT-PRF\n")


def main():
    parser = argparse.ArgumentParser(description="Simplified ColBERT-PRF Reranking System")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to ColBERT checkpoint")
    parser.add_argument("--index-root", required=True, help="ColBERT index root directory")
    parser.add_argument("--index-name", required=True, help="ColBERT index name")
    parser.add_argument("--dataset", required=True, help="IR dataset name (e.g., 'msmarco-passage/dev')")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--output", required=True, help="Path to output reranked file")

    # Optional arguments
    parser.add_argument("--rerank-depth", type=int, default=100, help="Number of top docs to rerank")
    parser.add_argument("--fb-docs", type=int, default=3, help="Number of feedback documents")
    parser.add_argument("--k", type=int, default=24, help="Number of clusters")
    parser.add_argument("--fb-embs", type=int, default=10, help="Number of expansion embeddings")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight of expansion embeddings")

    args = parser.parse_args()

    # Initialize reranker
    reranker = ColBERTPRFReranker(
        checkpoint_path=args.checkpoint,
        index_root=args.index_root,
        index_name=args.index_name,
        dataset_name=args.dataset,
        fb_docs=args.fb_docs,
        k=args.k,
        fb_embs=args.fb_embs,
        beta=args.beta
    )

    # Perform reranking
    reranker.rerank_run(
        run_path=args.run,
        output_path=args.output,
        rerank_depth=args.rerank_depth
    )

    print(f"Reranking completed. Results saved to {args.output}")


if __name__ == "__main__":
    import time

    main()