#!/usr/bin/env python3
"""
Extract RM3 + Semantic Similarity Features

This script extracts the core features needed for neural reranking:
1. RM3 weights for expansion terms
2. Semantic similarity scores (cosine similarity with query)
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from src.core.feature_extractor import ExpansionFeatureExtractor
from src.utils.file_utils import save_json, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        return query_obj.title
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract RM3 and semantic similarity features for neural reranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='IR dataset name (e.g., msmarco-passage/trec-dl-2019)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for features')
    parser.add_argument('--query-ids-file', type=str,
                        help='File with query IDs to process (optional)')

    # RM3 arguments
    parser.add_argument('--index-path', type=str, required=True,
                        help='Path to Lucene index for RM3 expansion')
    parser.add_argument('--lucene-path', type=str, required=True,
                        help='Path to Lucene JAR files')
    parser.add_argument('--rm-alpha', type=float, default=0.5,
                        help='Mixing parameter for RM3 (weight of original query model)')

    # Feature extraction arguments
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model for semantic similarity')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms per query')
    parser.add_argument('--top-k-pseudo-docs', type=int, default=10,
                        help='Number of pseudo-relevant documents for RM3')

    # Other arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_features", args.log_level,
                                      str(output_dir / 'feature_extraction.log'))
    log_experiment_info(logger, **vars(args))

    feature_extractor = None
    try:
        # Initialize components
        with TimedOperation(logger, "Initializing RM3 and semantic similarity components"):
            from src.utils.lucene_utils import initialize_lucene
            if not initialize_lucene(args.lucene_path):
                raise RuntimeError("Failed to initialize Lucene")

            config = {
                'index_path': args.index_path,
                'embedding_model': args.semantic_model,
                'max_expansion_terms': args.max_expansion_terms,
                'top_k_pseudo_docs': args.top_k_pseudo_docs,
                'rm_alpha': args.rm_alpha
            }
            feature_extractor = ExpansionFeatureExtractor(config)
            logger.info("All components initialized successfully")

        # Load dataset queries
        with TimedOperation(logger, f"Loading queries from dataset: {args.dataset}"):
            dataset = ir_datasets.load(args.dataset)
            queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}

            # Filter to query subset if specified
            if args.query_ids_file:
                logger.info(f"Filtering to query subset from: {args.query_ids_file}")
                with open(args.query_ids_file) as f:
                    subset_qids = {line.strip() for line in f if line.strip()}
                queries = {qid: text for qid, text in queries.items() if qid in subset_qids}

            logger.info(f"Loaded {len(queries)} queries to be processed.")

        # Extract features and write them to a JSONL file line by line
        output_path = output_dir / f"{args.dataset.replace('/', '_')}_features.jsonl"
        with TimedOperation(logger, f"Extracting and writing features to {output_path}"):
            with open(output_path, 'w') as f_out:
                for query_id, query_text in tqdm(queries.items(), desc="Extracting features"):
                    features = feature_extractor.extract_features_for_query(query_id, query_text)
                    # The feature set for a single query is a dictionary, not a list
                    if features:
                        f_out.write(json.dumps(features) + '\n')

            logger.info(f"Successfully extracted and saved features for {len(queries)} queries.")

        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
        logger.info(f"Features saved to: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Crucial final step to prevent resource leaks
        if feature_extractor:
            logger.info("Closing resources...")
            feature_extractor.close()


if __name__ == "__main__":
    main()