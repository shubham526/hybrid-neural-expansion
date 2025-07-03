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
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.core.feature_extractor import ExpansionFeatureExtractor
from src.utils.file_utils import save_json, load_trec_run, ensure_dir
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
    parser.add_argument('--run-file-path', type=str,
                        help='Path to first-stage run file (if not using dataset scoreddocs)')
    parser.add_argument('--query-ids-file', type=str,
                        help='File with query IDs to process (optional)')

    # RM3 arguments
    parser.add_argument('--index-path', type=str, required=True,
                        help='Path to Lucene index for RM3 expansion')
    parser.add_argument('--lucene-path', type=str, required=True,
                        help='Path to Lucene JAR files')

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

    try:
        # Initialize components
        with TimedOperation(logger, "Initializing RM3 and semantic similarity components"):
            # Initialize Lucene for RM3
            from src.utils.lucene_utils import initialize_lucene
            if not initialize_lucene(args.lucene_path):
                raise RuntimeError("Failed to initialize Lucene")

            # Create RM3 expansion
            rm_expansion = RMExpansion(args.index_path)

            # Create semantic similarity
            semantic_similarity = SemanticSimilarity(args.semantic_model)

            # Create feature extractor
            feature_extractor = ExpansionFeatureExtractor(
                rm_expansion=rm_expansion,
                semantic_similarity=semantic_similarity,
                max_expansion_terms=args.max_expansion_terms,
                top_k_pseudo_docs=args.top_k_pseudo_docs
            )

            logger.info("All components initialized successfully")

        # Load dataset
        with TimedOperation(logger, f"Loading dataset: {args.dataset}"):
            dataset = ir_datasets.load(args.dataset)

            # Load queries
            queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}
            logger.info(f"Loaded {len(queries)} queries")

            # Load documents
            logger.info("Loading document collection...")
            documents = {}
            for doc in tqdm(dataset.docs_iter(), desc="Loading docs"):
                doc_text = doc.text if hasattr(doc, 'text') else doc.body
                documents[doc.doc_id] = doc_text
            logger.info(f"Loaded {len(documents)} documents")

            # Load first-stage runs
            first_stage_runs = {}
            if dataset.has_scoreddocs():
                logger.info("Using dataset scoreddocs for first-stage runs")
                for sdoc in dataset.scoreddocs_iter():
                    if sdoc.query_id not in first_stage_runs:
                        first_stage_runs[sdoc.query_id] = []
                    first_stage_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))

                # Sort by score
                for qid in first_stage_runs:
                    first_stage_runs[qid].sort(key=lambda x: x[1], reverse=True)

            elif args.run_file_path:
                logger.info(f"Loading first-stage runs from: {args.run_file_path}")
                first_stage_runs = load_trec_run(args.run_file_path)
            else:
                raise ValueError("Need either dataset scoreddocs or --run-file-path")

            logger.info(f"Loaded first-stage runs for {len(first_stage_runs)} queries")

            # Filter to query subset if specified
            if args.query_ids_file:
                logger.info(f"Filtering to query subset from: {args.query_ids_file}")
                with open(args.query_ids_file) as f:
                    subset_qids = {line.strip() for line in f if line.strip()}

                queries = {qid: text for qid, text in queries.items() if qid in subset_qids}
                first_stage_runs = {qid: data for qid, data in first_stage_runs.items() if qid in subset_qids}

                logger.info(f"Filtered to {len(queries)} queries")

        # Extract features
        with TimedOperation(logger, "Extracting RM3 and semantic similarity features"):
            features = feature_extractor.extract_features_for_dataset(
                queries=queries,
                first_stage_runs=first_stage_runs,
                document_collection=documents
            )

            logger.info(f"Successfully extracted features for {len(features)} queries")

        # Compute and log statistics
        stats = feature_extractor.get_feature_statistics(features)
        logger.info("Feature extraction statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        logger.info(f"    {sub_key}: {sub_value:.4f}")
                    else:
                        logger.info(f"    {sub_key}: {sub_value}")
            else:
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

        # Save features
        with TimedOperation(logger, "Saving extracted features"):
            # Create filename based on dataset and settings
            dataset_name = args.dataset.replace('/', '_')
            subset_name = "subset" if args.query_ids_file else "full"
            feature_filename = f"{dataset_name}_{subset_name}_features.json"
            feature_path = output_dir / feature_filename

            # Save features
            save_json(features, feature_path, compress=True)
            logger.info(f"Saved features to: {feature_path}")

            # Save statistics
            stats_path = output_dir / f"{dataset_name}_{subset_name}_stats.json"
            save_json(stats, stats_path)
            logger.info(f"Saved statistics to: {stats_path}")

            # Save extraction metadata
            metadata = {
                'dataset': args.dataset,
                'semantic_model': args.semantic_model,
                'max_expansion_terms': args.max_expansion_terms,
                'top_k_pseudo_docs': args.top_k_pseudo_docs,
                'num_queries_processed': len(features),
                'feature_file': str(feature_path),
                'stats_file': str(stats_path)
            }

            metadata_path = output_dir / f"{dataset_name}_{subset_name}_metadata.json"
            save_json(metadata, metadata_path)
            logger.info(f"Saved metadata to: {metadata_path}")

        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Queries processed: {len(features)}")
        logger.info(f"Features saved to: {feature_path}")
        logger.info(f"Average terms per query: {stats['avg_terms_per_query']:.2f}")
        logger.info("=" * 60)
        logger.info("Next step: Train neural model with:")
        logger.info(f"  python scripts/2_train_neural_model.py --feature-file {feature_path} --dataset {args.dataset}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()