#!/usr/bin/env python3
"""
Create Train/Test Data for Neural Reranking

This script creates train.jsonl and test.jsonl files for each fold based on:
1. Query expansion features (RM3 + semantic similarity)
2. Relevance judgments from ir_datasets
3. First-stage runs (from scoreddocs or run file)
4. Fold definitions from folds.json

Each line in the JSONL files contains:
{
    "query_id": "123",
    "query_text": "machine learning algorithms",
    "expansion_features": {
        "neural": {"rm_weight": 0.8, "semantic_score": 0.9},
        "networks": {"rm_weight": 0.6, "semantic_score": 0.7}
    },
    "candidates": [
        {"doc_id": "doc1", "score": 0.95, "relevance": 2},
        {"doc_id": "doc2", "score": 0.88, "relevance": 0}
    ]
}
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from src.utils.file_utils import load_json, save_json, load_trec_run, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        if hasattr(query_obj, 'description') and query_obj.description:
            return f"{query_obj.title} {query_obj.description}"
        return query_obj.title
    return ""


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL (one JSON object per line)."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class TrainTestDataCreator:
    """Creates train/test data for neural reranking experiments."""

    def __init__(self,
                 max_candidates_per_query: int = 100,
                 ensure_positive_examples: bool = True):
        """
        Initialize data creator.

        Args:
            max_candidates_per_query: Maximum candidates to include per query
            ensure_positive_examples: Whether to ensure training queries have positive examples
        """
        self.max_candidates_per_query = max_candidates_per_query
        self.ensure_positive_examples = ensure_positive_examples

        logger.info(f"TrainTestDataCreator initialized:")
        logger.info(f"  Max candidates per query: {max_candidates_per_query}")
        logger.info(f"  Ensure positive examples: {ensure_positive_examples}")

    def load_dataset_components(self, dataset_name: str, run_file_path: str = None) -> Dict[str, Any]:
        """
        Load all components from ir_datasets.

        Args:
            dataset_name: IR dataset name
            run_file_path: Optional path to run file (if not using scoreddocs)

        Returns:
            Dictionary with all loaded components
        """
        logger.info(f"Loading dataset components: {dataset_name}")

        dataset = ir_datasets.load(dataset_name)

        # Load queries
        queries = {}
        for query in dataset.queries_iter():
            queries[query.query_id] = get_query_text(query)
        logger.info(f"Loaded {len(queries)} queries")

        # Load qrels
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        logger.info(f"Loaded qrels for {len(qrels)} queries")

        # Load first-stage runs
        first_stage_runs = defaultdict(list)

        if dataset.has_scoreddocs():
            logger.info("Using dataset scoreddocs for first-stage runs")
            for sdoc in dataset.scoreddocs_iter():
                first_stage_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))

            # Sort by score (descending)
            for qid in first_stage_runs:
                first_stage_runs[qid].sort(key=lambda x: x[1], reverse=True)

        elif run_file_path:
            logger.info(f"Loading first-stage runs from: {run_file_path}")
            run_data = load_trec_run(run_file_path)
            first_stage_runs.update(run_data)

        else:
            raise ValueError("Need either dataset with scoreddocs or --run-file-path")

        logger.info(f"Loaded first-stage runs for {len(first_stage_runs)} queries")

        # Load documents (if needed - for large collections, might want to load on-demand)
        documents = {}
        try:
            doc_count = 0
            for doc in tqdm(dataset.docs_iter(), desc="Loading documents"):
                doc_text = doc.text if hasattr(doc, 'text') else doc.body
                documents[doc.doc_id] = doc_text
                doc_count += 1

                # For very large collections, you might want to limit this
                # or implement on-demand loading
                if doc_count > 1000000:  # 1M document limit
                    logger.warning(f"Stopping document loading at {doc_count} documents")
                    break

            logger.info(f"Loaded {len(documents)} documents")

        except Exception as e:
            logger.warning(f"Could not load full document collection: {e}")
            logger.info("Will load documents on-demand during processing")
            documents = None

        return {
            'queries': dict(queries),
            'qrels': dict(qrels),
            'first_stage_runs': dict(first_stage_runs),
            'documents': documents,
            'dataset': dataset
        }

    def create_training_example(self,
                                query_id: str,
                                query_text: str,
                                expansion_features: Dict[str, Dict[str, float]],
                                candidates: List[tuple],
                                qrels: Dict[str, int]) -> Dict[str, Any]:
        """
        Create a training example for one query.

        Args:
            query_id: Query identifier
            query_text: Query text
            expansion_features: Term features from feature extraction
            candidates: List of (doc_id, score) tuples from first-stage
            qrels: Relevance judgments for this query

        Returns:
            Training example dictionary
        """
        # Limit candidates
        candidates = candidates[:self.max_candidates_per_query]

        # Create candidate list with relevance labels
        candidate_list = []
        has_positive = False

        for doc_id, score in candidates:
            relevance = qrels.get(doc_id, 0)
            if relevance > 0:
                has_positive = True

            candidate_list.append({
                'doc_id': doc_id,
                'score': float(score),
                'relevance': int(relevance)
            })

        # Skip queries without positive examples in training (if required)
        if self.ensure_positive_examples and not has_positive:
            return None

        example = {
            'query_id': query_id,
            'query_text': query_text,
            'expansion_features': expansion_features,
            'candidates': candidate_list,
            'num_candidates': len(candidate_list),
            'num_positive': sum(1 for c in candidate_list if c['relevance'] > 0)
        }

        return example

    def create_fold_data(self,
                         fold_info: Dict[str, List[str]],
                         features: Dict[str, Any],
                         dataset_components: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Create train/test data for one fold.

        Args:
            fold_info: Dictionary with 'training' and 'testing' query ID lists
            features: Extracted expansion features
            dataset_components: Loaded dataset components

        Returns:
            Dictionary with 'train' and 'test' data lists
        """
        queries = dataset_components['queries']
        qrels = dataset_components['qrels']
        first_stage_runs = dataset_components['first_stage_runs']

        train_qids = set(fold_info['training'])
        test_qids = set(fold_info['testing'])

        train_data = []
        test_data = []

        # Process training queries
        logger.info(f"Processing {len(train_qids)} training queries...")
        for query_id in tqdm(train_qids, desc="Training queries"):
            if (query_id not in features or
                    query_id not in queries or
                    query_id not in first_stage_runs):
                logger.debug(f"Skipping training query {query_id} (missing data)")
                continue

            query_text = queries[query_id]
            expansion_features = features[query_id]['term_features']
            candidates = first_stage_runs[query_id]
            query_qrels = qrels.get(query_id, {})

            example = self.create_training_example(
                query_id, query_text, expansion_features, candidates, query_qrels
            )

            if example is not None:
                train_data.append(example)

        # Process test queries
        logger.info(f"Processing {len(test_qids)} test queries...")
        for query_id in tqdm(test_qids, desc="Test queries"):
            if (query_id not in features or
                    query_id not in queries or
                    query_id not in first_stage_runs):
                logger.debug(f"Skipping test query {query_id} (missing data)")
                continue

            query_text = queries[query_id]
            expansion_features = features[query_id]['term_features']
            candidates = first_stage_runs[query_id]
            query_qrels = qrels.get(query_id, {})

            # For test, don't filter out queries without positive examples
            example = self.create_training_example(
                query_id, query_text, expansion_features, candidates, query_qrels
            )

            if example is not None:
                test_data.append(example)

        logger.info(f"Created {len(train_data)} training examples, {len(test_data)} test examples")

        return {
            'train': train_data,
            'test': test_data
        }

    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics about the created data."""
        if not data:
            return {}

        total_queries = len(data)
        total_candidates = sum(len(example['candidates']) for example in data)
        total_positive = sum(example['num_positive'] for example in data)

        queries_with_positives = sum(1 for example in data if example['num_positive'] > 0)

        avg_candidates_per_query = total_candidates / total_queries if total_queries > 0 else 0
        avg_positive_per_query = total_positive / total_queries if total_queries > 0 else 0

        # Feature statistics
        all_expansion_terms = set()
        for example in data:
            all_expansion_terms.update(example['expansion_features'].keys())

        stats = {
            'num_queries': total_queries,
            'num_candidates': total_candidates,
            'num_positive_examples': total_positive,
            'queries_with_positives': queries_with_positives,
            'avg_candidates_per_query': avg_candidates_per_query,
            'avg_positive_per_query': avg_positive_per_query,
            'positive_rate': total_positive / total_candidates if total_candidates > 0 else 0,
            'unique_expansion_terms': len(all_expansion_terms)
        }

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test data for neural reranking experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='IR dataset name (e.g., disks45/nocr/trec-robust-2004)')
    parser.add_argument('--features-file', type=str, required=True,
                        help='Path to extracted features file')
    parser.add_argument('--folds-file', type=str, required=True,
                        help='Path to folds.json file with train/test splits')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for train/test files')

    # Optional data source
    parser.add_argument('--run-file-path', type=str,
                        help='Path to TREC run file (if not using dataset scoreddocs)')

    # Data creation parameters
    parser.add_argument('--max-candidates-per-query', type=int, default=100,
                        help='Maximum candidates to include per query')
    parser.add_argument('--ensure-positive-training', action='store_true',
                        help='Ensure training queries have positive examples')

    # Processing options
    parser.add_argument('--specific-fold', type=str,
                        help='Process only specific fold (e.g., "0")')
    parser.add_argument('--save-statistics', action='store_true',
                        help='Save data statistics')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_train_test", args.log_level,
                                      str(output_dir / 'data_creation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Load folds definition
        with TimedOperation(logger, "Loading fold definitions"):
            folds = load_json(args.folds_file)
            logger.info(f"Loaded {len(folds)} folds")

            if args.specific_fold:
                if args.specific_fold not in folds:
                    raise ValueError(f"Fold '{args.specific_fold}' not found in folds file")
                folds = {args.specific_fold: folds[args.specific_fold]}
                logger.info(f"Processing only fold {args.specific_fold}")

        # Load features
        with TimedOperation(logger, "Loading extracted features"):
            features = load_json(args.features_file)
            logger.info(f"Loaded features for {len(features)} queries")

        # Load dataset components
        with TimedOperation(logger, "Loading dataset components"):
            dataset_components = TrainTestDataCreator().load_dataset_components(
                args.dataset, args.run_file_path
            )

        # Create data creator
        data_creator = TrainTestDataCreator(
            max_candidates_per_query=args.max_candidates_per_query,
            ensure_positive_examples=args.ensure_positive_training
        )

        # Process each fold
        for fold_id, fold_info in folds.items():
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing Fold {fold_id}")
            logger.info(f"{'=' * 50}")

            # Create fold directory
            fold_dir = ensure_dir(output_dir / f"fold_{fold_id}")

            # Log fold info
            train_qids = fold_info['training']
            test_qids = fold_info['testing']
            logger.info(f"Training queries: {len(train_qids)}")
            logger.info(f"Test queries: {len(test_qids)}")

            # Create train/test data for this fold
            with TimedOperation(logger, f"Creating data for fold {fold_id}"):
                fold_data = data_creator.create_fold_data(
                    fold_info, features, dataset_components
                )

            # Save train/test files
            train_file = fold_dir / 'train.jsonl'
            test_file = fold_dir / 'test.jsonl'

            save_jsonl(fold_data['train'], train_file)
            save_jsonl(fold_data['test'], test_file)

            logger.info(f"Saved training data to: {train_file}")
            logger.info(f"Saved test data to: {test_file}")

            # Compute and save statistics
            if args.save_statistics:
                train_stats = data_creator.get_data_statistics(fold_data['train'])
                test_stats = data_creator.get_data_statistics(fold_data['test'])

                stats = {
                    'fold_id': fold_id,
                    'dataset': args.dataset,
                    'train_stats': train_stats,
                    'test_stats': test_stats,
                    'train_file': str(train_file),
                    'test_file': str(test_file)
                }

                stats_file = fold_dir / 'data_stats.json'
                save_json(stats, stats_file)
                logger.info(f"Saved statistics to: {stats_file}")

                # Log key statistics
                logger.info(f"Training data statistics:")
                logger.info(f"  Queries: {train_stats['num_queries']}")
                logger.info(f"  Candidates: {train_stats['num_candidates']}")
                logger.info(f"  Positive examples: {train_stats['num_positive_examples']}")
                logger.info(f"  Positive rate: {train_stats['positive_rate']:.3f}")

                logger.info(f"Test data statistics:")
                logger.info(f"  Queries: {test_stats['num_queries']}")
                logger.info(f"  Candidates: {test_stats['num_candidates']}")
                logger.info(f"  Positive examples: {test_stats['num_positive_examples']}")
                logger.info(f"  Positive rate: {test_stats['positive_rate']:.3f}")

        # Create overall summary
        summary = {
            'dataset': args.dataset,
            'features_file': args.features_file,
            'folds_file': args.folds_file,
            'num_folds_processed': len(folds),
            'max_candidates_per_query': args.max_candidates_per_query,
            'ensure_positive_training': args.ensure_positive_training,
            'output_directory': str(output_dir)
        }

        summary_file = output_dir / 'summary.json'
        save_json(summary, summary_file)
        logger.info(f"Saved overall summary to: {summary_file}")

        logger.info("\n" + "=" * 60)
        logger.info("TRAIN/TEST DATA CREATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Folds processed: {len(folds)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)
        logger.info("Next steps:")
        logger.info("1. Train neural models for each fold:")
        for fold_id in folds.keys():
            logger.info(
                f"   python scripts/2_train_neural_model.py --train-file {output_dir}/fold_{fold_id}/train.jsonl")
        logger.info("2. Evaluate trained models:")
        for fold_id in folds.keys():
            logger.info(f"   python scripts/3_evaluate_model.py --test-file {output_dir}/fold_{fold_id}/test.jsonl")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Data creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()