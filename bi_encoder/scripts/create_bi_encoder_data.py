#!/usr/bin/env python3
"""
Create Bi-Encoder Training Data for Contrastive Learning

This script converts cross-encoder training data (query-document pairs) into
bi-encoder format suitable for contrastive learning. It reuses the same
feature extraction pipeline but restructures data with proper positive/negative sampling.

Usage:
    python create_bi_encoder_data.py \
        --input-file path/to/cross_encoder_train.jsonl \
        --output-dir path/to/bi_encoder_data/ \
        --num-positives 3 \
        --num-negatives 7 \
        --num-hard-negatives 5

Expected input format (from cross-encoder):
{
    'query_id': str,
    'query_text': str,
    'expansion_features': {term: {'rm_weight': float, 'semantic_score': float}},
    'candidates': [
        {'doc_id': str, 'doc_text': str, 'relevance': int, 'score': float}
    ]
}

Output format (for bi-encoder):
{
    'query_id': str,
    'query_text': str,
    'expansion_features': {...},  # Same as input
    'positive_docs': [doc_text1, doc_text2, ...],
    'negative_docs': [doc_text1, doc_text2, ...],
    'hard_negatives': [doc_text1, doc_text2, ...]  # High BM25, low relevance
}
"""

import argparse
import json
import logging
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.utils.file_utils import ensure_dir, save_json, load_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class BiEncoderDataCreator:
    """
    Convert cross-encoder data to bi-encoder contrastive learning format.
    """

    def __init__(self,
                 num_positives: int = 3,
                 num_negatives: int = 7,
                 num_hard_negatives: int = 5,
                 min_positive_relevance: int = 1,
                 hard_negative_score_threshold: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize bi-encoder data creator.

        Args:
            num_positives: Number of positive documents per query
            num_negatives: Number of random negative documents per query
            num_hard_negatives: Number of hard negatives (high BM25, low relevance)
            min_positive_relevance: Minimum relevance score for positive documents
            hard_negative_score_threshold: Minimum BM25 score for hard negatives
            random_seed: Random seed for reproducibility
        """
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.num_hard_negatives = num_hard_negatives
        self.min_positive_relevance = min_positive_relevance
        self.hard_negative_score_threshold = hard_negative_score_threshold
        self.random_seed = random_seed

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(f"BiEncoderDataCreator initialized:")
        logger.info(f"  Positives per query: {num_positives}")
        logger.info(f"  Negatives per query: {num_negatives}")
        logger.info(f"  Hard negatives per query: {num_hard_negatives}")
        logger.info(f"  Min positive relevance: {min_positive_relevance}")
        logger.info(f"  Hard negative score threshold: {hard_negative_score_threshold}")

    def load_cross_encoder_data(self, input_file: Path) -> List[Dict[str, Any]]:
        """Load cross-encoder training data."""
        logger.info(f"Loading cross-encoder data from: {input_file}")

        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")

        logger.info(f"Loaded {len(data)} queries from cross-encoder data")
        return data

    def extract_documents_by_relevance(self, candidates: List[Dict[str, Any]]) -> Tuple[
        List[Dict], List[Dict], List[Dict]]:
        """
        Separate candidates into positive, negative, and hard negative categories.

        Args:
            candidates: List of candidate documents with relevance scores

        Returns:
            Tuple of (positives, negatives, hard_negatives)
        """
        positives = []
        negatives = []
        hard_negatives = []

        for candidate in candidates:
            # Skip if no document text
            if 'doc_text' not in candidate:
                continue

            relevance = candidate.get('relevance', 0)
            score = candidate.get('score', candidate.get('first_stage_score', 0.0))

            if relevance >= self.min_positive_relevance:
                positives.append(candidate)
            elif relevance == 0:
                # Hard negatives: high first-stage score but not relevant
                if score >= self.hard_negative_score_threshold:
                    hard_negatives.append(candidate)
                else:
                    negatives.append(candidate)

        return positives, negatives, hard_negatives

    def sample_documents(self, documents: List[Dict[str, Any]], num_samples: int,
                         strategy: str = 'random') -> List[str]:
        """
        Sample documents from a list.

        Args:
            documents: List of document dictionaries
            num_samples: Number of documents to sample
            strategy: Sampling strategy ('random', 'top_scored', 'diverse')

        Returns:
            List of document texts
        """
        if not documents:
            return []

        # Limit to available documents
        num_samples = min(num_samples, len(documents))

        if strategy == 'random':
            sampled = random.sample(documents, num_samples)
        elif strategy == 'top_scored':
            # Sort by score and take top
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
            sampled = sorted_docs[:num_samples]
        elif strategy == 'diverse':
            # Try to get diverse documents (simple implementation)
            # Could be enhanced with clustering or other diversity measures
            if len(documents) <= num_samples:
                sampled = documents
            else:
                # Take every nth document for diversity
                step = len(documents) // num_samples
                sampled = [documents[i * step] for i in range(num_samples)]
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return [doc['doc_text'] for doc in sampled]

    def create_bi_encoder_example(self, cross_encoder_example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single cross-encoder example to bi-encoder format.

        Args:
            cross_encoder_example: Cross-encoder training example

        Returns:
            Bi-encoder training example or None if insufficient data
        """
        query_id = cross_encoder_example['query_id']
        query_text = cross_encoder_example['query_text']
        expansion_features = cross_encoder_example['expansion_features']
        candidates = cross_encoder_example['candidates']

        # Separate documents by relevance
        positives, negatives, hard_negatives = self.extract_documents_by_relevance(candidates)

        # Check if we have enough positive documents
        if len(positives) == 0:
            logger.debug(f"Skipping query {query_id} - no positive documents")
            return None

        # Sample documents
        positive_docs = self.sample_documents(positives, self.num_positives, strategy='random')
        negative_docs = self.sample_documents(negatives, self.num_negatives, strategy='random')
        hard_negative_docs = self.sample_documents(hard_negatives, self.num_hard_negatives, strategy='top_scored')

        # Create bi-encoder example
        bi_encoder_example = {
            'query_id': query_id,
            'query_text': query_text,
            'expansion_features': expansion_features,
            'positive_docs': positive_docs,
            'negative_docs': negative_docs,
            'hard_negatives': hard_negative_docs,
            'stats': {
                'num_candidates': len(candidates),
                'num_available_positives': len(positives),
                'num_available_negatives': len(negatives),
                'num_available_hard_negatives': len(hard_negatives),
                'num_sampled_positives': len(positive_docs),
                'num_sampled_negatives': len(negative_docs),
                'num_sampled_hard_negatives': len(hard_negative_docs)
            }
        }

        return bi_encoder_example

    def process_cross_encoder_data(self, cross_encoder_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all cross-encoder data into bi-encoder format.

        Args:
            cross_encoder_data: List of cross-encoder examples

        Returns:
            List of bi-encoder examples
        """
        bi_encoder_data = []
        skipped_queries = 0

        logger.info(f"Processing {len(cross_encoder_data)} cross-encoder examples...")

        for example in tqdm(cross_encoder_data, desc="Converting to bi-encoder format"):
            bi_encoder_example = self.create_bi_encoder_example(example)

            if bi_encoder_example is not None:
                bi_encoder_data.append(bi_encoder_example)
            else:
                skipped_queries += 1

        logger.info(f"Conversion completed:")
        logger.info(f"  Processed: {len(bi_encoder_data)} queries")
        logger.info(f"  Skipped: {skipped_queries} queries (no positive documents)")

        return bi_encoder_data

    def compute_data_statistics(self, bi_encoder_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics about the created bi-encoder data."""
        if not bi_encoder_data:
            return {}

        # Collect statistics
        total_queries = len(bi_encoder_data)
        total_positives = sum(len(ex['positive_docs']) for ex in bi_encoder_data)
        total_negatives = sum(len(ex['negative_docs']) for ex in bi_encoder_data)
        total_hard_negatives = sum(len(ex['hard_negatives']) for ex in bi_encoder_data)

        # Document length statistics
        all_positive_lengths = []
        all_negative_lengths = []

        for example in bi_encoder_data:
            for doc in example['positive_docs']:
                all_positive_lengths.append(len(doc.split()))
            for doc in example['negative_docs']:
                all_negative_lengths.append(len(doc.split()))
            for doc in example['hard_negatives']:
                all_negative_lengths.append(len(doc.split()))

        # Expansion feature statistics
        expansion_terms_per_query = [len(ex['expansion_features']) for ex in bi_encoder_data]

        stats = {
            'total_queries': total_queries,
            'total_positive_docs': total_positives,
            'total_negative_docs': total_negatives,
            'total_hard_negatives': total_hard_negatives,
            'avg_positives_per_query': total_positives / total_queries if total_queries > 0 else 0,
            'avg_negatives_per_query': total_negatives / total_queries if total_queries > 0 else 0,
            'avg_hard_negatives_per_query': total_hard_negatives / total_queries if total_queries > 0 else 0,
            'avg_expansion_terms_per_query': np.mean(expansion_terms_per_query) if expansion_terms_per_query else 0,
            'document_length_stats': {
                'positive_docs': {
                    'mean': np.mean(all_positive_lengths) if all_positive_lengths else 0,
                    'std': np.std(all_positive_lengths) if all_positive_lengths else 0,
                    'min': np.min(all_positive_lengths) if all_positive_lengths else 0,
                    'max': np.max(all_positive_lengths) if all_positive_lengths else 0
                },
                'negative_docs': {
                    'mean': np.mean(all_negative_lengths) if all_negative_lengths else 0,
                    'std': np.std(all_negative_lengths) if all_negative_lengths else 0,
                    'min': np.min(all_negative_lengths) if all_negative_lengths else 0,
                    'max': np.max(all_negative_lengths) if all_negative_lengths else 0
                }
            }
        }

        return stats

    def create_train_val_split(self, bi_encoder_data: List[Dict[str, Any]],
                               val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Split bi-encoder data into train and validation sets.

        Args:
            bi_encoder_data: List of bi-encoder examples
            val_ratio: Fraction of data to use for validation

        Returns:
            Tuple of (train_data, val_data)
        """
        # Shuffle data
        shuffled_data = bi_encoder_data.copy()
        random.shuffle(shuffled_data)

        # Split
        val_size = int(len(shuffled_data) * val_ratio)
        val_data = shuffled_data[:val_size]
        train_data = shuffled_data[val_size:]

        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation")

        return train_data, val_data

    def save_bi_encoder_data(self, bi_encoder_data: List[Dict[str, Any]],
                             output_file: Path):
        """Save bi-encoder data in JSONL format."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in bi_encoder_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(bi_encoder_data)} bi-encoder examples to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create bi-encoder training data from cross-encoder data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output arguments
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to cross-encoder training data (JSONL format)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for bi-encoder data')
    parser.add_argument('--split-data', action='store_true',
                        help='Create train/validation split')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio')

    # Sampling arguments
    parser.add_argument('--num-positives', type=int, default=3,
                        help='Number of positive documents per query')
    parser.add_argument('--num-negatives', type=int, default=7,
                        help='Number of random negative documents per query')
    parser.add_argument('--num-hard-negatives', type=int, default=5,
                        help='Number of hard negatives per query')
    parser.add_argument('--min-positive-relevance', type=int, default=1,
                        help='Minimum relevance score for positive documents')
    parser.add_argument('--hard-negative-threshold', type=float, default=0.5,
                        help='Minimum BM25 score for hard negatives')

    # Other arguments
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_bi_encoder_data", args.log_level,
                                      str(output_dir / 'data_creation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Create data creator
        data_creator = BiEncoderDataCreator(
            num_positives=args.num_positives,
            num_negatives=args.num_negatives,
            num_hard_negatives=args.num_hard_negatives,
            min_positive_relevance=args.min_positive_relevance,
            hard_negative_score_threshold=args.hard_negative_threshold,
            random_seed=args.random_seed
        )

        # Load cross-encoder data
        with TimedOperation(logger, "Loading cross-encoder data"):
            cross_encoder_data = data_creator.load_cross_encoder_data(Path(args.input_file))

        # Convert to bi-encoder format
        with TimedOperation(logger, "Converting to bi-encoder format"):
            bi_encoder_data = data_creator.process_cross_encoder_data(cross_encoder_data)

        if not bi_encoder_data:
            logger.error("No bi-encoder data created. Check input data and parameters.")
            sys.exit(1)

        # Compute statistics
        with TimedOperation(logger, "Computing data statistics"):
            stats = data_creator.compute_data_statistics(bi_encoder_data)

            logger.info("Data Statistics:")
            logger.info(f"  Total queries: {stats['total_queries']}")
            logger.info(f"  Avg positives per query: {stats['avg_positives_per_query']:.2f}")
            logger.info(f"  Avg negatives per query: {stats['avg_negatives_per_query']:.2f}")
            logger.info(f"  Avg hard negatives per query: {stats['avg_hard_negatives_per_query']:.2f}")
            logger.info(f"  Avg expansion terms per query: {stats['avg_expansion_terms_per_query']:.2f}")

        # Save data
        if args.split_data:
            # Create train/validation split
            train_data, val_data = data_creator.create_train_val_split(
                bi_encoder_data, args.val_ratio
            )

            # Save splits
            data_creator.save_bi_encoder_data(train_data, output_dir / 'train.jsonl')
            data_creator.save_bi_encoder_data(val_data, output_dir / 'val.jsonl')

            # Save statistics for each split
            train_stats = data_creator.compute_data_statistics(train_data)
            val_stats = data_creator.compute_data_statistics(val_data)

            all_stats = {
                'overall_stats': stats,
                'train_stats': train_stats,
                'val_stats': val_stats,
                'split_ratio': args.val_ratio,
                'creation_params': {
                    'num_positives': args.num_positives,
                    'num_negatives': args.num_negatives,
                    'num_hard_negatives': args.num_hard_negatives,
                    'min_positive_relevance': args.min_positive_relevance,
                    'hard_negative_threshold': args.hard_negative_threshold,
                    'random_seed': args.random_seed
                }
            }
        else:
            # Save all data in one file
            data_creator.save_bi_encoder_data(bi_encoder_data, output_dir / 'train.jsonl')

            all_stats = {
                'overall_stats': stats,
                'creation_params': {
                    'num_positives': args.num_positives,
                    'num_negatives': args.num_negatives,
                    'num_hard_negatives': args.num_hard_negatives,
                    'min_positive_relevance': args.min_positive_relevance,
                    'hard_negative_threshold': args.hard_negative_threshold,
                    'random_seed': args.random_seed
                }
            }

        # Save statistics
        save_json(all_stats, output_dir / 'data_statistics.json')

        # Create summary
        summary = {
            'input_file': str(args.input_file),
            'output_directory': str(output_dir),
            'total_bi_encoder_examples': len(bi_encoder_data),
            'data_split': args.split_data,
            'files_created': ['train.jsonl', 'data_statistics.json']
        }

        if args.split_data:
            summary['files_created'].append('val.jsonl')
            summary['train_examples'] = len(train_data)
            summary['val_examples'] = len(val_data)

        save_json(summary, output_dir / 'creation_summary.json')

        logger.info("\n" + "=" * 60)
        logger.info("BI-ENCODER DATA CREATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Input: {args.input_file}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Total examples created: {len(bi_encoder_data)}")

        if args.split_data:
            logger.info(f"Train examples: {len(train_data)}")
            logger.info(f"Validation examples: {len(val_data)}")

        logger.info("\nNext steps:")
        logger.info("1. Train bi-encoder model:")
        logger.info(f"   python bi_encoder/scripts/train_bi_encoder.py --train-file {output_dir}/train.jsonl")
        logger.info("2. Evaluate model:")
        logger.info(f"   python bi_encoder/scripts/evaluate_retrieval.py")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Bi-encoder data creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()