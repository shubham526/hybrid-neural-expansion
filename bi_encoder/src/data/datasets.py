"""
Bi-Encoder Datasets for Contrastive Learning

This module provides PyTorch datasets specifically designed for bi-encoder
contrastive learning training, handling positive/negative document sampling
and batch creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class BiEncoderDataset(Dataset):
    """
    Dataset for bi-encoder contrastive learning.

    Handles queries with positive and negative documents for training
    with InfoNCE or triplet loss.
    """

    def __init__(self,
                 data_path: Union[str, Path, List[Dict]],
                 max_positives: int = 3,
                 max_negatives: int = 7,
                 max_hard_negatives: int = 5,
                 include_hard_negatives: bool = True,
                 shuffle_negatives: bool = True,
                 random_seed: int = 42):
        """
        Initialize bi-encoder dataset.

        Args:
            data_path: Path to JSONL file or list of data examples
            max_positives: Maximum positive documents per query
            max_negatives: Maximum random negative documents per query
            max_hard_negatives: Maximum hard negatives per query
            include_hard_negatives: Whether to include hard negatives
            shuffle_negatives: Whether to shuffle negatives each epoch
            random_seed: Random seed for reproducibility
        """
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.include_hard_negatives = include_hard_negatives
        self.shuffle_negatives = shuffle_negatives

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load data
        if isinstance(data_path, list):
            self.data = data_path
        else:
            self.data = self._load_jsonl(data_path)

        # Filter valid examples
        self.data = self._filter_valid_examples(self.data)

        logger.info(f"BiEncoderDataset initialized:")
        logger.info(f"  Examples: {len(self.data)}")
        logger.info(f"  Max positives: {max_positives}")
        logger.info(f"  Max negatives: {max_negatives}")
        logger.info(f"  Max hard negatives: {max_hard_negatives}")
        logger.info(f"  Include hard negatives: {include_hard_negatives}")

    def _load_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")

        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data

    def _filter_valid_examples(self, data: List[Dict]) -> List[Dict]:
        """Filter examples that have sufficient positive documents."""
        valid_data = []

        for example in data:
            # Check required fields
            if not all(field in example for field in ['query_text', 'positive_docs']):
                continue

            # Check if we have enough positive documents
            if len(example['positive_docs']) == 0:
                continue

            valid_data.append(example)

        logger.info(f"Filtered to {len(valid_data)} valid examples")
        return valid_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training example.

        Returns:
            Dictionary with query, positive docs, negative docs, and expansion features
        """
        example = self.data[idx]

        # Extract basic fields
        query_text = example['query_text']
        expansion_features = example.get('expansion_features', {})

        # Sample positive documents
        positive_docs = example['positive_docs']
        if len(positive_docs) > self.max_positives:
            positive_docs = random.sample(positive_docs, self.max_positives)

        # Sample negative documents
        negative_docs = []

        # Random negatives
        if 'negative_docs' in example and example['negative_docs']:
            available_negatives = example['negative_docs']
            if self.shuffle_negatives:
                random.shuffle(available_negatives)

            num_to_sample = min(self.max_negatives, len(available_negatives))
            negative_docs.extend(available_negatives[:num_to_sample])

        # Hard negatives
        if self.include_hard_negatives and 'hard_negatives' in example:
            hard_negatives = example['hard_negatives']
            if hard_negatives:
                if self.shuffle_negatives:
                    random.shuffle(hard_negatives)

                num_to_sample = min(self.max_hard_negatives, len(hard_negatives))
                negative_docs.extend(hard_negatives[:num_to_sample])

        return {
            'query_text': query_text,
            'expansion_features': expansion_features,
            'positive_docs': positive_docs,
            'negative_docs': negative_docs,
            'query_id': example.get('query_id', f'query_{idx}')
        }


class InBatchNegativeDataset(Dataset):
    """
    Dataset for in-batch negative sampling.

    Uses other positive documents in the batch as negatives,
    which is more memory efficient for large-scale training.
    """

    def __init__(self,
                 data_path: Union[str, Path, List[Dict]],
                 max_positives_per_query: int = 1,
                 random_seed: int = 42):
        """
        Initialize in-batch negative dataset.

        Args:
            data_path: Path to JSONL file or list of data examples
            max_positives_per_query: Number of positive docs to sample per query
            random_seed: Random seed
        """
        self.max_positives_per_query = max_positives_per_query

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load and filter data
        if isinstance(data_path, list):
            self.data = data_path
        else:
            self.data = self._load_jsonl(data_path)

        self.data = self._filter_valid_examples(self.data)

        logger.info(f"InBatchNegativeDataset initialized with {len(self.data)} examples")

    def _load_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _filter_valid_examples(self, data: List[Dict]) -> List[Dict]:
        """Filter valid examples."""
        valid_data = []
        for example in data:
            if (example.get('positive_docs') and
                    len(example['positive_docs']) > 0 and
                    'query_text' in example):
                valid_data.append(example)
        return valid_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training example for in-batch negative training."""
        example = self.data[idx]

        # Sample one positive document
        positive_docs = example['positive_docs']
        if len(positive_docs) > self.max_positives_per_query:
            positive_docs = random.sample(positive_docs, self.max_positives_per_query)

        return {
            'query_text': example['query_text'],
            'expansion_features': example.get('expansion_features', {}),
            'positive_docs': positive_docs,
            'query_id': example.get('query_id', f'query_{idx}')
        }


class PairwiseDataset(Dataset):
    """
    Dataset for pairwise ranking training.

    Creates (query, positive, negative) triplets for triplet loss training.
    """

    def __init__(self,
                 data_path: Union[str, Path, List[Dict]],
                 pairs_per_query: int = 5,
                 random_seed: int = 42):
        """
        Initialize pairwise dataset.

        Args:
            data_path: Path to JSONL file or list of data examples
            pairs_per_query: Number of positive-negative pairs per query
            random_seed: Random seed
        """
        self.pairs_per_query = pairs_per_query

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load data and create pairs
        if isinstance(data_path, list):
            raw_data = data_path
        else:
            raw_data = self._load_jsonl(data_path)

        self.pairs = self._create_pairs(raw_data)

        logger.info(f"PairwiseDataset initialized with {len(self.pairs)} pairs")

    def _load_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _create_pairs(self, data: List[Dict]) -> List[Dict]:
        """Create positive-negative pairs for triplet training."""
        pairs = []

        for example in data:
            if not example.get('positive_docs') or not example.get('negative_docs'):
                continue

            query_text = example['query_text']
            expansion_features = example.get('expansion_features', {})
            positive_docs = example['positive_docs']
            negative_docs = example.get('negative_docs', []) + example.get('hard_negatives', [])

            if not negative_docs:
                continue

            # Create pairs
            pair_count = 0
            for pos_doc in positive_docs:
                for neg_doc in negative_docs:
                    if pair_count >= self.pairs_per_query:
                        break

                    pairs.append({
                        'query_text': query_text,
                        'expansion_features': expansion_features,
                        'positive_doc': pos_doc,
                        'negative_doc': neg_doc,
                        'query_id': example.get('query_id', 'unknown')
                    })
                    pair_count += 1

                if pair_count >= self.pairs_per_query:
                    break

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training pair."""
        return self.pairs[idx]


def collate_bi_encoder_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for bi-encoder training.

    Args:
        batch: List of examples from BiEncoderDataset

    Returns:
        Batched data suitable for bi-encoder training
    """
    query_texts = [item['query_text'] for item in batch]
    expansion_features = [item['expansion_features'] for item in batch]
    query_ids = [item['query_id'] for item in batch]

    # Collect all positive documents
    positive_docs = []
    for item in batch:
        positive_docs.extend(item['positive_docs'])

    # Collect all negative documents
    negative_docs = []
    for item in batch:
        negative_docs.extend(item['negative_docs'])

    return {
        'query_texts': query_texts,
        'expansion_features': expansion_features,
        'positive_docs': positive_docs,
        'negative_docs': negative_docs if negative_docs else None,
        'query_ids': query_ids,
        'batch_size': len(batch)
    }


def collate_in_batch_negative(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for in-batch negative training.

    Args:
        batch: List of examples from InBatchNegativeDataset

    Returns:
        Batched data for in-batch negative training
    """
    query_texts = [item['query_text'] for item in batch]
    expansion_features = [item['expansion_features'] for item in batch]
    query_ids = [item['query_id'] for item in batch]

    # For in-batch negatives, each query gets one positive
    positive_docs = []
    for item in batch:
        # Take the first positive doc (or random sample)
        if item['positive_docs']:
            positive_docs.append(item['positive_docs'][0])
        else:
            positive_docs.append("")  # Fallback

    return {
        'query_texts': query_texts,
        'expansion_features': expansion_features,
        'positive_docs': positive_docs,
        'query_ids': query_ids
    }


def collate_pairwise_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for pairwise training.

    Args:
        batch: List of pairs from PairwiseDataset

    Returns:
        Batched data for triplet loss training
    """
    query_texts = [item['query_text'] for item in batch]
    expansion_features = [item['expansion_features'] for item in batch]
    positive_docs = [item['positive_doc'] for item in batch]
    negative_docs = [item['negative_doc'] for item in batch]
    query_ids = [item['query_id'] for item in batch]

    return {
        'query_texts': query_texts,
        'expansion_features': expansion_features,
        'positive_docs': positive_docs,
        'negative_docs': negative_docs,
        'query_ids': query_ids
    }


# Factory functions
def create_bi_encoder_dataloader(data_path: Union[str, Path],
                                 batch_size: int = 16,
                                 shuffle: bool = True,
                                 num_workers: int = 0,
                                 dataset_type: str = "contrastive",
                                 **dataset_kwargs) -> DataLoader:
    """
    Create DataLoader for bi-encoder training.

    Args:
        data_path: Path to training data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        dataset_type: Type of dataset ("contrastive", "in_batch", "pairwise")
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        Configured DataLoader
    """
    if dataset_type == "contrastive":
        dataset = BiEncoderDataset(data_path, **dataset_kwargs)
        collate_fn = collate_bi_encoder_batch
    elif dataset_type == "in_batch":
        dataset = InBatchNegativeDataset(data_path, **dataset_kwargs)
        collate_fn = collate_in_batch_negative
    elif dataset_type == "pairwise":
        dataset = PairwiseDataset(data_path, **dataset_kwargs)
        collate_fn = collate_pairwise_batch
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test data
    test_data = [
        {
            'query_id': 'q1',
            'query_text': 'machine learning algorithms',
            'expansion_features': {
                'neural': {'rm_weight': 0.5, 'semantic_score': 0.8},
                'algorithm': {'rm_weight': 0.7, 'semantic_score': 0.9}
            },
            'positive_docs': ['Neural networks are ML algorithms', 'Algorithms for classification'],
            'negative_docs': ['Cooking recipes', 'Sports news'],
            'hard_negatives': ['Computer hardware', 'Software tools']
        },
        {
            'query_id': 'q2',
            'query_text': 'information retrieval',
            'expansion_features': {
                'search': {'rm_weight': 0.6, 'semantic_score': 0.7},
                'document': {'rm_weight': 0.8, 'semantic_score': 0.6}
            },
            'positive_docs': ['Document search systems', 'Text retrieval methods'],
            'negative_docs': ['Weather forecast', 'Movie reviews']
        }
    ]

    print("Testing bi-encoder datasets...")

    # Test contrastive dataset
    dataset = BiEncoderDataset(test_data, max_positives=2, max_negatives=2)
    print(f"Contrastive dataset size: {len(dataset)}")

    example = dataset[0]
    print(f"Example keys: {example.keys()}")
    print(f"Positive docs: {len(example['positive_docs'])}")
    print(f"Negative docs: {len(example['negative_docs'])}")

    # Test dataloader
    dataloader = create_bi_encoder_dataloader(
        test_data, batch_size=2, dataset_type="contrastive"
    )

    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch size: {batch['batch_size']}")
        print(f"Query texts: {len(batch['query_texts'])}")
        print(f"Positive docs: {len(batch['positive_docs'])}")
        break

    print("Bi-encoder datasets test completed!")