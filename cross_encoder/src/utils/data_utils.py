import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DocumentAwareExpansionDataset(Dataset):
    """Dataset for document-aware neural reranking training."""

    def __init__(self,
                 jsonl_data: List[Dict[str, Any]],
                 max_candidates_per_query: int = 100,
                 max_negatives_per_query: int = 50):
        """
        Initialize dataset from JSONL data.

        Args:
            jsonl_data: List of training examples from JSONL file
            max_candidates_per_query: Limit candidates per query
            max_negatives_per_query: Limit negative examples per query (for efficiency)
        """
        self.max_candidates = max_candidates_per_query
        self.max_negatives = max_negatives_per_query

        # Create training examples (query, document, features, relevance)
        self.examples = self._create_training_examples(jsonl_data)

        logger.info(f"Created dataset with {len(self.examples)} training examples")

    def _create_training_examples(self, jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create (query, document) pair examples from JSONL data."""
        examples = []

        for query_data in jsonl_data:
            query_id = query_data['query_id']
            query_text = query_data['query_text']
            expansion_features = query_data['expansion_features']
            candidates = query_data['candidates']

            # Separate positive and negative examples
            positive_candidates = [c for c in candidates if c['relevance'] > 0]
            negative_candidates = [c for c in candidates if c['relevance'] == 0]

            # Limit negatives for efficiency
            if len(negative_candidates) > self.max_negatives:
                negative_candidates = negative_candidates[:self.max_negatives]

            # Create examples for all candidates
            all_candidates = positive_candidates + negative_candidates

            for candidate in all_candidates[:self.max_candidates]:
                # Skip if no document text available
                if 'doc_text' not in candidate:
                    logger.debug(f"Skipping candidate {candidate['doc_id']} - no document text")
                    continue

                example = {
                    'query_id': query_id,
                    'query_text': query_text,
                    'expansion_features': expansion_features,
                    'doc_id': candidate['doc_id'],
                    'doc_text': candidate['doc_text'],
                    'first_stage_score': candidate.get('first_stage_score', candidate.get('score', 0.0)),
                    'relevance': candidate['relevance']
                }

                examples.append(example)

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PairwiseExpansionDataset(Dataset):
    """Dataset for pairwise ranking training (alternative approach)."""

    def __init__(self, jsonl_data: List[Dict[str, Any]], max_pairs_per_query: int = 50):
        """
        Create pairwise training examples.

        Args:
            jsonl_data: JSONL training data
            max_pairs_per_query: Maximum positive-negative pairs per query
        """
        self.max_pairs = max_pairs_per_query
        self.pairs = self._create_pairwise_examples(jsonl_data)

        logger.info(f"Created pairwise dataset with {len(self.pairs)} pairs")

    def _create_pairwise_examples(self, jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create positive-negative pairs for ranking loss."""
        pairs = []

        for query_data in jsonl_data:
            query_id = query_data['query_id']
            query_text = query_data['query_text']
            expansion_features = query_data['expansion_features']
            candidates = query_data['candidates']

            # Get positive and negative candidates with document text
            positives = [c for c in candidates if c['relevance'] > 0 and 'doc_text' in c]
            negatives = [c for c in candidates if c['relevance'] == 0 and 'doc_text' in c]

            if not positives or not negatives:
                continue

            # Create pairs
            pair_count = 0
            for pos_candidate in positives:
                for neg_candidate in negatives:
                    if pair_count >= self.max_pairs:
                        break

                    pair = {
                        'query_id': query_id,
                        'query_text': query_text,
                        'expansion_features': expansion_features,
                        'positive_doc': pos_candidate,
                        'negative_doc': neg_candidate
                    }

                    pairs.append(pair)
                    pair_count += 1

                if pair_count >= self.max_pairs:
                    break

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# Add this to the END of src/utils/data_utils.py (after the existing classes)

def expansion_collate_fn(batch):
    """
    Custom collate function for expansion dataset.

    Args:
        batch: List of examples from DocumentAwareExpansionDataset

    Returns:
        Batched data with proper handling of complex structures
    """
    return {
        'query_id': [item['query_id'] for item in batch],
        'query_text': [item['query_text'] for item in batch],
        'expansion_features': [item['expansion_features'] for item in batch],
        'doc_id': [item['doc_id'] for item in batch],
        'doc_text': [item['doc_text'] for item in batch],
        'first_stage_score': [item.get('first_stage_score', 0.0) for item in batch],
        'relevance': [item['relevance'] for item in batch]
    }


def pairwise_collate_fn(batch):
    """
    Custom collate function for pairwise ranking dataset.

    Args:
        batch: List of examples from PairwiseExpansionDataset

    Returns:
        Batched pairwise data
    """
    return {
        'query_id': [item['query_id'] for item in batch],
        'query_text': [item['query_text'] for item in batch],
        'expansion_features': [item['expansion_features'] for item in batch],
        'positive_doc': [item['positive_doc'] for item in batch],
        'negative_doc': [item['negative_doc'] for item in batch]
    }