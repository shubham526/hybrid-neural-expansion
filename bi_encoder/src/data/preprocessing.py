"""
Bi-Encoder Data Preprocessing

This module provides preprocessing utilities for bi-encoder training data,
including document sampling strategies, data augmentation, and quality filtering.
"""

import logging
import random
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentSampler:
    """
    Intelligent document sampling for bi-encoder training.

    Provides various strategies for sampling positive and negative documents
    to create balanced and effective training data.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize document sampler."""
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(f"DocumentSampler initialized with seed {random_seed}")

    def sample_positives(self,
                         positive_docs: List[str],
                         num_samples: int,
                         strategy: str = 'random') -> List[str]:
        """
        Sample positive documents using specified strategy.

        Args:
            positive_docs: List of positive document texts
            num_samples: Number of documents to sample
            strategy: Sampling strategy ('random', 'diverse', 'longest', 'shortest')

        Returns:
            List of sampled positive documents
        """
        if not positive_docs:
            return []

        num_samples = min(num_samples, len(positive_docs))

        if strategy == 'random':
            return random.sample(positive_docs, num_samples)

        elif strategy == 'diverse':
            return self._diverse_sampling(positive_docs, num_samples)

        elif strategy == 'longest':
            # Prefer longer documents
            sorted_docs = sorted(positive_docs, key=len, reverse=True)
            return sorted_docs[:num_samples]

        elif strategy == 'shortest':
            # Prefer shorter documents
            sorted_docs = sorted(positive_docs, key=len)
            return sorted_docs[:num_samples]

        else:
            raise ValueError(f"Unknown positive sampling strategy: {strategy}")

    def sample_negatives(self,
                         negative_docs: List[str],
                         hard_negatives: List[str],
                         num_random_negatives: int,
                         num_hard_negatives: int,
                         strategy: str = 'mixed') -> List[str]:
        """
        Sample negative documents with mixing of random and hard negatives.

        Args:
            negative_docs: List of random negative documents
            hard_negatives: List of hard negative documents
            num_random_negatives: Number of random negatives to sample
            num_hard_negatives: Number of hard negatives to sample
            strategy: Sampling strategy ('mixed', 'hard_only', 'random_only')

        Returns:
            List of sampled negative documents
        """
        sampled_negatives = []

        if strategy == 'mixed':
            # Sample from both random and hard negatives
            if negative_docs and num_random_negatives > 0:
                random_samples = min(num_random_negatives, len(negative_docs))
                sampled_negatives.extend(random.sample(negative_docs, random_samples))

            if hard_negatives and num_hard_negatives > 0:
                hard_samples = min(num_hard_negatives, len(hard_negatives))
                sampled_negatives.extend(random.sample(hard_negatives, hard_samples))

        elif strategy == 'hard_only':
            # Only use hard negatives
            if hard_negatives:
                total_needed = num_random_negatives + num_hard_negatives
                total_available = min(total_needed, len(hard_negatives))
                sampled_negatives.extend(random.sample(hard_negatives, total_available))

        elif strategy == 'random_only':
            # Only use random negatives
            if negative_docs:
                total_needed = num_random_negatives + num_hard_negatives
                total_available = min(total_needed, len(negative_docs))
                sampled_negatives.extend(random.sample(negative_docs, total_available))

        else:
            raise ValueError(f"Unknown negative sampling strategy: {strategy}")

        return sampled_negatives

    def _diverse_sampling(self, documents: List[str], num_samples: int) -> List[str]:
        """
        Sample diverse documents based on length and lexical diversity.

        This is a simple diversity measure - could be enhanced with
        embedding-based diversity or other sophisticated methods.
        """
        if len(documents) <= num_samples:
            return documents

        # Score documents by diversity (length + unique words)
        doc_scores = []
        for doc in documents:
            words = set(doc.lower().split())
            diversity_score = len(doc) * 0.1 + len(words)  # Simple diversity metric
            doc_scores.append((doc, diversity_score))

        # Sort by diversity and take diverse samples
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Take samples spread across the diversity spectrum
        selected_docs = []
        step = len(doc_scores) // num_samples

        for i in range(num_samples):
            idx = i * step
            if idx < len(doc_scores):
                selected_docs.append(doc_scores[idx][0])

        # Fill remaining with random samples if needed
        while len(selected_docs) < num_samples:
            remaining_docs = [doc for doc, _ in doc_scores if doc not in selected_docs]
            if remaining_docs:
                selected_docs.append(random.choice(remaining_docs))
            else:
                break

        return selected_docs


class DataAugmenter:
    """
    Data augmentation techniques for bi-encoder training.
    """

    def __init__(self, random_seed: int = 42):
        """Initialize data augmenter."""
        self.random_seed = random_seed
        random.seed(random_seed)

        logger.info("DataAugmenter initialized")

    def augment_query(self, query: str, method: str = 'synonym_replace') -> str:
        """
        Augment query text using specified method.

        Args:
            query: Original query text
            method: Augmentation method ('synonym_replace', 'word_drop', 'word_shuffle')

        Returns:
            Augmented query text
        """
        if method == 'word_drop':
            return self._random_word_drop(query, drop_prob=0.1)
        elif method == 'word_shuffle':
            return self._word_shuffle(query, shuffle_prob=0.2)
        elif method == 'synonym_replace':
            # Placeholder for synonym replacement
            # Would need external library like NLTK or word embeddings
            return query
        else:
            return query

    def _random_word_drop(self, text: str, drop_prob: float = 0.1) -> str:
        """Randomly drop words from text."""
        words = text.split()
        if len(words) <= 2:  # Don't drop from very short texts
            return text

        kept_words = [word for word in words if random.random() > drop_prob]

        # Ensure at least half the words remain
        if len(kept_words) < len(words) // 2:
            return text

        return ' '.join(kept_words)

    def _word_shuffle(self, text: str, shuffle_prob: float = 0.2) -> str:
        """Randomly shuffle adjacent words."""
        words = text.split()
        if len(words) <= 2:
            return text

        result_words = words.copy()

        for i in range(len(words) - 1):
            if random.random() < shuffle_prob:
                # Swap with next word
                result_words[i], result_words[i + 1] = result_words[i + 1], result_words[i]

        return ' '.join(result_words)

    def create_augmented_examples(self,
                                  examples: List[Dict[str, Any]],
                                  augmentation_ratio: float = 0.2,
                                  methods: List[str] = None) -> List[Dict[str, Any]]:
        """
        Create augmented training examples.

        Args:
            examples: Original training examples
            augmentation_ratio: Fraction of examples to augment
            methods: List of augmentation methods to use

        Returns:
            Original examples + augmented examples
        """
        if methods is None:
            methods = ['word_drop', 'word_shuffle']

        augmented_examples = []
        num_to_augment = int(len(examples) * augmentation_ratio)

        # Randomly select examples to augment
        selected_examples = random.sample(examples, min(num_to_augment, len(examples)))

        for example in selected_examples:
            for method in methods:
                augmented_example = example.copy()
                augmented_example['query_text'] = self.augment_query(
                    example['query_text'], method
                )
                # Add marker to identify augmented examples
                augmented_example['is_augmented'] = True
                augmented_example['augmentation_method'] = method
                augmented_examples.append(augmented_example)

        logger.info(f"Created {len(augmented_examples)} augmented examples")
        return examples + augmented_examples


class DataFilter:
    """
    Quality filtering for bi-encoder training data.
    """

    def __init__(self):
        """Initialize data filter."""
        logger.info("DataFilter initialized")

    def filter_by_length(self,
                         examples: List[Dict[str, Any]],
                         min_query_length: int = 3,
                         max_query_length: int = 200,
                         min_doc_length: int = 10,
                         max_doc_length: int = 2000) -> List[Dict[str, Any]]:
        """
        Filter examples by text length constraints.

        Args:
            examples: List of training examples
            min_query_length: Minimum query length (characters)
            max_query_length: Maximum query length (characters)
            min_doc_length: Minimum document length (characters)
            max_doc_length: Maximum document length (characters)

        Returns:
            Filtered examples
        """
        filtered_examples = []

        for example in examples:
            query_text = example.get('query_text', '')

            # Check query length
            if not (min_query_length <= len(query_text) <= max_query_length):
                continue

            # Check document lengths
            valid_example = True

            for doc_list_key in ['positive_docs', 'negative_docs', 'hard_negatives']:
                if doc_list_key in example and example[doc_list_key]:
                    for doc in example[doc_list_key]:
                        if not (min_doc_length <= len(doc) <= max_doc_length):
                            valid_example = False
                            break

                if not valid_example:
                    break

            if valid_example:
                filtered_examples.append(example)

        logger.info(f"Length filtering: {len(examples)} -> {len(filtered_examples)} examples")
        return filtered_examples

    def filter_by_quality(self,
                          examples: List[Dict[str, Any]],
                          min_positive_docs: int = 1,
                          min_total_negatives: int = 1,
                          require_expansion_features: bool = True) -> List[Dict[str, Any]]:
        """
        Filter examples by data quality requirements.

        Args:
            examples: List of training examples
            min_positive_docs: Minimum number of positive documents required
            min_total_negatives: Minimum total negative documents required
            require_expansion_features: Whether expansion features are required

        Returns:
            Quality-filtered examples
        """
        filtered_examples = []

        for example in examples:
            # Check positive documents
            positive_docs = example.get('positive_docs', [])
            if len(positive_docs) < min_positive_docs:
                continue

            # Check negative documents
            negative_docs = example.get('negative_docs', [])
            hard_negatives = example.get('hard_negatives', [])
            total_negatives = len(negative_docs) + len(hard_negatives)

            if total_negatives < min_total_negatives:
                continue

            # Check expansion features
            if require_expansion_features:
                expansion_features = example.get('expansion_features', {})
                if not expansion_features:
                    continue

            # Check for valid text content
            if not example.get('query_text', '').strip():
                continue

            # Check documents are not empty
            if any(not doc.strip() for doc in positive_docs):
                continue

            filtered_examples.append(example)

        logger.info(f"Quality filtering: {len(examples)} -> {len(filtered_examples)} examples")
        return filtered_examples

    def remove_duplicates(self,
                          examples: List[Dict[str, Any]],
                          dedup_by: str = 'query_text') -> List[Dict[str, Any]]:
        """
        Remove duplicate examples.

        Args:
            examples: List of training examples
            dedup_by: Field to use for deduplication ('query_text', 'query_id')

        Returns:
            Deduplicated examples
        """
        seen = set()
        deduplicated = []

        for example in examples:
            key = example.get(dedup_by, '')
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(example)

        logger.info(f"Deduplication: {len(examples)} -> {len(deduplicated)} examples")
        return deduplicated


class BiEncoderPreprocessor:
    """
    Main preprocessing pipeline for bi-encoder training data.
    """

    def __init__(self,
                 random_seed: int = 42,
                 enable_augmentation: bool = False,
                 enable_filtering: bool = True):
        """
        Initialize bi-encoder preprocessor.

        Args:
            random_seed: Random seed for reproducibility
            enable_augmentation: Whether to enable data augmentation
            enable_filtering: Whether to enable quality filtering
        """
        self.random_seed = random_seed
        self.enable_augmentation = enable_augmentation
        self.enable_filtering = enable_filtering

        # Initialize components
        self.sampler = DocumentSampler(random_seed)
        self.augmenter = DataAugmenter(random_seed) if enable_augmentation else None
        self.filter = DataFilter() if enable_filtering else None

        logger.info(f"BiEncoderPreprocessor initialized:")
        logger.info(f"  Augmentation: {enable_augmentation}")
        logger.info(f"  Filtering: {enable_filtering}")
        logger.info(f"  Random seed: {random_seed}")

    def preprocess_dataset(self,
                           input_path: Path,
                           output_path: Path,
                           preprocessing_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete preprocessing pipeline.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output preprocessed JSONL file
            preprocessing_config: Configuration for preprocessing steps

        Returns:
            Preprocessing statistics
        """
        if preprocessing_config is None:
            preprocessing_config = self._get_default_config()

        # Load data
        logger.info(f"Loading data from: {input_path}")
        with open(input_path, 'r') as f:
            examples = [json.loads(line) for line in f if line.strip()]

        initial_count = len(examples)
        logger.info(f"Loaded {initial_count} examples")

        stats = {
            'initial_examples': initial_count,
            'steps': []
        }

        # Step 1: Quality filtering
        if self.enable_filtering:
            logger.info("Step 1: Quality filtering...")
            examples = self.filter.filter_by_quality(
                examples,
                min_positive_docs=preprocessing_config.get('min_positive_docs', 1),
                min_total_negatives=preprocessing_config.get('min_total_negatives', 1),
                require_expansion_features=preprocessing_config.get('require_expansion_features', True)
            )
            stats['steps'].append(('quality_filtering', len(examples)))

        # Step 2: Length filtering
        if self.enable_filtering:
            logger.info("Step 2: Length filtering...")
            examples = self.filter.filter_by_length(
                examples,
                min_query_length=preprocessing_config.get('min_query_length', 3),
                max_query_length=preprocessing_config.get('max_query_length', 200),
                min_doc_length=preprocessing_config.get('min_doc_length', 10),
                max_doc_length=preprocessing_config.get('max_doc_length', 2000)
            )
            stats['steps'].append(('length_filtering', len(examples)))

        # Step 3: Deduplication
        if self.enable_filtering:
            logger.info("Step 3: Deduplication...")
            examples = self.filter.remove_duplicates(
                examples,
                dedup_by=preprocessing_config.get('dedup_by', 'query_text')
            )
            stats['steps'].append(('deduplication', len(examples)))

        # Step 4: Document sampling optimization
        logger.info("Step 4: Document sampling optimization...")
        examples = self._optimize_document_sampling(examples, preprocessing_config)
        stats['steps'].append(('sampling_optimization', len(examples)))

        # Step 5: Data augmentation
        if self.enable_augmentation:
            logger.info("Step 5: Data augmentation...")
            examples = self.augmenter.create_augmented_examples(
                examples,
                augmentation_ratio=preprocessing_config.get('augmentation_ratio', 0.2),
                methods=preprocessing_config.get('augmentation_methods', ['word_drop'])
            )
            stats['steps'].append(('augmentation', len(examples)))

        # Save preprocessed data
        logger.info(f"Saving preprocessed data to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        stats['final_examples'] = len(examples)
        stats['preprocessing_config'] = preprocessing_config

        # Log summary
        logger.info("Preprocessing completed:")
        logger.info(f"  Initial examples: {initial_count}")
        logger.info(f"  Final examples: {len(examples)}")
        logger.info(f"  Retention rate: {len(examples) / initial_count:.1%}")

        return stats

    def _optimize_document_sampling(self,
                                    examples: List[Dict[str, Any]],
                                    config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize document sampling for training efficiency.

        Args:
            examples: List of training examples
            config: Preprocessing configuration

        Returns:
            Examples with optimized document sampling
        """
        optimized_examples = []

        max_positives = config.get('max_positives_per_query', 3)
        max_negatives = config.get('max_negatives_per_query', 7)
        max_hard_negatives = config.get('max_hard_negatives_per_query', 5)

        positive_strategy = config.get('positive_sampling_strategy', 'random')
        negative_strategy = config.get('negative_sampling_strategy', 'mixed')

        for example in tqdm(examples, desc="Optimizing sampling"):
            optimized_example = example.copy()

            # Sample positive documents
            if 'positive_docs' in example:
                optimized_example['positive_docs'] = self.sampler.sample_positives(
                    example['positive_docs'],
                    max_positives,
                    positive_strategy
                )

            # Sample negative documents
            negative_docs = example.get('negative_docs', [])
            hard_negatives = example.get('hard_negatives', [])

            sampled_negatives = self.sampler.sample_negatives(
                negative_docs,
                hard_negatives,
                max_negatives,
                max_hard_negatives,
                negative_strategy
            )

            # Split back into regular and hard negatives if needed
            if negative_strategy == 'mixed':
                # Keep original separation
                random_count = min(max_negatives, len(negative_docs))
                optimized_example['negative_docs'] = sampled_negatives[:random_count]
                optimized_example['hard_negatives'] = sampled_negatives[random_count:]
            else:
                optimized_example['negative_docs'] = sampled_negatives
                optimized_example['hard_negatives'] = []

            optimized_examples.append(optimized_example)

        return optimized_examples

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'min_positive_docs': 1,
            'min_total_negatives': 1,
            'require_expansion_features': True,
            'min_query_length': 3,
            'max_query_length': 200,
            'min_doc_length': 10,
            'max_doc_length': 2000,
            'dedup_by': 'query_text',
            'max_positives_per_query': 3,
            'max_negatives_per_query': 7,
            'max_hard_negatives_per_query': 5,
            'positive_sampling_strategy': 'random',
            'negative_sampling_strategy': 'mixed',
            'augmentation_ratio': 0.2,
            'augmentation_methods': ['word_drop']
        }

    def analyze_dataset(self, data_path: Path) -> Dict[str, Any]:
        """
        Analyze dataset statistics.

        Args:
            data_path: Path to dataset JSONL file

        Returns:
            Dataset analysis statistics
        """
        logger.info(f"Analyzing dataset: {data_path}")

        with open(data_path, 'r') as f:
            examples = [json.loads(line) for line in f if line.strip()]

        # Basic statistics
        total_examples = len(examples)

        # Query statistics
        query_lengths = [len(ex.get('query_text', '')) for ex in examples]

        # Document statistics
        positive_counts = [len(ex.get('positive_docs', [])) for ex in examples]
        negative_counts = [len(ex.get('negative_docs', [])) for ex in examples]
        hard_negative_counts = [len(ex.get('hard_negatives', [])) for ex in examples]

        # Expansion feature statistics
        expansion_feature_counts = [len(ex.get('expansion_features', {})) for ex in examples]

        analysis = {
            'total_examples': total_examples,
            'query_length_stats': {
                'mean': np.mean(query_lengths),
                'std': np.std(query_lengths),
                'min': np.min(query_lengths),
                'max': np.max(query_lengths)
            },
            'positive_docs_stats': {
                'mean': np.mean(positive_counts),
                'std': np.std(positive_counts),
                'min': np.min(positive_counts),
                'max': np.max(positive_counts)
            },
            'negative_docs_stats': {
                'mean': np.mean(negative_counts),
                'std': np.std(negative_counts),
                'min': np.min(negative_counts),
                'max': np.max(negative_counts)
            },
            'hard_negatives_stats': {
                'mean': np.mean(hard_negative_counts),
                'std': np.std(hard_negative_counts),
                'min': np.min(hard_negative_counts),
                'max': np.max(hard_negative_counts)
            },
            'expansion_features_stats': {
                'mean': np.mean(expansion_feature_counts),
                'std': np.std(expansion_feature_counts),
                'min': np.min(expansion_feature_counts),
                'max': np.max(expansion_feature_counts)
            }
        }

        logger.info("Dataset Analysis:")
        logger.info(f"  Total examples: {total_examples}")
        logger.info(f"  Avg query length: {analysis['query_length_stats']['mean']:.1f}")
        logger.info(f"  Avg positive docs: {analysis['positive_docs_stats']['mean']:.1f}")
        logger.info(f"  Avg negative docs: {analysis['negative_docs_stats']['mean']:.1f}")
        logger.info(f"  Avg expansion features: {analysis['expansion_features_stats']['mean']:.1f}")

        return analysis


# Factory functions
def create_preprocessor(enable_augmentation: bool = False,
                        enable_filtering: bool = True,
                        random_seed: int = 42) -> BiEncoderPreprocessor:
    """Create bi-encoder preprocessor with specified configuration."""
    return BiEncoderPreprocessor(
        random_seed=random_seed,
        enable_augmentation=enable_augmentation,
        enable_filtering=enable_filtering
    )


def preprocess_bi_encoder_data(input_path: Path,
                               output_path: Path,
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function for preprocessing bi-encoder data.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        config: Preprocessing configuration

    Returns:
        Preprocessing statistics
    """
    preprocessor = create_preprocessor()
    return preprocessor.preprocess_dataset(input_path, output_path, config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test data
    test_data = [
        {
            'query_id': 'q1',
            'query_text': 'machine learning algorithms',
            'expansion_features': {'neural': {'rm_weight': 0.5, 'semantic_score': 0.8}},
            'positive_docs': ['Neural networks are ML algorithms'] * 5,  # Many positives
            'negative_docs': ['Cooking recipes', 'Sports news'],
            'hard_negatives': ['Computer hardware', 'Software tools']
        },
        {
            'query_id': 'q2',
            'query_text': 'x',  # Too short
            'expansion_features': {},
            'positive_docs': ['Short doc'],
            'negative_docs': []
        }
    ]

    print("Testing bi-encoder preprocessing...")

    # Test document sampler
    sampler = DocumentSampler()
    positive_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    sampled = sampler.sample_positives(positive_docs, 3, 'random')
    print(f"Sampled positives: {len(sampled)}/3")

    # Test data filter
    filter_obj = DataFilter()
    filtered = filter_obj.filter_by_quality(test_data)
    print(f"Quality filtering: {len(test_data)} -> {len(filtered)}")

    # Test length filtering
    length_filtered = filter_obj.filter_by_length(test_data, min_query_length=5)
    print(f"Length filtering: {len(test_data)} -> {len(length_filtered)}")

    print("Bi-encoder preprocessing test completed!")