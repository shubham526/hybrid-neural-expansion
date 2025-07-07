#!/usr/bin/env python3
"""
Create Train/Test Data for Neural Reranking

Enhanced version that supports:
1. Fold-based cross-validation (original functionality)
2. Single train/test splits (e.g., DL19 train, DL20 test)
3. Proper DL experimental setup (MS MARCO train + DL val/test)
4. Training subset support for sampled queries
5. Simple document loading with generator
"""

import argparse
import json
import logging
import sys
import random
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from cross_encoder.src.utils.file_utils import load_json, save_json, load_trec_run, ensure_dir, load_features_file
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


def load_train_subset_queries(subset_file: str) -> Set[str]:
    """
    Load training subset query IDs from TSV file.

    Args:
        subset_file: Path to TSV file with query_id and query_text columns

    Returns:
        Set of query IDs to use for training
    """
    logger.info(f"Loading training subset from: {subset_file}")

    try:
        # Try pandas first (handles various formats well)
        df = pd.read_csv(subset_file, sep='\t', dtype=str)

        # Check for expected columns
        if 'query_id' not in df.columns:
            raise ValueError("TSV file must have 'query_id' column")

        query_ids = set(df['query_id'].astype(str).tolist())
        logger.info(f"Loaded {len(query_ids)} query IDs from subset file")

        # Log a few examples
        sample_ids = list(query_ids)[:5]
        logger.info(f"Sample query IDs: {sample_ids}")

        return query_ids

    except Exception as e:
        # Fallback to manual parsing
        logger.warning(f"Pandas loading failed ({e}), trying manual parsing...")

        query_ids = set()
        with open(subset_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')

            if 'query_id' not in header:
                raise ValueError("TSV file must have 'query_id' column in header")

            qid_idx = header.index('query_id')

            for line_num, line in enumerate(f, 2):  # Start from line 2 (after header)
                fields = line.strip().split('\t')
                if len(fields) > qid_idx:
                    query_ids.add(fields[qid_idx])
                else:
                    logger.warning(f"Line {line_num}: Not enough fields, skipping")

        logger.info(f"Manually parsed {len(query_ids)} query IDs from subset file")
        return query_ids


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


def create_trec_dl_validation_split(queries: Dict[str, str],
                                    qrels: Dict[str, Dict[str, int]],
                                    first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                    val_strategy: str = "random_split",
                                    val_size: float = 0.2,
                                    random_seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Create validation split for TREC DL data.

    Args:
        queries: Query data
        qrels: Relevance judgments
        first_stage_runs: First stage runs
        val_strategy: Validation strategy
        val_size: Fraction for validation (for random_split)
        random_seed: Random seed

    Returns:
        (train_data, val_data, remaining_data) dictionaries
    """
    random.seed(random_seed)

    if val_strategy == "random_split":
        # Random split of current queries
        all_qids = list(queries.keys())
        random.shuffle(all_qids)

        val_count = int(len(all_qids) * val_size)
        val_qids = set(all_qids[:val_count])
        train_qids = set(all_qids[val_count:])

        logger.info(f"Random split: {len(train_qids)} train, {len(val_qids)} validation")

    elif val_strategy == "first_n":
        # Use first N queries for validation
        all_qids = sorted(queries.keys())
        val_count = max(1, int(len(all_qids) * val_size))

        val_qids = set(all_qids[:val_count])
        train_qids = set(all_qids[val_count:])

        logger.info(f"First N split: {len(train_qids)} train, {len(val_qids)} validation")

    elif val_strategy == "no_val":
        # No validation split
        train_qids = set(queries.keys())
        val_qids = set()

        logger.info(f"No validation: {len(train_qids)} train queries")

    else:
        raise ValueError(f"Unknown validation strategy: {val_strategy}")

    # Split data
    def filter_data(qid_set):
        return (
            {qid: queries[qid] for qid in qid_set if qid in queries},
            {qid: qrels[qid] for qid in qid_set if qid in qrels},
            {qid: first_stage_runs[qid] for qid in qid_set if qid in first_stage_runs}
        )

    train_data = filter_data(train_qids)
    val_data = filter_data(val_qids) if val_qids else ({}, {}, {})
    remaining_data = (queries, qrels, first_stage_runs)  # Keep all for reference

    return train_data, val_data, remaining_data


class DocumentLoader:
    """Load all documents into memory for fast lookup."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.documents = self._load_all_documents()
        logger.info(f"Loaded {len(self.documents)} documents into memory")

    def _load_all_documents(self) -> Dict[str, str]:
        """Load all documents into a dictionary."""
        logger.info("Loading all documents into memory...")
        documents = {}
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents", total=self.dataset.docs_count()):
            # Handle different document field formats
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                title = getattr(doc, 'title', '')
                body = doc.body
                doc_text = f"{title} {body}".strip() if title else body
            else:
                doc_text = str(doc)  # Fallback

            documents[doc.doc_id] = doc_text

        logger.info(f"Finished loading {len(documents)} documents")
        return documents

    def get_document(self, doc_id: str) -> Optional[str]:
        """Get a single document by ID - O(1) lookup!"""
        return self.documents.get(doc_id)

    def get_documents(self, doc_ids) -> Dict[str, str]:
        """Get multiple documents - all O(1) lookups!"""
        return {doc_id: self.documents[doc_id]
                for doc_id in doc_ids
                if doc_id in self.documents}


class TrainTestDataCreator:
    """Enhanced data creator supporting both fold-based and single splits with document loading."""

    def __init__(self,
                 max_candidates_per_query: int = 100,
                 ensure_positive_examples: bool = True):
        """Initialize data creator."""
        self.max_candidates_per_query = max_candidates_per_query
        self.ensure_positive_examples = ensure_positive_examples

        logger.info(f"TrainTestDataCreator initialized:")
        logger.info(f"  Max candidates per query: {max_candidates_per_query}")
        logger.info(f"  Ensure positive examples: {ensure_positive_examples}")

    def load_dataset_components(self, dataset_name: str, run_file_path: str = None,
                                query_subset: Set[str] = None, shared_documents: Dict[str, str] = None) -> Dict[
        str, Any]:
        """
        Load all components from ir_datasets with simple document loading.

        Args:
            dataset_name: Name of the dataset
            run_file_path: Optional path to run file
            query_subset: Optional set of query IDs to filter to
            shared_documents: Pre-loaded documents to avoid reloading
        """
        logger.info(f"Loading dataset components: {dataset_name}")
        if query_subset:
            logger.info(f"Will filter to {len(query_subset)} queries from subset")

        dataset = ir_datasets.load(dataset_name)

        # Load queries
        queries = {}
        for query in dataset.queries_iter():
            # Filter to subset if specified
            if query_subset and query.query_id not in query_subset:
                continue
            queries[query.query_id] = get_query_text(query)

        if query_subset:
            logger.info(f"Loaded {len(queries)} queries (filtered from subset)")
        else:
            logger.info(f"Loaded {len(queries)} queries")

        # Load qrels
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            # Filter to subset if specified
            if query_subset and qrel.query_id not in query_subset:
                continue
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        logger.info(f"Loaded qrels for {len(qrels)} queries")

        # Load first-stage runs
        first_stage_runs = defaultdict(list)

        if run_file_path:
            logger.info(f"Loading first-stage runs from: {run_file_path}")
            run_data = load_trec_run(run_file_path)

            # Filter to subset if specified
            if query_subset:
                run_data = {qid: docs for qid, docs in run_data.items()
                            if qid in query_subset}

            first_stage_runs.update(run_data)
        elif dataset.has_scoreddocs():
            logger.info("Using dataset scoreddocs for first-stage runs")
            for sdoc in dataset.scoreddocs_iter():
                # Filter to subset if specified
                if query_subset and sdoc.query_id not in query_subset:
                    continue
                first_stage_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))

            # Sort by score (descending)
            for qid in first_stage_runs:
                first_stage_runs[qid].sort(key=lambda x: x[1], reverse=True)

        else:
            raise ValueError("Need either dataset with scoreddocs or --run-file-path")

        logger.info(f"Loaded first-stage runs for {len(first_stage_runs)} queries")

        # Use shared documents or create document loader
        if shared_documents is not None:
            logger.info(f"Using shared document collection ({len(shared_documents)} documents)")
            documents = shared_documents
        else:
            logger.info("Setting up document loader...")
            document_loader = DocumentLoader(dataset)
            documents = document_loader.documents

        return {
            'queries': dict(queries),
            'qrels': dict(qrels),
            'first_stage_runs': dict(first_stage_runs),
            'documents': documents,
            'dataset': dataset
        }

    def load_ms_marco_validation_data(self, shared_documents: Dict[str, str] = None) -> Dict[str, Any]:
        """Load full MS MARCO dev/small set for validation."""
        logger.info("Loading MS MARCO dev/small set for validation")

        try:
            # Load MS MARCO dev/small set from ir_datasets
            ms_marco_dev_small = ir_datasets.load("msmarco-passage/dev/small")

            val_queries = {}
            val_qrels = defaultdict(dict)
            val_runs = defaultdict(list)

            # Load all queries
            for query in ms_marco_dev_small.queries_iter():
                val_queries[query.query_id] = get_query_text(query)

            # Load all qrels
            for qrel in ms_marco_dev_small.qrels_iter():
                val_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

            # Load all scored docs if available
            if ms_marco_dev_small.has_scoreddocs():
                for sdoc in ms_marco_dev_small.scoreddocs_iter():
                    val_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))

            # Use shared documents if available, otherwise load them
            if shared_documents is not None:
                logger.info(f"Using shared document collection for MS MARCO validation")
                documents = shared_documents
            else:
                logger.info("Loading documents for MS MARCO validation")
                document_loader = DocumentLoader(ms_marco_dev_small)
                documents = document_loader.documents

            logger.info(f"Loaded MS MARCO dev/small: {len(val_queries)} queries, {len(val_qrels)} qrels")

            return {
                'queries': dict(val_queries),
                'qrels': dict(val_qrels),
                'first_stage_runs': dict(val_runs),
                'documents': documents
            }

        except Exception as e:
            logger.warning(f"Could not load MS MARCO dev/small data: {e}")
            return None

    def create_training_example(self,
                                query_id: str,
                                query_text: str,
                                expansion_features: Dict[str, Dict[str, float]],
                                candidates: List[tuple],
                                qrels: Dict[str, int],
                                documents: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a training example for one query WITH document content."""

        # Create candidate list
        candidate_list = []
        has_positive = False

        for doc_id, score in candidates:
            relevance = qrels.get(doc_id, 0)
            if relevance > 0:
                has_positive = True

            candidate_entry = {
                'doc_id': doc_id,
                'score': float(score),
                'relevance': int(relevance)
            }

            # Add document text if available
            if doc_id in documents:
                candidate_entry['doc_text'] = documents[doc_id]
            else:
                # Skip candidates without document text for training
                logger.debug(f"Skipping {doc_id} - no document text found")
                continue

            candidate_list.append(candidate_entry)

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

    def create_proper_dl_experiment_data(self,
                                         val_year: str,  # "19" or "20"
                                         test_year: str,  # "20" or "19"
                                         features_train: Dict[str, Any],
                                         features_val: Dict[str, Any],
                                         features_test: Dict[str, Any],
                                         run_file_path_train: str = None,
                                         run_file_path_val: str = None,
                                         run_file_path_test: str = None,
                                         train_subset_file: str = None) -> Dict[str, Any]:
        """
        Create proper TREC DL experimental setup.

        Args:
            val_year: DL year for validation ("19" or "20")
            test_year: DL year for testing ("20" or "19")
            features_train: Features for MS MARCO train/judged
            features_val: Features for validation DL year
            features_test: Features for test DL year
            train_subset_file: Optional TSV file with subset of training queries

        Returns:
            Dictionary with train/val/test data
        """

        # Load training subset if specified
        train_query_subset = None
        if train_subset_file:
            train_query_subset = load_train_subset_queries(train_subset_file)
            logger.info(f"Will use {len(train_query_subset)} queries from training subset")

        # Dataset names
        train_dataset = "msmarco-passage/train/judged"
        val_dataset = f"msmarco-passage/trec-dl-20{val_year}"
        test_dataset = f"msmarco-passage/trec-dl-20{test_year}"

        logger.info(f"Creating proper DL experiment:")
        logger.info(f"  Train: {train_dataset}")
        if train_subset_file:
            logger.info(f"    Using subset from: {train_subset_file}")
        logger.info(f"  Validation: {val_dataset}")
        logger.info(f"  Test: {test_dataset}")

        # Load all dataset components
        train_components = self.load_dataset_components(train_dataset, run_file_path_train,
                                                        query_subset=train_query_subset)

        # For proper_dl mode, check if we can share documents between val/test datasets
        shared_documents = None
        if self._datasets_share_documents(val_dataset, test_dataset):
            logger.info("Validation and test datasets share the same document collection")
            # Load documents once for both val and test
            temp_dataset = ir_datasets.load(val_dataset)
            document_loader = DocumentLoader(temp_dataset)
            shared_documents = document_loader.documents

        val_components = self.load_dataset_components(val_dataset, run_file_path_val,
                                                      shared_documents=shared_documents)
        test_components = self.load_dataset_components(test_dataset, run_file_path_test,
                                                       shared_documents=shared_documents)

        # Log dataset sizes
        logger.info(f"Train queries: {len(train_components['queries'])}")
        logger.info(f"Validation queries: {len(val_components['queries'])}")
        logger.info(f"Test queries: {len(test_components['queries'])}")

        # Create examples for each split
        splits = {}

        for split_name, (components, features) in [
            ('train', (train_components, features_train)),
            ('val', (val_components, features_val)),
            ('test', (test_components, features_test))
        ]:
            if not components['queries']:
                logger.warning(f"Empty {split_name} split")
                splits[split_name] = []
                continue

            split_data = []

            # Track statistics
            queries_processed = 0
            queries_with_features = 0
            queries_with_runs = 0
            queries_with_qrels = 0

            for query_id, query_text in tqdm(components['queries'].items(), desc=f"Processing {split_name}"):
                queries_processed += 1

                # Check if we have all required data
                has_features = query_id in features
                has_runs = query_id in components['first_stage_runs']
                has_qrels = query_id in components['qrels']

                if has_features:
                    queries_with_features += 1
                if has_runs:
                    queries_with_runs += 1
                if has_qrels:
                    queries_with_qrels += 1

                if not (has_features and has_runs):
                    logger.debug(
                        f"Skipping {split_name} query {query_id} (missing features={not has_features}, runs={not has_runs})")
                    continue

                expansion_features = features[query_id]['term_features']
                candidates = components['first_stage_runs'][query_id]
                query_qrels = components['qrels'].get(query_id, {})

                # For training, optionally filter queries without positive examples
                # For val/test, keep all queries for proper evaluation
                ensure_positive = self.ensure_positive_examples and split_name == 'train'

                original_setting = self.ensure_positive_examples
                self.ensure_positive_examples = ensure_positive

                example = self.create_training_example(
                    query_id=query_id,
                    query_text=query_text,
                    expansion_features=expansion_features,
                    candidates=candidates,
                    qrels=query_qrels,
                    documents=components['documents']
                )

                self.ensure_positive_examples = original_setting

                if example is not None:
                    split_data.append(example)

            splits[split_name] = split_data

            # Log detailed statistics
            logger.info(f"{split_name.title()} split statistics:")
            logger.info(f"  Total queries: {queries_processed}")
            logger.info(f"  With features: {queries_with_features}")
            logger.info(f"  With runs: {queries_with_runs}")
            logger.info(f"  With qrels: {queries_with_qrels}")
            logger.info(f"  Final examples: {len(split_data)}")

        return splits

    def create_single_split_data(self,
                                 train_dataset_name: str,
                                 test_dataset_name: str,
                                 features_train: Dict[str, Any],
                                 features_test: Dict[str, Any],
                                 val_strategy: str = "random_split",
                                 run_file_path_train: str = None,
                                 run_file_path_test: str = None,
                                 features_val: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create single train/val/test split (e.g., DL19 train, DL20 test).

        Args:
            train_dataset_name: Name of training dataset
            test_dataset_name: Name of test dataset
            features_train: Extracted features for training data
            features_test: Extracted features for test data
            val_strategy: How to create validation set
            run_file_path_train: Optional run file for training
            run_file_path_test: Optional run file for testing

        Returns:
            Dictionary with train/val/test data
        """
        logger.info(f"Creating single split: {train_dataset_name} → {test_dataset_name}")

        # Check if both datasets use the same document collection
        shared_documents = None
        if self._datasets_share_documents(train_dataset_name, test_dataset_name):
            logger.info("Datasets share the same document collection - loading documents once")
            # Load documents from the first dataset
            temp_dataset = ir_datasets.load(train_dataset_name)
            document_loader = DocumentLoader(temp_dataset)
            shared_documents = document_loader.documents

        # Load training dataset
        train_components = self.load_dataset_components(train_dataset_name, run_file_path_train,
                                                        shared_documents=shared_documents)

        # Load test dataset (reuse documents if available)
        test_components = self.load_dataset_components(test_dataset_name, run_file_path_test,
                                                       shared_documents=shared_documents)

        # Create validation split from training data
        if val_strategy == "ms_marco":
            # Use MS MARCO dev set for validation
            val_components = self.load_ms_marco_validation_data(shared_documents=shared_documents)
            if val_components is None:
                # Fallback to random split
                logger.warning("MS MARCO validation failed, using random split")
                val_strategy = "random_split"
                features_val = features_train  # Use train features for fallback

        if val_strategy != "ms_marco" or val_components is None:
            # Create validation from training data
            train_data, val_data, _ = create_trec_dl_validation_split(
                train_components['queries'],
                train_components['qrels'],
                train_components['first_stage_runs'],
                val_strategy=val_strategy
            )

            # Update train_components with split data
            train_components['queries'], train_components['qrels'], train_components['first_stage_runs'] = train_data

            val_components = {
                'queries': val_data[0],
                'qrels': val_data[1],
                'first_stage_runs': val_data[2],
                'documents': train_components['documents']
            }
            # Use train features when splitting from training data
            features_val = features_train

        # Create examples for each split
        splits = {}

        for split_name, (components, features) in [
            ('train', (train_components, features_train)),
            ('val', (val_components, features_val)),  # Use appropriate val features
            ('test', (test_components, features_test))
        ]:
            if not components['queries']:
                logger.info(f"Skipping empty {split_name} split")
                splits[split_name] = []
                continue

            split_data = []

            for query_id, query_text in tqdm(components['queries'].items(), desc=f"Processing {split_name}"):
                if (query_id not in features or
                        query_id not in components['first_stage_runs']):
                    logger.debug(f"Skipping {split_name} query {query_id} (missing data)")
                    continue

                expansion_features = features[query_id]['term_features']
                candidates = components['first_stage_runs'][query_id]
                query_qrels = components['qrels'].get(query_id, {})

                # For test set, don't filter out queries without positive examples
                ensure_positive = self.ensure_positive_examples and split_name == 'train'

                # Temporarily override setting
                original_setting = self.ensure_positive_examples
                self.ensure_positive_examples = ensure_positive

                example = self.create_training_example(
                    query_id=query_id,
                    query_text=query_text,
                    expansion_features=expansion_features,
                    candidates=candidates,
                    qrels=query_qrels,
                    documents=components['documents']
                )

                self.ensure_positive_examples = original_setting

                if example is not None:
                    split_data.append(example)

            splits[split_name] = split_data
            logger.info(f"Created {len(split_data)} {split_name} examples")

        return splits

    def _datasets_share_documents(self, dataset1: str, dataset2: str) -> bool:
        """Check if two datasets share the same document collection."""
        # MS MARCO passage datasets share the same collection
        msmarco_passage_datasets = [
            'msmarco-passage/train',
            'msmarco-passage/train/judged',
            'msmarco-passage/dev',
            'msmarco-passage/dev/small',
            'msmarco-passage/trec-dl-2019',
            'msmarco-passage/trec-dl-2020',
            'msmarco-passage/trec-dl-2019/judged',
            'msmarco-passage/trec-dl-2020/judged'
        ]

        # Check if both datasets are MS MARCO passage datasets
        dataset1_base = dataset1.split('/')[0] + '/' + dataset1.split('/')[1] if '/' in dataset1 else dataset1
        dataset2_base = dataset2.split('/')[0] + '/' + dataset2.split('/')[1] if '/' in dataset2 else dataset2

        return (any(dataset1.startswith(d.split('/')[0] + '/' + d.split('/')[1]) for d in msmarco_passage_datasets) and
                any(dataset2.startswith(d.split('/')[0] + '/' + d.split('/')[1]) for d in msmarco_passage_datasets))

    def create_fold_data(self, fold_info: Dict[str, List[str]],
                         features: Dict[str, Any],
                         dataset_components: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Create train/test data for one fold (original functionality)."""
        queries = dataset_components['queries']
        qrels = dataset_components['qrels']
        first_stage_runs = dataset_components['first_stage_runs']
        documents = dataset_components['documents']

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
                query_id, query_text, expansion_features, candidates, query_qrels,
                documents=documents
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
            original_setting = self.ensure_positive_examples
            self.ensure_positive_examples = False

            example = self.create_training_example(
                query_id, query_text, expansion_features, candidates, query_qrels,
                documents=documents
            )

            self.ensure_positive_examples = original_setting

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
        candidates_with_doc_text = sum(
            sum(1 for c in example['candidates'] if 'doc_text' in c)
            for example in data
        )

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
            'candidates_with_doc_text': candidates_with_doc_text,
            'doc_text_coverage': candidates_with_doc_text / total_candidates if total_candidates > 0 else 0,
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

    # Mode selection
    parser.add_argument('--mode', type=str,
                        choices=['folds', 'single', 'proper_dl'],
                        default='folds',
                        help='Data creation mode: folds for cross-validation, single for train/test split, proper_dl for MS MARCO train + DL val/test')

    # Common arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for train/test files')
    parser.add_argument('--max-candidates-per-query', type=int, default=100,
                        help='Maximum candidates to include per query')
    parser.add_argument('--ensure-positive-training', action='store_true',
                        help='Ensure training queries have positive examples')

    # Fold mode arguments
    parser.add_argument('--dataset', type=str,
                        help='IR dataset name (for fold mode)')
    parser.add_argument('--features-file', type=str,
                        help='Path to extracted features file (for fold mode)')
    parser.add_argument('--folds-file', type=str,
                        help='Path to folds.json file (for fold mode)')
    parser.add_argument('--run-file-path', type=str,
                        help='Path to TREC run file (if not using dataset scoreddocs)')
    parser.add_argument('--specific-fold', type=str,
                        help='Process only specific fold (e.g., "0")')

    # Single split mode arguments
    parser.add_argument('--train-dataset', type=str,
                        help='Training dataset name (for single mode)')
    parser.add_argument('--test-dataset', type=str,
                        help='Test dataset name (for single mode)')
    parser.add_argument('--train-features-file', type=str,
                        help='Path to training features file (for single mode)')
    parser.add_argument('--test-features-file', type=str,
                        help='Path to test features file (for single mode)')
    parser.add_argument('--val-features-file', type=str,
                        help='Path to validation features file (for single mode with ms_marco strategy)')
    parser.add_argument('--val-strategy', type=str, default='random_split',
                        choices=['random_split', 'first_n', 'ms_marco', 'no_val'],
                        help='Validation set creation strategy')
    parser.add_argument('--run-file-train', type=str,
                        help='Path to training run file (for single mode)')
    parser.add_argument('--run-file-test', type=str,
                        help='Path to test run file (for single mode)')

    # Proper DL mode arguments
    parser.add_argument('--val-year', type=str, choices=['19', '20'],
                        help='DL year for validation (proper_dl mode)')
    parser.add_argument('--test-year', type=str, choices=['19', '20'],
                        help='DL year for testing (proper_dl mode)')
    parser.add_argument('--train-subset-file', type=str,
                        help='Path to TSV file with subset of training queries (proper_dl mode)')

    # Other arguments
    parser.add_argument('--save-statistics', action='store_true',
                        help='Save data statistics')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'folds':
        required_fold_args = ['dataset', 'features_file', 'folds_file']
        missing = [arg for arg in required_fold_args if not getattr(args, arg)]
        if missing:
            parser.error(f"Fold mode requires: {', '.join(missing)}")

    elif args.mode == 'single':
        required_single_args = ['train_dataset', 'test_dataset', 'train_features_file', 'test_features_file']
        missing = [arg for arg in required_single_args if not getattr(args, arg)]
        if missing:
            parser.error(f"Single mode requires: {', '.join(missing)}")

        # Check if ms_marco validation strategy requires val features file
        if args.val_strategy == 'ms_marco' and not args.val_features_file:
            parser.error("ms_marco validation strategy requires --val-features-file")

    elif args.mode == 'proper_dl':
        required_dl_args = ['val_year', 'test_year', 'train_features_file', 'val_features_file', 'test_features_file']
        missing = [arg for arg in required_dl_args if not getattr(args, arg)]
        if missing:
            parser.error(f"Proper DL mode requires: {', '.join(missing)}")

        if args.val_year == args.test_year:
            parser.error("Validation and test years must be different")

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_train_test", args.log_level,
                                      str(output_dir / 'data_creation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Create data creator
        data_creator = TrainTestDataCreator(
            max_candidates_per_query=args.max_candidates_per_query,
            ensure_positive_examples=args.ensure_positive_training
        )

        if args.mode == 'folds':
            # Original fold-based functionality
            with TimedOperation(logger, "Loading fold definitions"):
                folds = load_json(args.folds_file)
                logger.info(f"Loaded {len(folds)} folds")

                if args.specific_fold:
                    if args.specific_fold not in folds:
                        raise ValueError(f"Fold '{args.specific_fold}' not found in folds file")
                    folds = {args.specific_fold: folds[args.specific_fold]}
                    logger.info(f"Processing only fold {args.specific_fold}")

            with TimedOperation(logger, "Loading extracted features"):
                features = load_features_file(args.features_file)
                logger.info(f"Loaded features for {len(features)} queries")

            with TimedOperation(logger, "Loading dataset components"):
                dataset_components = data_creator.load_dataset_components(
                    args.dataset, args.run_file_path
                )

            # Process each fold
            for fold_id, fold_info in folds.items():
                logger.info(f"\n{'=' * 50}")
                logger.info(f"Processing Fold {fold_id}")
                logger.info(f"{'=' * 50}")

                fold_dir = ensure_dir(output_dir / f"fold_{fold_id}")

                train_qids = fold_info['training']
                test_qids = fold_info['testing']
                logger.info(f"Training queries: {len(train_qids)}")
                logger.info(f"Test queries: {len(test_qids)}")

                with TimedOperation(logger, f"Creating data for fold {fold_id}"):
                    fold_data = data_creator.create_fold_data(
                        fold_info, features, dataset_components
                    )

                # Save files
                train_file = fold_dir / 'train.jsonl'
                test_file = fold_dir / 'test.jsonl'

                save_jsonl(fold_data['train'], train_file)
                save_jsonl(fold_data['test'], test_file)

                logger.info(f"Saved training data to: {train_file}")
                logger.info(f"Saved test data to: {test_file}")

                # Save statistics
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
                    logger.info(
                        f"  With document text: {train_stats['candidates_with_doc_text']} ({train_stats['doc_text_coverage']:.1%})")
                    logger.info(f"  Positive examples: {train_stats['num_positive_examples']}")
                    logger.info(f"  Positive rate: {train_stats['positive_rate']:.3f}")

                    logger.info(f"Test data statistics:")
                    logger.info(f"  Queries: {test_stats['num_queries']}")
                    logger.info(f"  Candidates: {test_stats['num_candidates']}")
                    logger.info(
                        f"  With document text: {test_stats['candidates_with_doc_text']} ({test_stats['doc_text_coverage']:.1%})")
                    logger.info(f"  Positive examples: {test_stats['num_positive_examples']}")
                    logger.info(f"  Positive rate: {test_stats['positive_rate']:.3f}")

        elif args.mode == 'single':
            # Single split functionality
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Creating Single Split")
            logger.info(f"Train: {args.train_dataset}")
            logger.info(f"Test: {args.test_dataset}")
            logger.info(f"Validation strategy: {args.val_strategy}")
            logger.info(f"{'=' * 50}")

            with TimedOperation(logger, "Loading extracted features"):
                features_train = load_features_file(args.train_features_file)
                features_test = load_features_file(args.test_features_file)
                logger.info(f"Loaded train features for {len(features_train)} queries")
                logger.info(f"Loaded test features for {len(features_test)} queries")

                # Load validation features if specified
                features_val = None
                if args.val_features_file:
                    features_val = load_features_file(args.val_features_file)
                    logger.info(f"Loaded validation features for {len(features_val)} queries")

            with TimedOperation(logger, "Creating single split data"):
                split_data = data_creator.create_single_split_data(
                    train_dataset_name=args.train_dataset,
                    test_dataset_name=args.test_dataset,
                    features_train=features_train,
                    features_test=features_test,
                    val_strategy=args.val_strategy,
                    run_file_path_train=args.run_file_train,
                    run_file_path_test=args.run_file_test,
                    features_val=features_val
                )

            # Save files
            for split_name, split_examples in split_data.items():
                if not split_examples:
                    continue

                split_file = output_dir / f'{split_name}.jsonl'
                save_jsonl(split_examples, split_file)
                logger.info(f"Saved {split_name} data to: {split_file}")

            # Save statistics
            if args.save_statistics:
                all_stats = {}
                for split_name, split_examples in split_data.items():
                    if split_examples:
                        all_stats[f'{split_name}_stats'] = data_creator.get_data_statistics(split_examples)

                summary_stats = {
                    'train_dataset': args.train_dataset,
                    'test_dataset': args.test_dataset,
                    'val_strategy': args.val_strategy,
                    'splits': all_stats,
                    'files': {split: str(output_dir / f'{split}.jsonl')
                              for split in split_data.keys() if split_data[split]}
                }

                stats_file = output_dir / 'data_stats.json'
                save_json(summary_stats, stats_file)
                logger.info(f"Saved statistics to: {stats_file}")

                # Log key statistics
                for split_name, stats in all_stats.items():
                    logger.info(f"{split_name.replace('_stats', '').title()} data statistics:")
                    logger.info(f"  Queries: {stats['num_queries']}")
                    logger.info(f"  Candidates: {stats['num_candidates']}")
                    logger.info(
                        f"  With document text: {stats['candidates_with_doc_text']} ({stats['doc_text_coverage']:.1%})")
                    logger.info(f"  Positive examples: {stats['num_positive_examples']}")
                    logger.info(f"  Positive rate: {stats['positive_rate']:.3f}")

        elif args.mode == 'proper_dl':
            # Proper DL experimental setup
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Creating Proper DL Experimental Setup")
            logger.info(f"Train: MS MARCO passage/train/judged")
            if args.train_subset_file:
                logger.info(f"  Using subset from: {args.train_subset_file}")
            logger.info(f"Validation: TREC DL 20{args.val_year}")
            logger.info(f"Test: TREC DL 20{args.test_year}")
            logger.info(f"{'=' * 50}")

            with TimedOperation(logger, "Loading extracted features"):
                features_train = load_features_file(args.train_features_file)
                features_val = load_features_file(args.val_features_file)
                features_test = load_features_file(args.test_features_file)

                logger.info(f"Loaded train features for {len(features_train)} queries")
                logger.info(f"Loaded val features for {len(features_val)} queries")
                logger.info(f"Loaded test features for {len(features_test)} queries")

            with TimedOperation(logger, "Creating proper DL split data"):
                split_data = data_creator.create_proper_dl_experiment_data(
                    val_year=args.val_year,
                    test_year=args.test_year,
                    features_train=features_train,
                    features_val=features_val,
                    features_test=features_test,
                    train_subset_file=args.train_subset_file
                )

            # Save files
            for split_name, split_examples in split_data.items():
                if not split_examples:
                    continue

                split_file = output_dir / f'{split_name}.jsonl'
                save_jsonl(split_examples, split_file)
                logger.info(f"Saved {split_name} data to: {split_file}")

            # Save statistics
            if args.save_statistics:
                all_stats = {}
                for split_name, split_examples in split_data.items():
                    if split_examples:
                        all_stats[f'{split_name}_stats'] = data_creator.get_data_statistics(split_examples)

                summary_stats = {
                    'mode': 'proper_dl',
                    'train_dataset': 'msmarco-passage/train/judged',
                    'train_subset_file': args.train_subset_file,
                    'val_dataset': f'msmarco-passage/trec-dl-20{args.val_year}',
                    'test_dataset': f'msmarco-passage/trec-dl-20{args.test_year}',
                    'splits': all_stats,
                    'files': {split: str(output_dir / f'{split}.jsonl')
                              for split in split_data.keys() if split_data[split]}
                }

                stats_file = output_dir / 'data_stats.json'
                save_json(summary_stats, stats_file)
                logger.info(f"Saved statistics to: {stats_file}")

                # Log key statistics
                for split_name, stats in all_stats.items():
                    split_display = split_name.replace('_stats', '').title()
                    logger.info(f"{split_display} data statistics:")
                    logger.info(f"  Queries: {stats['num_queries']}")
                    logger.info(f"  Candidates: {stats['num_candidates']}")
                    logger.info(
                        f"  With document text: {stats['candidates_with_doc_text']} ({stats['doc_text_coverage']:.1%})")
                    logger.info(f"  Positive examples: {stats['num_positive_examples']}")
                    logger.info(f"  Positive rate: {stats['positive_rate']:.3f}")

        # Create overall summary
        if args.mode == 'folds':
            summary = {
                'mode': 'folds',
                'dataset': args.dataset,
                'features_file': args.features_file,
                'folds_file': args.folds_file,
                'num_folds_processed': len(folds),
                'max_candidates_per_query': args.max_candidates_per_query,
                'ensure_positive_training': args.ensure_positive_training,
                'output_directory': str(output_dir)
            }
        elif args.mode == 'single':
            summary = {
                'mode': 'single',
                'train_dataset': args.train_dataset,
                'test_dataset': args.test_dataset,
                'train_features_file': args.train_features_file,
                'test_features_file': args.test_features_file,
                'val_strategy': args.val_strategy,
                'max_candidates_per_query': args.max_candidates_per_query,
                'ensure_positive_training': args.ensure_positive_training,
                'output_directory': str(output_dir)
            }
        else:  # proper_dl mode
            summary = {
                'mode': 'proper_dl',
                'train_dataset': 'msmarco-passage/train/judged',
                'train_subset_file': args.train_subset_file,
                'val_dataset': f'msmarco-passage/trec-dl-20{args.val_year}',
                'test_dataset': f'msmarco-passage/trec-dl-20{args.test_year}',
                'train_features_file': args.train_features_file,
                'val_features_file': args.val_features_file,
                'test_features_file': args.test_features_file,
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

        if args.mode == 'folds':
            logger.info(f"Dataset: {args.dataset}")
            logger.info(f"Folds processed: {len(folds)}")
            logger.info("Next steps:")
            logger.info("1. Train neural models for each fold:")
            for fold_id in folds.keys():
                logger.info(f"   python scripts/train.py --train-file {output_dir}/fold_{fold_id}/train.jsonl")
        elif args.mode == 'single':
            logger.info(f"Train dataset: {args.train_dataset}")
            logger.info(f"Test dataset: {args.test_dataset}")
            logger.info("Next steps:")
            logger.info("1. Train neural model:")
            logger.info(
                f"   python scripts/train.py --train-file {output_dir}/train.jsonl --val-file {output_dir}/val.jsonl")
            logger.info("2. Evaluate model:")
            logger.info(f"   python scripts/evaluate.py --test-file {output_dir}/test.jsonl")
        else:  # proper_dl mode
            if args.train_subset_file:
                logger.info(f"Train: MS MARCO passage/train/judged (subset from {args.train_subset_file})")
            else:
                logger.info(f"Train: MS MARCO passage/train/judged (~532K queries)")
            logger.info(f"Validation: TREC DL 20{args.val_year}")
            logger.info(f"Test: TREC DL 20{args.test_year}")
            logger.info("Next steps:")
            logger.info("1. Train neural model:")
            logger.info(
                f"   python scripts/train.py --train-file {output_dir}/train.jsonl --val-file {output_dir}/val.jsonl")
            logger.info("2. Evaluate model:")
            logger.info(f"   python scripts/evaluate.py --test-file {output_dir}/test.jsonl")

        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Data creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()