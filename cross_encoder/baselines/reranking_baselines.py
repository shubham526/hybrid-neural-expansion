#!/usr/bin/env python3
"""
Unified reranker script supporting multiple reranking models with ir_datasets integration
Re-scores query-document pairs using Bi-Encoder, Cross-Encoder, INSTRUCTOR, DPR, or FlagLLM reranker
Now supports ir_datasets for automatic document and query loading
"""

import os
import sys
import json
import pickle
import argparse
import torch
import unicodedata
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import logging

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# ir_datasets import
try:
    import ir_datasets

    IR_DATASETS_AVAILABLE = True
except ImportError:
    IR_DATASETS_AVAILABLE = False
    print("⚠️  ir_datasets not available, dataset mode disabled")
    print("   Install with: pip install ir_datasets")

# Optional imports based on availability
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sentence_transformers.util import cos_sim

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, cross-encoder, bi-encoder, and instructor modes disabled")

try:
    from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, \
        DPRContextEncoderTokenizer
    import torch.nn.functional as F

    DPR_AVAILABLE = True
except ImportError:
    DPR_AVAILABLE = False
    print("⚠️  transformers not available, DPR mode disabled")

try:
    from FlagEmbedding import FlagLLMReranker

    FLAG_AVAILABLE = True
except ImportError:
    FLAG_AVAILABLE = False
    print("⚠️  FlagEmbedding not available, FlagLLM mode disabled")

try:
    from pygaggle.rerank.base import Query, Text
    from pygaggle.rerank.transformer import MonoT5, MonoBERT
    import spacy

    PYGAGGLE_AVAILABLE = True
except ImportError:
    PYGAGGLE_AVAILABLE = False
    print("⚠️  pygaggle not available, MonoT5 and MonoBERT modes disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean text by removing problematic characters and normalizing."""
    if not text:
        return ""

    text = str(text)

    # Handle very long texts early
    if len(text) > 100000:
        text = text[:100000]
        logger.debug("Pre-truncated very long text before cleaning")

    try:
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)

        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

    except Exception as e:
        logger.warning(f"Error in text cleaning: {e}, returning truncated original")
        text = str(text)[:10000]
        text = re.sub(r'\s+', ' ', text)

    result = text.strip()

    if not result or len(result.strip()) < 3:
        logger.debug("Text cleaning resulted in very short/empty text")
        return ""

    return result


class IRDatasetHandler:
    """Handles ir_datasets integration for any dataset."""

    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None, lazy_loading: bool = False):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading

        if not IR_DATASETS_AVAILABLE:
            raise ImportError("ir_datasets is required but not installed. Install with: pip install ir_datasets")

        try:
            self.dataset = ir_datasets.load(dataset_name)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.info("Available datasets can be found at: https://ir-datasets.com/")
            raise

        # Initialize caches
        self.doc_cache = {}
        self.query_cache = {}

        if not lazy_loading:
            self._build_caches()
        else:
            logger.info("Lazy loading enabled - documents and queries will be loaded on demand")

    def _build_caches(self):
        """Build both document and query caches."""
        self._build_doc_cache()
        self._build_query_cache()

    def _build_doc_cache(self):
        """Build document ID to text cache with flexible field handling."""
        logger.info("Building document cache...")

        if not hasattr(self.dataset, 'docs_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have documents")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_docs.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.doc_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.doc_cache)} documents from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load doc cache: {e}")

        # Build cache from dataset
        doc_count = 0
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents", total=self.dataset.docs_count()):
            doc_text = self._extract_document_text(doc)
            if doc_text.strip():  # Only cache non-empty documents
                self.doc_cache[doc.doc_id] = doc_text
                doc_count += 1

        logger.info(f"Loaded {doc_count} documents")

        # Save to cache if directory provided
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.doc_cache, f)
                logger.info(f"Saved document cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save doc cache: {e}")

    def _build_query_cache(self):
        """Build query ID to text cache with flexible field handling."""
        logger.info("Building query cache...")

        if not hasattr(self.dataset, 'queries_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have queries")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_queries.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.query_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.query_cache)} queries from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load query cache: {e}")

        # Build cache from dataset
        query_count = 0
        for query in self.dataset.queries_iter():
            query_text = self._extract_query_text(query)
            if query_text.strip():  # Only cache non-empty queries
                self.query_cache[query.query_id] = query_text
                query_count += 1

        logger.info(f"Loaded {query_count} queries")

        # Save to cache if directory provided
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.query_cache, f)
                logger.info(f"Saved query cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save query cache: {e}")

    def _extract_document_text(self, doc) -> str:
        """Extract and clean document text with robust handling."""
        text_parts = []

        # Common document fields to check (in order of preference)
        text_fields = ['title', 'text', 'body', 'content', 'abstract', 'summary']

        for field in text_fields:
            if hasattr(doc, field):
                field_value = getattr(doc, field)
                if field_value and str(field_value).strip():
                    text_parts.append(str(field_value).strip())

        # If no standard fields found, try to get any string attributes
        if not text_parts:
            for attr_name in dir(doc):
                if not attr_name.startswith('_') and attr_name not in ['doc_id']:
                    try:
                        attr_value = getattr(doc, attr_name)
                        if isinstance(attr_value, str) and attr_value.strip():
                            text_parts.append(attr_value.strip())
                    except:
                        continue

        combined_text = " ".join(text_parts) if text_parts else ""

        # Aggressive truncation for very long documents
        max_length = 50000
        if len(combined_text) > max_length:
            logger.warning(
                f"Truncating doc {getattr(doc, 'doc_id', 'unknown')} from {len(combined_text)} to {max_length} chars.")
            combined_text = combined_text[:max_length]

        # Clean the text
        return clean_text(combined_text)

    def _extract_query_text(self, query) -> str:
        """Extract text from query with flexible field handling."""
        # Common query fields to check
        text_fields = ['text', 'title', 'query', 'description', 'narrative']

        for field in text_fields:
            if hasattr(query, field):
                field_value = getattr(query, field)
                if field_value and str(field_value).strip():
                    return clean_text(str(field_value).strip())

        # If no standard fields found, try to get any string attributes
        for attr_name in dir(query):
            if not attr_name.startswith('_') and attr_name not in ['query_id']:
                try:
                    attr_value = getattr(query, attr_name)
                    if isinstance(attr_value, str) and attr_value.strip():
                        return clean_text(attr_value.strip())
                except:
                    continue

        return ""

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get document text by ID with lazy loading support."""
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]

        if self.lazy_loading:
            # Try to find document on demand
            try:
                for doc in self.dataset.docs_iter():
                    if doc.doc_id == doc_id:
                        doc_text = self._extract_document_text(doc)
                        self.doc_cache[doc_id] = doc_text  # Cache for future use
                        return doc_text
            except Exception as e:
                logger.warning(f"Error during lazy loading of document {doc_id}: {e}")

        return None

    def get_query_text(self, query_id: str) -> Optional[str]:
        """Get query text by ID with lazy loading support."""
        if query_id in self.query_cache:
            return self.query_cache[query_id]

        if self.lazy_loading:
            # Try to find query on demand
            try:
                for query in self.dataset.queries_iter():
                    if query.query_id == query_id:
                        query_text = self._extract_query_text(query)
                        self.query_cache[query_id] = query_text  # Cache for future use
                        return query_text
            except Exception as e:
                logger.warning(f"Error during lazy loading of query {query_id}: {e}")

        return None

    def get_documents_text(self, doc_ids: List[str]) -> List[str]:
        """Get multiple document texts with batch lazy loading."""
        doc_texts = []
        missing_ids = []

        # First, get cached documents
        for doc_id in doc_ids:
            if doc_id in self.doc_cache:
                doc_texts.append(self.doc_cache[doc_id])
            else:
                doc_texts.append("")  # Placeholder
                missing_ids.append((len(doc_texts) - 1, doc_id))

        # If lazy loading and we have missing documents, try to find them
        if self.lazy_loading and missing_ids:
            missing_id_set = {doc_id for _, doc_id in missing_ids}

            try:
                for doc in self.dataset.docs_iter():
                    if doc.doc_id in missing_id_set:
                        doc_text = self._extract_document_text(doc)
                        self.doc_cache[doc.doc_id] = doc_text

                        # Update the corresponding positions in doc_texts
                        for idx, doc_id in missing_ids:
                            if doc_id == doc.doc_id:
                                doc_texts[idx] = doc_text

                        missing_id_set.remove(doc.doc_id)
                        if not missing_id_set:
                            break
            except Exception as e:
                logger.warning(f"Error during batch lazy loading: {e}")

        return doc_texts

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        info = {
            'name': self.dataset_name,
            'has_docs': hasattr(self.dataset, 'docs_iter'),
            'has_queries': hasattr(self.dataset, 'queries_iter'),
            'has_qrels': hasattr(self.dataset, 'qrels_iter'),
            'cached_docs': len(self.doc_cache),
            'cached_queries': len(self.query_cache)
        }

        # Try to get additional dataset metadata
        if hasattr(self.dataset, 'documentation'):
            info['documentation'] = self.dataset.documentation()

        return info


def load_trec_run(run_file: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load TREC-style run file"""
    print(f"📋 Loading TREC run file: {run_file}")

    run_data = defaultdict(list)

    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])

                run_data[query_id].append((doc_id, score, rank))

    # Sort by rank for each query
    for query_id in run_data:
        run_data[query_id].sort(key=lambda x: x[2])  # Sort by rank

    print(f"   Loaded {len(run_data)} queries")
    return dict(run_data)


def rescore_with_bi_encoder(bi_encoder: SentenceTransformer,
                            run_data: Dict[str, List[Tuple[str, float]]],
                            dataset_handler: IRDatasetHandler,
                            batch_size: int = 64) -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using Sentence-Transformers bi-encoder"""

    print(f"🔄 Re-scoring with Sentence-Transformers bi-encoder...")
    print(f"   Batch size: {batch_size}")

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
        query_text = dataset_handler.get_query_text(query_id)
        if not query_text:
            print(f"   Warning: Query {query_id} not found in dataset")
            continue

        rescored_docs = []

        # Encode query once
        try:
            query_embedding = bi_encoder.encode(query_text, convert_to_tensor=True)

            # Process documents in batches
            for i in range(0, len(doc_list), batch_size):
                batch_docs = doc_list[i:i + batch_size]

                # Prepare batch
                doc_texts = []
                batch_doc_ids = []
                batch_original_ranks = []

                for doc_id, original_score, original_rank in batch_docs:
                    doc_text = dataset_handler.get_document_text(doc_id)
                    if doc_text:
                        doc_texts.append(doc_text)
                        batch_doc_ids.append(doc_id)
                        batch_original_ranks.append(original_rank)
                    else:
                        print(f"   Warning: Document {doc_id} not found in dataset")

                if not doc_texts:
                    continue

                # Encode documents
                doc_embeddings = bi_encoder.encode(doc_texts, convert_to_tensor=True)

                # Compute similarities using model's native similarity function
                similarities = bi_encoder.similarity(query_embedding, doc_embeddings)[
                    0]  # Get similarities for first (only) query

                # Store rescored documents
                for doc_id, similarity, orig_rank in zip(batch_doc_ids, similarities, batch_original_ranks):
                    rescored_docs.append((doc_id, float(similarity), orig_rank))

                processed_pairs += len(doc_texts)

        except Exception as e:
            print(f"   Error processing query {query_id}: {e}")
            # Fallback: use original scores
            for doc_id, orig_score, orig_rank in doc_list:
                rescored_docs.append((doc_id, orig_score, orig_rank))

        # Sort by similarity score (descending)
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def rescore_with_cross_encoder(cross_encoder: CrossEncoder,
                               run_data: Dict[str, List[Tuple[str, float]]],
                               dataset_handler: IRDatasetHandler,
                               batch_size: int = 32) -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using Sentence-Transformers cross-encoder"""

    print(f"🔄 Re-scoring with Sentence-Transformers cross-encoder...")
    print(f"   Batch size: {batch_size}")

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
        query_text = dataset_handler.get_query_text(query_id)
        if not query_text:
            print(f"   Warning: Query {query_id} not found in dataset")
            continue

        rescored_docs = []

        # Process documents in batches
        for i in range(0, len(doc_list), batch_size):
            batch_docs = doc_list[i:i + batch_size]

            # Prepare batch - create [query, document] pairs
            cross_inp = []
            batch_doc_ids = []
            batch_original_ranks = []

            for doc_id, original_score, original_rank in batch_docs:
                doc_text = dataset_handler.get_document_text(doc_id)
                if doc_text:
                    cross_inp.append([query_text, doc_text])
                    batch_doc_ids.append(doc_id)
                    batch_original_ranks.append(original_rank)
                else:
                    print(f"   Warning: Document {doc_id} not found in dataset")

            if not cross_inp:
                continue

            # Get cross-encoder scores
            try:
                cross_scores = cross_encoder.predict(cross_inp)

                # Handle single score vs array of scores
                if not isinstance(cross_scores, (list, np.ndarray)):
                    cross_scores = [cross_scores]

                # Store rescored documents
                for doc_id, cross_score, orig_rank in zip(batch_doc_ids, cross_scores, batch_original_ranks):
                    rescored_docs.append((doc_id, float(cross_score), orig_rank))

                processed_pairs += len(cross_inp)

            except Exception as e:
                print(f"   Error processing batch for query {query_id}: {e}")
                # Fallback: use original scores
                for doc_id, orig_score, orig_rank in zip(batch_doc_ids, [0.0] * len(batch_doc_ids),
                                                         batch_original_ranks):
                    rescored_docs.append((doc_id, orig_score, orig_rank))

        # Sort by cross-encoder score (descending)
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def rescore_with_instructor(instructor_model: SentenceTransformer,
                            run_data: Dict[str, List[Tuple[str, float]]],
                            dataset_handler: IRDatasetHandler,
                            query_instruction: str,
                            corpus_instruction: str,
                            batch_size: int = 64) -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using INSTRUCTOR model"""

    print(f"🔄 Re-scoring with INSTRUCTOR model...")
    print(f"   Batch size: {batch_size}")
    print(f"   Query instruction: {query_instruction}")
    print(f"   Corpus instruction: {corpus_instruction}")

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
        query_text = dataset_handler.get_query_text(query_id)
        if not query_text:
            print(f"   Warning: Query {query_id} not found in dataset")
            continue

        rescored_docs = []

        # Encode query once with instruction
        try:
            query_embedding = instructor_model.encode(query_text, prompt=query_instruction, convert_to_tensor=True)

            # Process documents in batches
            for i in range(0, len(doc_list), batch_size):
                batch_docs = doc_list[i:i + batch_size]

                # Prepare batch
                doc_texts = []
                batch_doc_ids = []
                batch_original_ranks = []

                for doc_id, original_score, original_rank in batch_docs:
                    doc_text = dataset_handler.get_document_text(doc_id)
                    if doc_text:
                        doc_texts.append(doc_text)
                        batch_doc_ids.append(doc_id)
                        batch_original_ranks.append(original_rank)
                    else:
                        print(f"   Warning: Document {doc_id} not found in dataset")

                if not doc_texts:
                    continue

                # Encode documents with instruction
                doc_embeddings = instructor_model.encode(doc_texts, prompt=corpus_instruction, convert_to_tensor=True)

                # Compute cosine similarities (INSTRUCTOR uses cosine similarity)
                similarities = cos_sim(query_embedding, doc_embeddings)[0]  # Get similarities for first (only) query

                # Store rescored documents
                for doc_id, similarity, orig_rank in zip(batch_doc_ids, similarities, batch_original_ranks):
                    rescored_docs.append((doc_id, float(similarity), orig_rank))

                processed_pairs += len(doc_texts)

        except Exception as e:
            print(f"   Error processing query {query_id}: {e}")
            # Fallback: use original scores
            for doc_id, orig_score, orig_rank in doc_list:
                rescored_docs.append((doc_id, orig_score, orig_rank))

        # Sort by similarity score (descending)
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def rescore_with_dpr(question_encoder, context_encoder, question_tokenizer, context_tokenizer,
                     run_data: Dict[str, List[Tuple[str, float]]],
                     dataset_handler: IRDatasetHandler,
                     batch_size: int = 32,
                     device: str = 'cuda:0') -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using DPR (Dense Passage Retrieval)"""

    print(f"🔄 Re-scoring with DPR...")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")

    # Move models to device
    question_encoder.to(device)
    context_encoder.to(device)
    question_encoder.eval()
    context_encoder.eval()

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    with torch.no_grad():
        for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
            query_text = dataset_handler.get_query_text(query_id)
            if not query_text:
                print(f"   Warning: Query {query_id} not found in dataset")
                continue

            rescored_docs = []

            # Encode query once
            try:
                # Tokenize and encode query
                query_inputs = question_tokenizer(
                    query_text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(device)

                query_embedding = question_encoder(**query_inputs).pooler_output

                # Process documents in batches
                for i in range(0, len(doc_list), batch_size):
                    batch_docs = doc_list[i:i + batch_size]

                    # Prepare batch
                    doc_texts = []
                    batch_doc_ids = []
                    batch_original_ranks = []

                    for doc_id, original_score, original_rank in batch_docs:
                        doc_text = dataset_handler.get_document_text(doc_id)
                        if doc_text:
                            doc_texts.append(doc_text)
                            batch_doc_ids.append(doc_id)
                            batch_original_ranks.append(original_rank)
                        else:
                            print(f"   Warning: Document {doc_id} not found in dataset")

                    if not doc_texts:
                        continue

                    # Tokenize and encode documents
                    doc_inputs = context_tokenizer(
                        doc_texts,
                        return_tensors='pt',
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(device)

                    doc_embeddings = context_encoder(**doc_inputs).pooler_output

                    # Compute dot product similarities (DPR uses dot product)
                    similarities = torch.mm(query_embedding, doc_embeddings.T)[0]  # [batch_size]

                    # Store rescored documents
                    for doc_id, similarity, orig_rank in zip(batch_doc_ids, similarities, batch_original_ranks):
                        rescored_docs.append((doc_id, float(similarity), orig_rank))

                    processed_pairs += len(doc_texts)

            except Exception as e:
                print(f"   Error processing query {query_id}: {e}")
                # Fallback: use original scores
                for doc_id, orig_score, orig_rank in doc_list:
                    rescored_docs.append((doc_id, orig_score, orig_rank))

            # Sort by similarity score (descending)
            rescored_docs.sort(key=lambda x: x[1], reverse=True)
            rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def rescore_with_flag_reranker(reranker: 'FlagLLMReranker',
                               run_data: Dict[str, List[Tuple[str, float]]],
                               dataset_handler: IRDatasetHandler,
                               batch_size: int = 16) -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using FlagLLM reranker"""

    print(f"🔄 Re-scoring with FlagLLM reranker...")
    print(f"   Batch size: {batch_size}")

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
        query_text = dataset_handler.get_query_text(query_id)
        if not query_text:
            print(f"   Warning: Query {query_id} not found in dataset")
            continue

        rescored_docs = []

        # Process documents in batches
        for i in range(0, len(doc_list), batch_size):
            batch_docs = doc_list[i:i + batch_size]

            # Prepare batch
            query_doc_pairs = []
            batch_doc_ids = []
            batch_original_ranks = []

            for doc_id, original_score, original_rank in batch_docs:
                doc_text = dataset_handler.get_document_text(doc_id)
                if doc_text:
                    query_doc_pairs.append([query_text, doc_text])
                    batch_doc_ids.append(doc_id)
                    batch_original_ranks.append(original_rank)
                else:
                    print(f"   Warning: Document {doc_id} not found in dataset")

            if not query_doc_pairs:
                continue

            # Get FlagLLM reranker scores
            try:
                flag_scores = reranker.compute_score(query_doc_pairs)

                # Handle single score vs list of scores
                if not isinstance(flag_scores, list):
                    flag_scores = [flag_scores]

                # Store rescored documents
                for doc_id, flag_score, orig_rank in zip(batch_doc_ids, flag_scores, batch_original_ranks):
                    rescored_docs.append((doc_id, float(flag_score), orig_rank))

                processed_pairs += len(query_doc_pairs)

            except Exception as e:
                print(f"   Error processing batch for query {query_id}: {e}")
                # Fallback: use original scores
                for doc_id, orig_score, orig_rank in zip(batch_doc_ids, [0.0] * len(batch_doc_ids),
                                                         batch_original_ranks):
                    rescored_docs.append((doc_id, orig_score, orig_rank))

        # Sort by FlagLLM score (descending)
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def rescore_with_pygaggle(reranker,
                          run_data: Dict[str, List[Tuple[str, float]]],
                          dataset_handler: IRDatasetHandler,
                          max_length: int = 10,
                          stride: int = 5,
                          nlp=None) -> Dict[str, List[Tuple[str, float]]]:
    """Re-score query-document pairs using MonoT5 or MonoBERT reranker"""

    reranker_type = type(reranker).__name__
    print(f"🔄 Re-scoring with {reranker_type}...")
    print(f"   Max length: {max_length} sentences")
    print(f"   Stride: {stride} sentences")

    rescored_run = {}
    total_pairs = sum(len(docs) for docs in run_data.values())
    processed_pairs = 0

    for query_id, doc_list in tqdm(run_data.items(), desc="Processing queries"):
        query_text = dataset_handler.get_query_text(query_id)
        if not query_text:
            print(f"   Warning: Query {query_id} not found in dataset")
            continue

        # Create segments for all documents
        passages = []
        for doc_id, original_score, original_rank in doc_list:
            doc_text = dataset_handler.get_document_text(doc_id)
            if not doc_text:
                print(f"   Warning: Document {doc_id} not found in dataset")
                continue

            # Split document into sentences and create segments
            if nlp:
                doc = nlp(doc_text[:10000])  # Limit to first 10k chars
                sentences = [str(sent).strip() for sent in doc.sents]

                for i in range(0, len(sentences), stride):
                    segment = ' '.join(sentences[i:i + max_length])
                    passages.append([doc_id, segment])
                    if i + max_length >= len(sentences):
                        break
            else:
                # Fallback: use the entire document as one segment
                passages.append([doc_id, doc_text])

        if not passages:
            continue

        try:
            # Rerank using pygaggle
            query = Query(query_text)
            texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]
            ranked_results = reranker.rerank(query, texts)

            # Get scores from reranker - take maximum score per document
            final_scores = {}
            for result in ranked_results:
                doc_id = result.metadata["docid"]
                if doc_id not in final_scores:
                    final_scores[doc_id] = result.score
                else:
                    if final_scores[doc_id] < result.score:
                        final_scores[doc_id] = result.score

            # Create rescored documents list
            rescored_docs = []
            for doc_id, original_score, original_rank in doc_list:
                if doc_id in final_scores:
                    rescored_docs.append((doc_id, final_scores[doc_id], original_rank))
                else:
                    # Fallback: use original score
                    rescored_docs.append((doc_id, original_score, original_rank))

            processed_pairs += len(doc_list)

        except Exception as e:
            print(f"   Error processing query {query_id}: {e}")
            # Fallback: use original scores
            rescored_docs = []
            for doc_id, orig_score, orig_rank in doc_list:
                rescored_docs.append((doc_id, orig_score, orig_rank))

        # Sort by reranker score (descending)
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        rescored_run[query_id] = rescored_docs

    print(f"   Processed {processed_pairs} query-document pairs")
    return rescored_run


def save_trec_run(run_data: Dict[str, List[Tuple[str, float]]],
                  output_file: str,
                  run_name: str):
    """Save rescored run in TREC format"""

    print(f"💾 Saving run file: {output_file}")

    with open(output_file, 'w') as f:
        for query_id, doc_list in run_data.items():
            for rank, (doc_id, score, _) in enumerate(doc_list, 1):
                # TREC format: qid Q0 docid rank score run_name
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    print(f"   Saved results for {len(run_data)} queries")


def main():
    parser = argparse.ArgumentParser(
        description="Unified reranker supporting Bi-Encoder, Cross-Encoder, INSTRUCTOR, DPR, and FlagLLM with ir_datasets")

    # Required arguments
    parser.add_argument('--input-run', required=True,
                        help='Input TREC run file to re-score')
    parser.add_argument('--dataset', required=True,
                        help='IR dataset name (e.g., msmarco-passage/dev, trec-dl-2019/passage)')
    parser.add_argument('--output-run', required=True,
                        help='Output reranked run file')

    # Dataset arguments
    parser.add_argument('--cache-dir',
                        help='Directory to cache dataset files')
    parser.add_argument('--lazy-loading', action='store_true',
                        help='Enable lazy loading for large datasets')

    # Model selection
    parser.add_argument('--reranker-type',
                        choices=['bi-encoder', 'cross-encoder', 'instructor', 'dpr', 'flag', 'monot5', 'monobert'],
                        default='cross-encoder',
                        help='Type of reranker to use')
    parser.add_argument('--model-name',
                        help='Model name (default varies by reranker type)')

    # Common arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--max-docs', type=int, default=1000,
                        help='Maximum documents to re-score per query')

    # Cross-encoder specific arguments
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length (Cross-Encoder only)')

    # INSTRUCTOR-specific arguments
    parser.add_argument('--dataset-type',
                        choices=['trec-robust', 'trec-core', 'codec', 'custom'],
                        default='custom',
                        help='Dataset type for INSTRUCTOR prompts (INSTRUCTOR only)')
    parser.add_argument('--query-instruction',
                        help='Custom query instruction (INSTRUCTOR only)')
    parser.add_argument('--corpus-instruction',
                        help='Custom corpus instruction (INSTRUCTOR only)')

    # DPR-specific arguments
    parser.add_argument('--question-encoder',
                        help='DPR question encoder model name (DPR only)')
    parser.add_argument('--context-encoder',
                        help='DPR context encoder model name (DPR only)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use (DPR only)')

    # FlagLLM-specific arguments
    parser.add_argument('--use-fp16', action='store_true',
                        help='Use FP16 precision (FlagLLM only)')
    parser.add_argument('--use-bf16', action='store_true',
                        help='Use BF16 precision (FlagLLM only)')

    # MonoT5/MonoBERT-specific arguments
    parser.add_argument('--pygaggle-max-length', type=int, default=10,
                        help='Maximum number of sentences per segment (MonoT5/MonoBERT only)')
    parser.add_argument('--pygaggle-stride', type=int, default=5,
                        help='Stride in sentences between segments (MonoT5/MonoBERT only)')

    # Utility arguments
    parser.add_argument('--list-datasets', action='store_true',
                        help='List popular ir_datasets and exit')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_datasets:
        print("Popular ir_datasets datasets:")
        print("- msmarco-passage/dev, msmarco-passage/eval")
        print("- msmarco-document/dev, msmarco-document/eval")
        print("- trec-dl-2019/passage, trec-dl-2020/passage")
        print("- trec-covid, antique, nfcorpus")
        print("- clueweb09, clueweb12")
        print("- cord19/trec-covid")
        print("- See https://ir-datasets.com/ for complete list")
        return

    print("🚀 Unified Reranker Script with ir_datasets")
    print("=" * 50)

    # Check ir_datasets availability
    if not IR_DATASETS_AVAILABLE:
        print("❌ ir_datasets is required but not installed")
        print("   Install with: pip install ir_datasets")
        return

    # Validate reranker availability
    if args.reranker_type in ['cross-encoder', 'bi-encoder', 'instructor'] and not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Cross-Encoder, Bi-Encoder, and INSTRUCTOR modes require sentence-transformers library")
        print("   Install with: pip install sentence-transformers")
        return

    if args.reranker_type == 'dpr' and not DPR_AVAILABLE:
        print("❌ DPR mode requires transformers library")
        print("   Install with: pip install transformers torch")
        return

    if args.reranker_type == 'flag' and not FLAG_AVAILABLE:
        print("❌ FlagLLM mode requires FlagEmbedding library")
        print("   Install with: pip install FlagEmbedding")
        return

    if args.reranker_type in ['monot5', 'monobert'] and not PYGAGGLE_AVAILABLE:
        print("❌ MonoT5 and MonoBERT modes require pygaggle and spacy libraries")
        print("   Install with: pip install pygaggle spacy")
        print("   Also run: python -m spacy download en_core_web_sm")
        return

    # Set default model names
    if not args.model_name:
        if args.reranker_type == 'bi-encoder':
            args.model_name = 'multi-qa-mpnet-base-cos-v1'
        elif args.reranker_type == 'cross-encoder':
            args.model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        elif args.reranker_type == 'instructor':
            args.model_name = 'hkunlp/instructor-large'
        elif args.reranker_type == 'dpr':
            args.model_name = 'facebook/dpr-question_encoder-single-nq-base'  # Default question encoder
        elif args.reranker_type == 'monot5':
            args.model_name = 'castorini/monot5-base-msmarco'  # Default MonoT5 model
        elif args.reranker_type == 'monobert':
            args.model_name = 'castorini/monobert-large-msmarco'  # Default MonoBERT model
        else:  # flag
            args.model_name = 'BAAI/bge-reranker-v2-gemma'

    print(f"🤖 Using {args.reranker_type.upper()} reranker: {args.model_name}")
    print(f"📊 Dataset: {args.dataset}")

    # Initialize dataset handler
    try:
        dataset_handler = IRDatasetHandler(
            args.dataset,
            cache_dir=args.cache_dir,
            lazy_loading=args.lazy_loading
        )

        # Print dataset info
        info = dataset_handler.get_dataset_info()
        print(f"📈 Dataset info: {info}")

        if not info['has_docs'] or not info['has_queries']:
            print("⚠️  Dataset may be missing documents or queries")

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Set INSTRUCTOR prompts if using INSTRUCTOR model
    if args.reranker_type == 'instructor':
        if args.query_instruction and args.corpus_instruction:
            # Use custom instructions
            query_instruction = args.query_instruction
            corpus_instruction = args.corpus_instruction
        else:
            # Use dataset-specific prompts
            if args.dataset_type == 'trec-robust':
                query_instruction = "Represent the TREC query for retrieving relevant news articles: "
                corpus_instruction = "Represent the news article for retrieval: "
            elif args.dataset_type == 'trec-core':
                query_instruction = "Represent the TREC query for retrieving relevant news documents: "
                corpus_instruction = "Represent the news document for retrieval: "
            elif args.dataset_type == 'codec':
                query_instruction = "Represent the research question for retrieving documents that provide a comprehensive and nuanced answer:"
                corpus_instruction = "Represent the document for its relevance in answering a complex research question:"
            else:  # custom/default
                query_instruction = "Represent the query for retrieving relevant documents: "
                corpus_instruction = "Represent the document for retrieval: "

        print(f"   Query instruction: {query_instruction}")
        print(f"   Corpus instruction: {corpus_instruction}")

    # Set DPR model paths if using DPR
    if args.reranker_type == 'dpr':
        if not args.question_encoder:
            args.question_encoder = 'facebook/dpr-question_encoder-multiset-base'
        if not args.context_encoder:
            args.context_encoder = 'facebook/dpr-ctx_encoder-multiset-base'

        print(f"   Question encoder: {args.question_encoder}")
        print(f"   Context encoder: {args.context_encoder}")

        # Check device for DPR
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            args.device = 'cpu'

    # Load data
    run_data = load_trec_run(args.input_run)

    # Limit number of documents per query if specified
    if args.max_docs > 0:
        print(f"📏 Limiting to top {args.max_docs} documents per query")
        for query_id in run_data:
            run_data[query_id] = run_data[query_id][:args.max_docs]

    # Initialize and run reranker
    if args.reranker_type == 'bi-encoder':
        # Bi-Encoder path
        bi_encoder = SentenceTransformer(args.model_name)
        rescored_run = rescore_with_bi_encoder(
            bi_encoder, run_data, dataset_handler,
            batch_size=args.batch_size
        )
        run_name = "BiEncoder"

    elif args.reranker_type == 'cross-encoder':
        # Cross-Encoder path
        print(f"   Loading with max_length={args.max_length}")
        cross_encoder = CrossEncoder(args.model_name, max_length=args.max_length)
        rescored_run = rescore_with_cross_encoder(
            cross_encoder, run_data, dataset_handler,
            batch_size=args.batch_size
        )
        run_name = "CrossEncoder"

    elif args.reranker_type == 'instructor':
        # INSTRUCTOR path
        instructor_model = SentenceTransformer(args.model_name)
        rescored_run = rescore_with_instructor(
            instructor_model, run_data, dataset_handler,
            query_instruction, corpus_instruction,
            batch_size=args.batch_size
        )
        run_name = "INSTRUCTOR"

    elif args.reranker_type == 'dpr':
        # DPR path
        print(f"   Loading DPR models...")
        question_encoder = DPRQuestionEncoder.from_pretrained(args.question_encoder)
        context_encoder = DPRContextEncoder.from_pretrained(args.context_encoder)
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.question_encoder)
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.context_encoder)

        rescored_run = rescore_with_dpr(
            question_encoder, context_encoder, question_tokenizer, context_tokenizer,
            run_data, dataset_handler,
            batch_size=args.batch_size,
            device=args.device
        )
        run_name = "DPR"

    elif args.reranker_type in ['monot5', 'monobert']:
        print(f"   Loading spacy sentencizer...")
        nlp = spacy.blank("en")
        nlp.add_pipe('sentencizer')

        if args.reranker_type == 'monot5':
            reranker = MonoT5(args.model_name)
            run_name = "MonoT5"
        else:
            reranker = MonoBERT(args.model_name)
            run_name = "MonoBERT"

        rescored_run = rescore_with_pygaggle(
            reranker, run_data, dataset_handler,
            max_length=args.pygaggle_max_length,
            stride=args.pygaggle_stride,
            nlp=nlp
        )

    else:
        # FlagLLM path
        reranker_kwargs = {}
        if args.use_fp16:
            print("   Using FP16 precision")
            reranker_kwargs['use_fp16'] = True
        elif args.use_bf16:
            print("   Using BF16 precision")
            reranker_kwargs['use_bf16'] = True

        reranker = FlagLLMReranker(args.model_name, **reranker_kwargs)
        rescored_run = rescore_with_flag_reranker(
            reranker, run_data, dataset_handler,
            batch_size=args.batch_size
        )
        run_name = "FlagLLM_Reranker"

    # Save results
    save_trec_run(rescored_run, args.output_run, run_name)

    print(f"\n✅ {args.reranker_type.upper()} reranking complete!")
    print(f"📁 Output saved to: {args.output_run}")

    # Print summary
    total_queries = len(rescored_run)
    total_docs = sum(len(docs) for docs in rescored_run.values())
    avg_docs_per_query = total_docs / total_queries if total_queries > 0 else 0

    print(f"\n📊 Summary:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Reranker: {args.reranker_type.upper()}")
    if args.reranker_type == 'dpr':
        print(f"   Question Encoder: {args.question_encoder}")
        print(f"   Context Encoder: {args.context_encoder}")
    elif args.reranker_type in ['monot5', 'monobert']:
        print(f"   Model: {args.model_name}")
        print(f"   Max length: {args.pygaggle_max_length} sentences")
        print(f"   Stride: {args.pygaggle_stride} sentences")
    else:
        print(f"   Model: {args.model_name}")
    print(f"   Queries processed: {total_queries}")
    print(f"   Total documents: {total_docs}")
    print(f"   Avg docs per query: {avg_docs_per_query:.1f}")


if __name__ == '__main__':
    main()