#!/usr/bin/env python3
"""
ANCE-PRF Reranking System

A modular system to load ANCE-PRF checkpoints and rerank TREC run files.
Uses ir_datasets for document/query handling. No FAISS dependency required.
"""

import os
import json
import pickle
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import ir_datasets
from transformers import RobertaTokenizer, RobertaConfig
from tqdm import tqdm

# Import from the existing codebase
from model import RobertaDot_NLL_LN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and tokenization."""

    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def process_document(self, doc_text: str) -> Dict[str, torch.Tensor]:
        """Process a single document into model inputs."""
        encoded = self.tokenizer.encode_plus(
            doc_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def process_query_with_feedback(self, query: str, feedback_docs: List[str]) -> Dict[str, torch.Tensor]:
        """Process query with PRF documents."""
        # Combine query with feedback documents using SEP token
        combined_text = query
        for doc in feedback_docs:
            combined_text += f" {self.tokenizer.sep_token} {doc}"

        return self.process_document(combined_text)


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

    def get_query_docs(self, query_id: str, top_k: int = None) -> List[str]:
        """Get document IDs for a query, optionally limited to top_k."""
        if query_id not in self.run_data:
            return []

        docs = [doc_id for doc_id, _ in self.run_data[query_id]]
        return docs[:top_k] if top_k else docs


class ANCEPRFModel:
    """Wrapper for ANCE-PRF model with checkpoint loading."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.doc_processor = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from checkpoint."""
        logger.info(f"Loading ANCE-PRF model from {self.checkpoint_path}")

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.checkpoint_path)

        # Load model config and weights
        config = RobertaConfig.from_pretrained(self.checkpoint_path)
        self.model = RobertaDot_NLL_LN.from_pretrained(self.checkpoint_path, config=config)
        self.model.to(self.device)
        self.model.eval()

        # Initialize document processor
        self.doc_processor = DocumentProcessor(self.tokenizer)

        logger.info("Model loaded successfully")

    def encode_query(self, query: str, feedback_docs: List[str] = None) -> np.ndarray:
        """Encode query with optional PRF documents."""
        if feedback_docs:
            inputs = self.doc_processor.process_query_with_feedback(query, feedback_docs)
        else:
            inputs = self.doc_processor.process_document(query)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            query_emb = self.model.query_emb(**inputs)
            return query_emb.cpu().numpy()

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode a batch of documents."""
        embeddings = []

        for doc in tqdm(documents, desc="Encoding documents"):
            inputs = self.doc_processor.process_document(doc)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                doc_emb = self.model.body_emb(**inputs)
                embeddings.append(doc_emb.cpu().numpy())

        return np.vstack(embeddings)


class IRDatasetHandler:
    """Handles ir_datasets integration for any dataset."""

    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None, lazy_loading: bool = False):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading

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

        # Check if dataset has documents
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

        # Check if dataset has queries
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
        """Extract text from document with flexible field handling."""
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

        return " ".join(text_parts) if text_parts else ""

    def _extract_query_text(self, query) -> str:
        """Extract text from query with flexible field handling."""
        # Common query fields to check
        text_fields = ['text', 'title', 'query', 'description', 'narrative']

        for field in text_fields:
            if hasattr(query, field):
                field_value = getattr(query, field)
                if field_value and str(field_value).strip():
                    return str(field_value).strip()

        # If no standard fields found, try to get any string attributes
        for attr_name in dir(query):
            if not attr_name.startswith('_') and attr_name not in ['query_id']:
                try:
                    attr_value = getattr(query, attr_name)
                    if isinstance(attr_value, str) and attr_value.strip():
                        return attr_value.strip()
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

    def inspect_dataset_structure(self, num_samples: int = 3) -> Dict[str, Any]:
        """Inspect the structure of the dataset by sampling a few items."""
        structure = {}

        # Inspect documents
        if hasattr(self.dataset, 'docs_iter'):
            structure['document_fields'] = []
            doc_samples = []

            try:
                for i, doc in enumerate(self.dataset.docs_iter()):
                    if i >= num_samples:
                        break

                    # Get all attributes of the document
                    doc_attrs = {}
                    for attr in dir(doc):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(doc, attr)
                                if not callable(value):
                                    doc_attrs[attr] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                            except:
                                pass

                    doc_samples.append(doc_attrs)

                    # Collect field names
                    if i == 0:
                        structure['document_fields'] = list(doc_attrs.keys())

                structure['document_samples'] = doc_samples
            except Exception as e:
                structure['document_error'] = str(e)

        # Inspect queries
        if hasattr(self.dataset, 'queries_iter'):
            structure['query_fields'] = []
            query_samples = []

            try:
                for i, query in enumerate(self.dataset.queries_iter()):
                    if i >= num_samples:
                        break

                    # Get all attributes of the query
                    query_attrs = {}
                    for attr in dir(query):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(query, attr)
                                if not callable(value):
                                    query_attrs[attr] = str(value)[:100] + "..." if len(str(value)) > 100 else str(
                                        value)
                            except:
                                pass

                    query_samples.append(query_attrs)

                    # Collect field names
                    if i == 0:
                        structure['query_fields'] = list(query_attrs.keys())

                structure['query_samples'] = query_samples
            except Exception as e:
                structure['query_error'] = str(e)

        return structure


class ANCEPRFReranker:
    """Main reranking system that combines all components."""

    def __init__(self,
                 checkpoint_path: str,
                 dataset_name: str,
                 device: str = 'cuda',
                 num_feedback_docs: int = 3,
                 cache_dir: Optional[str] = None,
                 lazy_loading: bool = False):
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.device = device
        self.num_feedback_docs = num_feedback_docs

        # Initialize components
        self.model = ANCEPRFModel(checkpoint_path, device)
        self.dataset_handler = IRDatasetHandler(dataset_name, cache_dir, lazy_loading)

        # Print dataset info
        info = self.dataset_handler.get_dataset_info()
        logger.info(f"Dataset info: {info}")

        if not info['has_docs'] or not info['has_queries']:
            logger.warning("Dataset may be missing documents or queries")

        # Optionally inspect dataset structure for debugging
        if logger.isEnabledFor(logging.DEBUG):
            structure = self.dataset_handler.inspect_dataset_structure()
            logger.debug(f"Dataset structure: {json.dumps(structure, indent=2)}")

    def inspect_dataset(self, num_samples: int = 3) -> Dict[str, Any]:
        """Inspect the dataset structure - useful for debugging new datasets."""
        return self.dataset_handler.inspect_dataset_structure(num_samples)

    def rerank_run(self,
                   run_path: str,
                   output_path: str,
                   use_prf: bool = True,
                   rerank_depth: int = 100) -> None:
        """Rerank a TREC run file and save results."""

        # Load run file
        run_loader = TRECRunLoader(run_path)

        # Process each query
        reranked_results = {}

        for qid in tqdm(run_loader.run_data.keys(), desc="Reranking queries"):
            query_text = self.dataset_handler.get_query_text(qid)
            if not query_text:
                logger.warning(f"Query {qid} not found in dataset")
                continue

            # Get documents to rerank
            doc_ids = run_loader.get_query_docs(qid, rerank_depth)
            doc_texts = self.dataset_handler.get_documents_text(doc_ids)

            # Filter out empty documents
            valid_docs = [(doc_id, doc_text) for doc_id, doc_text in zip(doc_ids, doc_texts)
                          if doc_text.strip()]

            if not valid_docs:
                logger.warning(f"No valid documents found for query {qid}")
                continue

            doc_ids, doc_texts = zip(*valid_docs)

            # Get PRF documents if enabled
            feedback_docs = None
            if use_prf and len(doc_texts) >= self.num_feedback_docs:
                feedback_docs = list(doc_texts[:self.num_feedback_docs])

            # Encode query and documents
            query_emb = self.model.encode_query(query_text, feedback_docs)
            doc_embs = self.model.encode_documents(list(doc_texts))

            # Compute similarities using simple numpy operations (no FAISS needed)
            similarities = self._compute_similarities(query_emb, doc_embs)

            # Rank by similarity (highest first)
            ranked_indices = np.argsort(similarities)[::-1]

            # Store results
            reranked_results[qid] = [
                (doc_ids[idx], similarities[idx])
                for idx in ranked_indices
            ]

        # Save reranked results
        self._save_results(reranked_results, output_path)
        logger.info(f"Reranked results saved to {output_path}")

    def _compute_similarities(self, query_emb: np.ndarray, doc_embs: np.ndarray,
                              metric: str = "cosine") -> np.ndarray:
        """Compute similarities between query and documents without FAISS."""

        if metric == "cosine":
            # Cosine similarity: (q · d) / (||q|| * ||d||)
            query_norm = np.linalg.norm(query_emb)
            doc_norms = np.linalg.norm(doc_embs, axis=1)

            # Avoid division by zero
            if query_norm == 0:
                return np.zeros(len(doc_embs))

            dot_products = np.dot(query_emb, doc_embs.T).flatten()
            similarities = dot_products / (query_norm * doc_norms + 1e-8)

        elif metric == "dot_product":
            # Simple dot product (what ANCE typically uses)
            similarities = np.dot(query_emb, doc_embs.T).flatten()

        elif metric == "euclidean":
            # Negative euclidean distance (higher = more similar)
            distances = np.linalg.norm(doc_embs - query_emb, axis=1)
            similarities = -distances

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return similarities

    def _save_results(self, results: Dict[str, List[Tuple[str, float]]], output_path: str):
        """Save results in TREC format."""
        with open(output_path, 'w') as f:
            for qid, doc_scores in results.items():
                for rank, (doc_id, score) in enumerate(doc_scores, 1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} ANCE-PRF\n")


def main():
    parser = argparse.ArgumentParser(description="ANCE-PRF Reranking System")
    parser.add_argument("--checkpoint", required=True, help="Path to ANCE-PRF checkpoint")
    parser.add_argument("--dataset", required=True, help="IR dataset name (e.g., 'msmarco-passage/dev')")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--output", required=True, help="Path to output reranked file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--num-feedback", type=int, default=3, help="Number of PRF documents")
    parser.add_argument("--rerank-depth", type=int, default=100, help="Number of top docs to rerank")
    parser.add_argument("--no-prf", action="store_true", help="Disable pseudo relevance feedback")
    parser.add_argument("--cache-dir", help="Directory to cache dataset files")
    parser.add_argument("--lazy-loading", action="store_true", help="Enable lazy loading for large datasets")
    parser.add_argument("--inspect-dataset", action="store_true", help="Inspect dataset structure and exit")
    parser.add_argument("--list-datasets", action="store_true", help="List available ir_datasets and exit")

    args = parser.parse_args()

    if args.list_datasets:
        print("Popular ir_datasets datasets:")
        print("- msmarco-passage/dev, msmarco-passage/eval")
        print("- msmarco-document/dev, msmarco-document/eval")
        print("- trec-dl-2019/passage, trec-dl-2020/passage")
        print("- trec-covid, antique, nfcorpus")
        print("- clueweb09, clueweb12")
        print("- See https://ir-datasets.com/ for complete list")
        return

    if args.inspect_dataset:
        print(f"Inspecting dataset: {args.dataset}")
        try:
            handler = IRDatasetHandler(args.dataset, lazy_loading=True)
            info = handler.get_dataset_info()
            structure = handler.inspect_dataset_structure()

            print(f"\nDataset Info:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            print(f"\nDataset Structure:")
            print(json.dumps(structure, indent=2))

        except Exception as e:
            print(f"Error inspecting dataset: {e}")
        return

    # Initialize reranker
    reranker = ANCEPRFReranker(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        num_feedback_docs=args.num_feedback,
        cache_dir=args.cache_dir,
        lazy_loading=args.lazy_loading
    )

    # Perform reranking
    reranker.rerank_run(
        run_path=args.run,
        output_path=args.output,
        use_prf=not args.no_prf,
        rerank_depth=args.rerank_depth
    )

    print(f"Reranking completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()