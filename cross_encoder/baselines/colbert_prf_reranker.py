#!/usr/bin/env python3
"""
ColBERT-PRF Reranking System

A modular system to load ColBERT checkpoints and rerank TREC run files.
Uses ir_datasets for document/query handling and PyTerrier ColBERT for PRF.
"""

import os
import json
import pickle
import logging
import argparse
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyterrier as pt
import ir_datasets
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def get_query_docs_with_scores(self, query_id: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Get document IDs with scores for a query."""
        if query_id not in self.run_data:
            return []

        docs_scores = self.run_data[query_id]
        return docs_scores[:top_k] if top_k else docs_scores


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
        try:
            docs_total = self.dataset.docs_count() if hasattr(self.dataset, 'docs_count') else None
        except:
            docs_total = None

        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents", total=docs_total):
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


class ColBERTPRFReranker:
    """Main reranking system using ColBERT-PRF."""

    def __init__(self,
                 checkpoint_path: str,
                 dataset_name: str,
                 device: str = 'cuda',
                 cache_dir: Optional[str] = None,
                 lazy_loading: bool = False,
                 prf_k: int = 5,
                 alpha: float = 0.8,
                 beta: float = 0.2):
        self.checkpoint_path = checkpoint_path
        self.dataset_name = dataset_name
        self.device = device
        self.prf_k = prf_k
        self.alpha = alpha
        self.beta = beta

        # Check CUDA availability
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'

        logger.info(f"Using device: {self.device}")

        # Initialize PyTerrier if not started
        if not pt.started():
            pt.java.init()  # Use new initialization method

        # Initialize dataset handler
        self.dataset_handler = IRDatasetHandler(dataset_name, cache_dir, lazy_loading)

        # Print dataset info
        info = self.dataset_handler.get_dataset_info()
        logger.info(f"Dataset info: {info}")

        if not info['has_docs'] or not info['has_queries']:
            logger.warning("Dataset may be missing documents or queries")

        # Import ColBERT modules with version compatibility
        try:
            # Check PyTerrier version compatibility
            import pyterrier as pt_version_check
            pt_version = getattr(pt_version_check, '__version__', 'unknown')
            logger.info(f"PyTerrier version: {pt_version}")

            # Try to import ColBERT modules
            from pyterrier_colbert.indexing import ColBERTIndexer
            from pyterrier_colbert.ranking import ColBERTFactory

            self.ColBERTIndexer = ColBERTIndexer
            self.ColBERTFactory = ColBERTFactory
            logger.info("Successfully imported PyTerrier ColBERT modules")

        except ImportError as e:
            logger.error(f"Failed to import PyTerrier ColBERT: {e}")
            logger.error("Please install with: pip install git+https://github.com/terrierteam/pyterrier_colbert.git")
            raise

    def _create_colbert_index_for_query(self, query_id: str, doc_ids: List[str],
                                        doc_texts: List[str], index_path: str) -> int:
        """Create ColBERT index for a single query's candidates."""
        # Prepare documents for ColBERT
        documents = []
        for doc_id, doc_text in zip(doc_ids, doc_texts):
            if doc_text.strip():
                documents.append({
                    'docno': doc_id,
                    'text': doc_text
                })

        if not documents:
            raise ValueError(f"No valid documents to index for query {query_id}")

        # Set CUDA environment variables if using GPU
        if self.device == 'cuda':
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

        # Create ColBERT index using the correct NEW API
        index_path_obj = Path(index_path)
        index_root = str(index_path_obj.parent)
        index_name = index_path_obj.name

        logger.debug(f"Creating ColBERT index: root={index_root}, name={index_name}")

        try:
            indexer = self.ColBERTIndexer(
                checkpoint=self.checkpoint_path,
                index_root=index_root,
                index_name=index_name,
                chunksize=3.0,
                gpu=(self.device == 'cuda')
            )

            logger.debug(f"ColBERT indexer created successfully for query {query_id}")
            indexer.index(documents)
            logger.debug(f"ColBERT indexing completed for query {query_id}")

        except Exception as e:
            logger.error(f"ColBERT indexer error for query {query_id}: {e}")
            raise

        return len(documents)

    def _process_single_query(self, query_id: str, work_dir: Path) -> Dict[str, Any]:
        """Process a single query with ColBERT PRF."""
        try:
            query_text = self.dataset_handler.get_query_text(query_id)
            if not query_text:
                return {
                    'query_id': query_id,
                    'success': False,
                    'error': 'Query not found in dataset',
                    'results': []
                }

            # Get documents for this query from the run file
            doc_ids_scores = self.run_loader.get_query_docs_with_scores(query_id, self.rerank_depth)
            if not doc_ids_scores:
                return {
                    'query_id': query_id,
                    'success': False,
                    'error': 'No documents found in run file',
                    'results': []
                }

            doc_ids = [doc_id for doc_id, _ in doc_ids_scores]
            doc_texts = self.dataset_handler.get_documents_text(doc_ids)

            # Filter out empty documents
            valid_docs = [(doc_id, doc_text, score) for (doc_id, score), doc_text in
                          zip(doc_ids_scores, doc_texts) if doc_text.strip()]

            if not valid_docs:
                return {
                    'query_id': query_id,
                    'success': False,
                    'error': 'No valid documents with text',
                    'results': []
                }

            # Create temporary index for this query
            query_work_dir = work_dir / f"query_{query_id}"
            query_work_dir.mkdir(exist_ok=True)
            index_path = query_work_dir / "colbert_index"

            # Extract valid data
            valid_doc_ids, valid_doc_texts, valid_scores = zip(*valid_docs)

            # Debug: Print what we're about to index
            logger.debug(f"Query {query_id}: Indexing {len(valid_docs)} documents")
            logger.debug(f"Index path: {index_path}")

            # Create ColBERT index
            num_docs = self._create_colbert_index_for_query(
                query_id, list(valid_doc_ids), list(valid_doc_texts), str(index_path)
            )

            logger.debug(f"Query {query_id}: Successfully indexed {num_docs} documents")

            # Load ColBERT factory - this might be where the error occurs
            logger.debug(f"Query {query_id}: Loading ColBERT factory from {index_path}")

            try:
                colbert = self.ColBERTFactory.from_dataset(
                    dataset=str(index_path),
                    variant="colbert",
                    checkpoint=self.checkpoint_path
                )
            except Exception as factory_error:
                logger.error(f"Query {query_id}: ColBERT factory error: {factory_error}")
                return {
                    'query_id': query_id,
                    'success': False,
                    'error': f'ColBERT factory failed: {factory_error}',
                    'results': []
                }

            # Create query DataFrame
            query_df = pd.DataFrame([{
                'qid': query_id,
                'query': query_text
            }])

            # Create initial ranking DataFrame
            initial_ranking = []
            for i, (doc_id, score) in enumerate(zip(valid_doc_ids, valid_scores)):
                initial_ranking.append({
                    'qid': query_id,
                    'docno': doc_id,
                    'score': score,
                    'rank': i + 1
                })
            initial_ranking_df = pd.DataFrame(initial_ranking)

            # Create pipeline: initial retrieval -> ColBERT PRF -> ColBERT scoring
            initial_retriever = pt.Transformer.from_df(initial_ranking_df, uniform=True)

            # ColBERT PRF pipeline
            try:
                colbert_prf = colbert.prf(
                    k=self.prf_k,  # Number of docs for PRF
                    alpha=self.alpha,  # Weight for original query
                    beta=self.beta  # Weight for expansion
                )
            except Exception as prf_error:
                logger.error(f"Query {query_id}: ColBERT PRF creation error: {prf_error}")
                return {
                    'query_id': query_id,
                    'success': False,
                    'error': f'ColBERT PRF failed: {prf_error}',
                    'results': []
                }

            pipeline = initial_retriever >> colbert_prf >> colbert.text_scorer()

            # Run pipeline
            results = pipeline(query_df)

            # Convert to our format
            query_results = []
            for rank, (_, row) in enumerate(results.iterrows(), 1):
                query_results.append({
                    'query_id': str(row['qid']),
                    'doc_id': str(row['docno']),
                    'rank': rank,
                    'score': float(row['score'])
                })

            return {
                'query_id': query_id,
                'success': True,
                'results': query_results,
                'num_docs_indexed': num_docs,
                'num_results': len(query_results)
            }

        except Exception as e:
            logger.error(f"Query {query_id}: Unexpected error: {e}")
            import traceback
            logger.error(f"Query {query_id}: Traceback: {traceback.format_exc()}")
            return {
                'query_id': query_id,
                'success': False,
                'error': str(e),
                'results': []
            }

        finally:
            # Clean up query-specific index
            if 'query_work_dir' in locals() and query_work_dir.exists():
                shutil.rmtree(query_work_dir, ignore_errors=True)

    def rerank_run(self,
                   run_path: str,
                   output_path: str,
                   rerank_depth: int = 100) -> None:
        """Rerank a TREC run file and save results."""

        # Load run file
        self.run_loader = TRECRunLoader(run_path)
        self.rerank_depth = rerank_depth

        logger.info(f"Loaded run file with {len(self.run_loader.run_data)} queries")
        logger.info(f"Starting ColBERT-PRF reranking...")
        logger.info(f"  PRF k: {self.prf_k}")
        logger.info(f"  Alpha: {self.alpha}, Beta: {self.beta}")
        logger.info(f"  Rerank depth: {rerank_depth}")
        logger.info(f"  Device: {self.device}")

        # Create working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Process each query
            all_results = []
            successful_queries = 0
            failed_queries = 0

            for query_id in tqdm(self.run_loader.run_data.keys(), desc="Processing queries"):
                result = self._process_single_query(query_id, work_dir)

                if result['success']:
                    successful_queries += 1
                    all_results.extend(result['results'])
                    logger.debug(f"✓ {query_id}: {result['num_results']} results")
                else:
                    failed_queries += 1
                    logger.warning(f"✗ {query_id}: {result['error']}")

        # Save results
        if all_results:
            # Sort by query_id, then by rank
            all_results.sort(key=lambda x: (x['query_id'], x['rank']))
            self._save_results(all_results, output_path)

        logger.info(f"ColBERT-PRF reranking completed!")
        logger.info(f"  Successful queries: {successful_queries}")
        logger.info(f"  Failed queries: {failed_queries}")
        logger.info(f"  Total results: {len(all_results)}")
        logger.info(f"  Results saved to: {output_path}")

    def _save_results(self, results: List[Dict], output_path: str):
        """Save results in TREC format."""
        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"{result['query_id']} Q0 {result['doc_id']} "
                        f"{result['rank']} {result['score']:.6f} ColBERT-PRF\n")


def main():
    parser = argparse.ArgumentParser(description="ColBERT-PRF Reranking System")
    parser.add_argument("--checkpoint", required=True, help="Path to ColBERT checkpoint")
    parser.add_argument("--dataset", required=True, help="IR dataset name (e.g., 'msmarco-passage/dev')")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--output", required=True, help="Path to output reranked file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--rerank-depth", type=int, default=100, help="Number of top docs to rerank")
    parser.add_argument("--prf-k", type=int, default=5, help="Number of PRF documents")
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for original query in PRF")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight for expansion terms in PRF")
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
    reranker = ColBERTPRFReranker(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        cache_dir=args.cache_dir,
        lazy_loading=args.lazy_loading,
        prf_k=args.prf_k,
        alpha=args.alpha,
        beta=args.beta
    )

    # Perform reranking
    reranker.rerank_run(
        run_path=args.run,
        output_path=args.output,
        rerank_depth=args.rerank_depth
    )

    print(f"Reranking completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()