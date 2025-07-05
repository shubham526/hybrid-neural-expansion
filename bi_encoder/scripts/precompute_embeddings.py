#!/usr/bin/env python3
"""
Precompute Embeddings Script

This script precomputes embeddings for large document collections using trained
bi-encoder models. It's designed for efficient batch processing and caching of
embeddings for faster experimentation and index building.

Key features:
- Batch processing with configurable batch sizes
- Multiple storage formats (HDF5, NumPy, Pickle)
- Resume capability for interrupted runs
- Memory-efficient processing for large collections
- Validation and statistics reporting

Usage:
    # Precompute embeddings for a document collection
    python precompute_embeddings.py \
        --model-path ./models/bi_encoder \
        --documents-file documents.jsonl \
        --output-file embeddings.hdf5 \
        --batch-size 64 \
        --format hdf5

    # Resume interrupted computation
    python precompute_embeddings.py \
        --model-path ./models/bi_encoder \
        --documents-file documents.jsonl \
        --output-file embeddings.hdf5 \
        --resume
"""

import argparse
import json
import logging
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator
from tqdm import tqdm
import gc
import psutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bi_encoder.src.models.bi_encoder import create_hybrid_bi_encoder
from bi_encoder.src.utils.embeddings import create_embedding_manager
from cross_encoder.src.utils.file_utils import ensure_dir, load_json, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, TimedOperation

logger = logging.getLogger(__name__)


class DocumentIterator:
    """
    Memory-efficient iterator for document collections.
    """

    def __init__(self,
                 documents_file: Path,
                 start_idx: int = 0,
                 max_docs: Optional[int] = None,
                 validate_format: bool = True):
        """
        Initialize document iterator.

        Args:
            documents_file: Path to documents file (JSONL format)
            start_idx: Starting index (for resuming)
            max_docs: Maximum number of documents to process
            validate_format: Whether to validate document format
        """
        self.documents_file = documents_file
        self.start_idx = start_idx
        self.max_docs = max_docs
        self.validate_format = validate_format

        # Count total documents
        self.total_docs = self._count_documents()

        if max_docs:
            self.total_docs = min(self.total_docs, max_docs)

        logger.info(f"Document iterator initialized:")
        logger.info(f"  File: {documents_file}")
        logger.info(f"  Total documents: {self.total_docs}")
        logger.info(f"  Start index: {start_idx}")

    def _count_documents(self) -> int:
        """Count total documents in file."""
        count = 0
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def __len__(self) -> int:
        return self.total_docs - self.start_idx

    def __iter__(self) -> Iterator[Tuple[str, str, int]]:
        """
        Iterate over documents.

        Yields:
            Tuple of (doc_id, doc_text, line_number)
        """
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # Skip to start index
                if line_num < self.start_idx:
                    continue

                # Stop at max docs
                if self.max_docs and line_num >= self.max_docs:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)

                    # Extract document ID and text
                    if 'doc_id' in doc and 'doc_text' in doc:
                        doc_id = doc['doc_id']
                        doc_text = doc['doc_text']
                    elif 'id' in doc and 'text' in doc:
                        doc_id = doc['id']
                        doc_text = doc['text']
                    elif 'docno' in doc and 'text' in doc:
                        doc_id = doc['docno']
                        doc_text = doc['text']
                    else:
                        if self.validate_format:
                            logger.warning(f"Line {line_num + 1}: Missing required fields")
                            continue
                        else:
                            # Use line number as ID and raw content as text
                            doc_id = str(line_num)
                            doc_text = str(doc)

                    yield doc_id, doc_text, line_num + 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num + 1}: JSON decode error - {e}")
                    continue


class EmbeddingPrecomputer:
    """
    Efficient precomputation of embeddings for large document collections.
    """

    def __init__(self,
                 model_path: Path,
                 output_file: Path,
                 batch_size: int = 32,
                 storage_format: str = 'hdf5',
                 device: str = None,
                 normalize_embeddings: bool = True):
        """
        Initialize embedding precomputer.

        Args:
            model_path: Path to trained bi-encoder model
            output_file: Path to output embeddings file
            batch_size: Batch size for embedding computation
            storage_format: Storage format ('hdf5', 'numpy', 'pickle')
            device: Device for computation ('cuda', 'cpu', or None for auto)
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        self.model_path = model_path
        self.output_file = output_file
        self.batch_size = batch_size
        self.storage_format = storage_format
        self.normalize_embeddings = normalize_embeddings

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize components
        self.model = None
        self.embedding_manager = None

        # Progress tracking
        self.processed_count = 0
        self.start_time = None

        # Resume capability
        self.resume_file = output_file.with_suffix('.resume.json')

        logger.info(f"EmbeddingPrecomputer initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Storage format: {storage_format}")

    def load_model(self) -> None:
        """Load the bi-encoder model."""
        logger.info(f"Loading model from: {self.model_path}")

        # Load model info
        model_info_file = self.model_path / 'model_info.json'
        if not model_info_file.exists():
            raise FileNotFoundError(f"Model info not found: {model_info_file}")

        model_info = load_json(model_info_file)

        # Create model
        self.model = create_hybrid_bi_encoder(
            model_name=model_info['model_name'],
            max_expansion_terms=model_info['max_expansion_terms'],
            expansion_weight=model_info.get('expansion_weight', 0.3),
            similarity_function=model_info.get('similarity_function', 'cosine'),
            device=str(self.device),
            force_hf=model_info.get('force_hf', False),
            pooling_strategy=model_info.get('pooling_strategy', 'cls')
        )

        # Load trained weights
        best_model_file = self.model_path / 'best_model.pt'
        final_model_file = self.model_path / 'final_model.pt'

        if best_model_file.exists():
            self.model.load_state_dict(torch.load(best_model_file, map_location=self.device))
            logger.info("Loaded best model weights")
        elif final_model_file.exists():
            self.model.load_state_dict(torch.load(final_model_file, map_location=self.device))
            logger.info("Loaded final model weights")
        else:
            logger.warning("No trained weights found, using random initialization")

        # Set to evaluation mode
        self.model.eval()

        # Get embedding dimension
        embedding_dim = self.model.get_embedding_dimension()

        # Initialize embedding manager
        self.embedding_manager = create_embedding_manager(
            embedding_dim=embedding_dim,
            storage_format=self.storage_format,
            normalize=self.normalize_embeddings,
            device=str(self.device)
        )

        logger.info(f"Model loaded successfully, embedding dim: {embedding_dim}")

    def check_resume_state(self) -> Tuple[int, Optional[Dict]]:
        """
        Check if we can resume from previous run.

        Returns:
            Tuple of (start_index, resume_metadata)
        """
        if not self.resume_file.exists():
            return 0, None

        try:
            resume_data = load_json(self.resume_file)
            start_idx = resume_data.get('processed_count', 0)

            logger.info(f"Found resume file: {start_idx} documents already processed")
            return start_idx, resume_data

        except Exception as e:
            logger.warning(f"Could not load resume file: {e}")
            return 0, None

    def save_resume_state(self, processed_count: int, metadata: Dict[str, Any]) -> None:
        """Save current processing state for resume capability."""
        resume_data = {
            'processed_count': processed_count,
            'last_updated': time.time(),
            'output_file': str(self.output_file),
            'metadata': metadata
        }

        save_json(resume_data, self.resume_file)

    def process_documents(self,
                          documents_file: Path,
                          max_docs: Optional[int] = None,
                          resume: bool = False,
                          save_frequency: int = 1000) -> Dict[str, Any]:
        """
        Process documents and compute embeddings.

        Args:
            documents_file: Path to documents file
            max_docs: Maximum number of documents to process
            resume: Whether to resume from previous run
            save_frequency: How often to save intermediate results

        Returns:
            Processing statistics
        """
        # Check resume state
        start_idx = 0
        if resume:
            start_idx, resume_metadata = self.check_resume_state()

        # Initialize document iterator
        doc_iterator = DocumentIterator(
            documents_file=documents_file,
            start_idx=start_idx,
            max_docs=max_docs
        )

        # Initialize progress tracking
        self.start_time = time.time()
        self.processed_count = start_idx
        total_docs = len(doc_iterator) + start_idx

        # Batch processing
        batch_texts = []
        batch_ids = []
        batch_line_nums = []

        all_embeddings = []
        all_doc_ids = []

        logger.info(f"Starting embedding computation for {len(doc_iterator)} documents")

        # Progress bar
        pbar = tqdm(total=len(doc_iterator),
                    desc="Computing embeddings",
                    initial=0)

        try:
            for doc_id, doc_text, line_num in doc_iterator:
                batch_texts.append(doc_text)
                batch_ids.append(doc_id)
                batch_line_nums.append(line_num)

                # Process batch when full
                if len(batch_texts) >= self.batch_size:
                    embeddings = self._process_batch(batch_texts, batch_ids)

                    all_embeddings.extend(embeddings)
                    all_doc_ids.extend(batch_ids)

                    self.processed_count += len(batch_texts)
                    pbar.update(len(batch_texts))

                    # Save intermediate results
                    if self.processed_count % save_frequency == 0:
                        self._save_intermediate_results(
                            all_embeddings, all_doc_ids,
                            self.processed_count, total_docs
                        )
                        self._log_progress()

                    # Clear batch
                    batch_texts = []
                    batch_ids = []
                    batch_line_nums = []

                    # Memory cleanup
                    if self.processed_count % (save_frequency * 2) == 0:
                        self._cleanup_memory()

            # Process remaining documents
            if batch_texts:
                embeddings = self._process_batch(batch_texts, batch_ids)
                all_embeddings.extend(embeddings)
                all_doc_ids.extend(batch_ids)

                self.processed_count += len(batch_texts)
                pbar.update(len(batch_texts))

            pbar.close()

            # Save final results
            self._save_final_results(all_embeddings, all_doc_ids)

            # Compute statistics
            stats = self._compute_statistics(all_embeddings, all_doc_ids)

            # Clean up resume file
            if self.resume_file.exists():
                self.resume_file.unlink()

            logger.info("Embedding computation completed successfully")
            return stats

        except Exception as e:
            # Save current state for resume
            metadata = {
                'error': str(e),
                'documents_file': str(documents_file)
            }
            self.save_resume_state(self.processed_count, metadata)
            raise

    def _process_batch(self, texts: List[str], doc_ids: List[str]) -> List[np.ndarray]:
        """Process a batch of documents."""
        with torch.no_grad():
            # Encode documents
            embeddings = self.model.encode_documents(texts)

            # Convert to CPU and numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Normalize if requested
            if self.normalize_embeddings:
                from sklearn.preprocessing import normalize
                embeddings = normalize(embeddings, norm='l2', axis=1)

            return [embeddings[i] for i in range(len(embeddings))]

    def _save_intermediate_results(self,
                                   embeddings: List[np.ndarray],
                                   doc_ids: List[str],
                                   processed_count: int,
                                   total_count: int) -> None:
        """Save intermediate results."""
        # Save current state for resume
        metadata = {
            'processed_count': processed_count,
            'total_count': total_count,
            'progress_pct': (processed_count / total_count) * 100
        }
        self.save_resume_state(processed_count, metadata)

    def _save_final_results(self,
                            embeddings: List[np.ndarray],
                            doc_ids: List[str]) -> None:
        """Save final embeddings to file."""
        logger.info(f"Saving {len(embeddings)} embeddings to: {self.output_file}")

        # Convert to tensor format
        embeddings_tensor = torch.from_numpy(np.array(embeddings))

        # Save using embedding manager
        self.embedding_manager.save_embeddings(
            embeddings=embeddings_tensor,
            ids=doc_ids,
            filepath=self.output_file,
            metadata={
                'model_path': str(self.model_path),
                'batch_size': self.batch_size,
                'normalized': self.normalize_embeddings,
                'device': str(self.device),
                'created_with': 'precompute_embeddings.py'
            }
        )

    def _compute_statistics(self,
                            embeddings: List[np.ndarray],
                            doc_ids: List[str]) -> Dict[str, Any]:
        """Compute processing statistics."""
        total_time = time.time() - self.start_time
        docs_per_second = len(embeddings) / total_time

        # Embedding statistics
        embedding_array = np.array(embeddings)
        embedding_stats = {
            'mean_norm': float(np.mean(np.linalg.norm(embedding_array, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embedding_array, axis=1))),
            'min_norm': float(np.min(np.linalg.norm(embedding_array, axis=1))),
            'max_norm': float(np.max(np.linalg.norm(embedding_array, axis=1)))
        }

        stats = {
            'total_documents': len(embeddings),
            'total_time_seconds': total_time,
            'docs_per_second': docs_per_second,
            'embedding_dimension': embedding_array.shape[1],
            'batch_size': self.batch_size,
            'device': str(self.device),
            'storage_format': self.storage_format,
            'normalized': self.normalize_embeddings,
            'embedding_stats': embedding_stats,
            'output_file': str(self.output_file),
            'model_path': str(self.model_path)
        }

        return stats

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed_time = time.time() - self.start_time
        docs_per_second = self.processed_count / elapsed_time

        # Memory usage
        memory_usage = psutil.virtual_memory().percent
        gpu_memory = "N/A"
        if torch.cuda.is_available():
            gpu_memory = f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB"

        logger.info(f"Progress: {self.processed_count} docs, "
                    f"{docs_per_second:.1f} docs/sec, "
                    f"Memory: {memory_usage:.1f}%, "
                    f"GPU: {gpu_memory}")

    def _cleanup_memory(self) -> None:
        """Clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_documents_file(documents_file: Path, sample_size: int = 100) -> Dict[str, Any]:
    """
    Validate documents file format.

    Args:
        documents_file: Path to documents file
        sample_size: Number of documents to sample for validation

    Returns:
        Validation results
    """
    logger.info(f"Validating documents file: {documents_file}")

    if not documents_file.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_file}")

    # Sample documents
    valid_docs = 0
    invalid_docs = 0
    sample_docs = []
    field_variants = set()

    with open(documents_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > sample_size:
                break

            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)

                # Check for required fields
                has_id = any(field in doc for field in ['doc_id', 'id', 'docno'])
                has_text = any(field in doc for field in ['doc_text', 'text', 'body'])

                if has_id and has_text:
                    valid_docs += 1
                    # Track field variants
                    for field in doc.keys():
                        field_variants.add(field)
                else:
                    invalid_docs += 1

                if len(sample_docs) < 5:
                    sample_docs.append(doc)

            except json.JSONDecodeError:
                invalid_docs += 1

    validation_results = {
        'total_sampled': valid_docs + invalid_docs,
        'valid_documents': valid_docs,
        'invalid_documents': invalid_docs,
        'validation_rate': valid_docs / (valid_docs + invalid_docs) if (valid_docs + invalid_docs) > 0 else 0,
        'field_variants': list(field_variants),
        'sample_documents': sample_docs[:3]  # Show first 3 for inspection
    }

    logger.info(f"Validation results:")
    logger.info(f"  Valid documents: {valid_docs}/{valid_docs + invalid_docs} "
                f"({validation_results['validation_rate']:.1%})")
    logger.info(f"  Field variants found: {field_variants}")

    if validation_results['validation_rate'] < 0.9:
        logger.warning("Low validation rate - check document format")

    return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for document collections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained bi-encoder model directory')

    # Data arguments
    parser.add_argument('--documents-file', type=str, required=True,
                        help='Path to documents JSONL file')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to output embeddings file')
    parser.add_argument('--max-docs', type=int,
                        help='Maximum number of documents to process')

    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding computation')
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'],
                        help='Device for computation (auto-detect if not specified)')

    # Storage arguments
    parser.add_argument('--format', type=str, default='hdf5',
                        choices=['hdf5', 'numpy', 'pickle'],
                        help='Storage format for embeddings')
    parser.add_argument('--normalize', action='store_true',
                        help='L2 normalize embeddings')

    # Resume and validation
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous interrupted run')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate documents file format')
    parser.add_argument('--save-frequency', type=int, default=1000,
                        help='How often to save intermediate results')

    # Other arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(Path(args.output_file).parent)
    logger = setup_experiment_logging("precompute_embeddings", args.log_level,
                                      str(output_dir / 'embedding_computation.log'))

    try:
        documents_file = Path(args.documents_file)

        # Validate documents file
        with TimedOperation(logger, "Validating documents file"):
            validation_results = validate_documents_file(documents_file)

        if args.validate_only:
            logger.info("Validation completed. Results saved to log.")
            return

        if validation_results['validation_rate'] < 0.5:
            logger.error("Documents file validation failed. Please check format.")
            sys.exit(1)

        # Initialize precomputer
        precomputer = EmbeddingPrecomputer(
            model_path=Path(args.model_path),
            output_file=Path(args.output_file),
            batch_size=args.batch_size,
            storage_format=args.format,
            device=args.device,
            normalize_embeddings=args.normalize
        )

        # Load model
        with TimedOperation(logger, "Loading bi-encoder model"):
            precomputer.load_model()

        # Process documents
        with TimedOperation(logger, "Computing embeddings"):
            stats = precomputer.process_documents(
                documents_file=documents_file,
                max_docs=args.max_docs,
                resume=args.resume,
                save_frequency=args.save_frequency
            )

        # Save statistics
        stats_file = output_dir / 'embedding_stats.json'
        save_json(stats, stats_file)

        # Log final results
        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING PRECOMPUTATION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {stats['total_documents']:,}")
        logger.info(f"Total time: {stats['total_time_seconds']:.1f}s")
        logger.info(f"Processing speed: {stats['docs_per_second']:.1f} docs/sec")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"Output file: {stats['output_file']}")
        logger.info(f"File size: {Path(stats['output_file']).stat().st_size / 1024 ** 2:.1f} MB")
        logger.info("\nNext steps:")
        logger.info("1. Build dense index:")
        logger.info(f"   python bi_encoder/scripts/create_dense_index.py --embeddings {args.output_file}")
        logger.info("2. Evaluate retrieval:")
        logger.info(f"   python bi_encoder/scripts/evaluate_retrieval.py --index-dir <index_dir>")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Embedding precomputation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()