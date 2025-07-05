#!/usr/bin/env python3
"""
Create Dense Index Script - CLI wrapper for existing indexing functionality

This script provides a command-line interface to create dense indices using
the existing DenseIndex class from bi_encoder/src/utils/indexing.py.

Usage:
    python create_dense_index.py \
        --model-path path/to/trained/bi_encoder \
        --documents-file documents.jsonl \
        --output-dir ./dense_index \
        --index-type IndexFlatIP \
        --batch-size 32

The main difference from the existing indexing.py is that this script:
1. Provides a CLI interface
2. Handles document loading from files
3. Integrates with trained bi-encoder models
4. Provides progress tracking and logging
"""

import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bi_encoder.src.models.bi_encoder import create_hybrid_bi_encoder
from bi_encoder.src.utils.indexing import create_dense_index
from bi_encoder.src.utils.embeddings import create_embedding_manager
from cross_encoder.src.utils.file_utils import ensure_dir, load_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, TimedOperation

logger = logging.getLogger(__name__)


class DenseIndexBuilder:
    """
    Builder for creating dense indices from documents using trained bi-encoder models.

    This is essentially a CLI wrapper around the existing DenseIndex functionality.
    """

    def __init__(self,
                 model_path: Path = None,
                 model_config: Dict[str, Any] = None,
                 index_type: str = "IndexFlatIP",
                 use_gpu: bool = True,
                 batch_size: int = 32):
        """
        Initialize dense index builder.

        Args:
            model_path: Path to trained bi-encoder model
            model_config: Model configuration if loading from scratch
            index_type: FAISS index type
            use_gpu: Whether to use GPU for indexing
            batch_size: Batch size for encoding documents
        """
        self.model_path = model_path
        self.model_config = model_config
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        # Will be initialized later
        self.model = None
        self.dense_index = None
        self.embedding_manager = None

    def load_model(self) -> None:
        """Load bi-encoder model."""
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading trained model from: {self.model_path}")
            # Load model info
            model_info_file = self.model_path / 'model_info.json'
            if model_info_file.exists():
                model_info = load_json(model_info_file)

                # Create model with saved configuration
                self.model = create_hybrid_bi_encoder(
                    model_name=model_info['model_name'],
                    max_expansion_terms=model_info['max_expansion_terms'],
                    expansion_weight=model_info.get('expansion_weight', 0.3),
                    similarity_function=model_info.get('similarity_function', 'cosine'),
                    force_hf=model_info.get('force_hf', False),
                    pooling_strategy=model_info.get('pooling_strategy', 'cls')
                )

                # Load trained weights
                best_model_file = self.model_path / 'best_model.pt'
                final_model_file = self.model_path / 'final_model.pt'

                if best_model_file.exists():
                    self.model.load_state_dict(torch.load(best_model_file, map_location=self.model.device))
                    logger.info("Loaded best model weights")
                elif final_model_file.exists():
                    self.model.load_state_dict(torch.load(final_model_file, map_location=self.model.device))
                    logger.info("Loaded final model weights")
                else:
                    logger.warning("No trained weights found, using random initialization")

            else:
                raise FileNotFoundError(f"Model info file not found: {model_info_file}")

        elif self.model_config:
            logger.info("Creating new model from configuration")
            self.model = create_hybrid_bi_encoder(**self.model_config)

        else:
            raise ValueError("Either model_path or model_config must be provided")

        # Set to evaluation mode
        self.model.eval()
        logger.info(f"Model loaded on device: {self.model.device}")

    def load_documents(self, documents_file: Path) -> tuple[List[str], List[str]]:
        """
        Load documents from file.

        Args:
            documents_file: Path to documents file (JSONL format)

        Returns:
            Tuple of (document_texts, document_ids)
        """
        logger.info(f"Loading documents from: {documents_file}")

        documents = []
        doc_ids = []

        with open(documents_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading documents"), 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)

                    # Extract document ID and text
                    if 'doc_id' in doc and 'doc_text' in doc:
                        doc_ids.append(doc['doc_id'])
                        documents.append(doc['doc_text'])
                    elif 'id' in doc and 'text' in doc:
                        doc_ids.append(doc['id'])
                        documents.append(doc['text'])
                    else:
                        logger.warning(f"Line {line_num}: Missing required fields (doc_id/id, doc_text/text)")
                        continue

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
                    continue

        logger.info(f"Loaded {len(documents)} documents")
        return documents, doc_ids

    def create_index(self, documents: List[str], document_ids: List[str]) -> None:
        """
        Create dense index from documents.

        Args:
            documents: List of document texts
            document_ids: List of document IDs
        """
        # Get embedding dimension
        embedding_dim = self.model.get_embedding_dimension()

        # Create dense index
        self.dense_index = create_dense_index(
            embedding_dim=embedding_dim,
            index_type=self.index_type,
            use_gpu=self.use_gpu,
            normalize=True
        )

        # Create embedding manager for efficient processing
        self.embedding_manager = create_embedding_manager(
            embedding_dim=embedding_dim,
            storage_format='numpy',  # Use numpy for intermediate storage
            normalize=True
        )

        # Compute embeddings
        logger.info("Computing document embeddings...")
        embeddings = self.embedding_manager.compute_embeddings(
            texts=documents,
            model=self.model,
            batch_size=self.batch_size,
            show_progress=True,
            ids=document_ids
        )

        # Build index
        logger.info("Building dense index...")
        self.dense_index.build_index(
            embeddings=embeddings,
            document_ids=document_ids
        )

        # Log index info
        index_info = self.dense_index.get_index_info()
        logger.info("Index creation completed:")
        for key, value in index_info.items():
            logger.info(f"  {key}: {value}")

    def save_index(self, output_dir: Path) -> None:
        """
        Save the created index.

        Args:
            output_dir: Output directory for index
        """
        if self.dense_index is None:
            raise ValueError("No index created yet. Call create_index() first.")

        logger.info(f"Saving index to: {output_dir}")
        self.dense_index.save_index(output_dir, save_embeddings=True)

        # Save model info for future reference
        if hasattr(self.model, 'model_name'):
            index_metadata = {
                'model_name': self.model.model_name,
                'embedding_dim': self.model.get_embedding_dimension(),
                'index_type': self.index_type,
                'use_gpu': self.use_gpu,
                'created_with': 'create_dense_index.py'
            }

            metadata_file = output_dir / 'creation_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(index_metadata, f, indent=2)

        logger.info("Index saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Create dense index using trained bi-encoder model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model-path', type=str,
                        help='Path to trained bi-encoder model directory')
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Model name (if creating new model)')

    # Data arguments
    parser.add_argument('--documents-file', type=str, required=True,
                        help='Path to documents JSONL file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for dense index')

    # Index arguments
    parser.add_argument('--index-type', type=str, default='IndexFlatIP',
                        choices=['IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat', 'IndexIVFPQ', 'IndexHNSW'],
                        help='FAISS index type')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for FAISS indexing')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for document encoding')

    # Other arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_dense_index", args.log_level,
                                      str(output_dir / 'indexing.log'))

    try:
        # Initialize builder
        builder = DenseIndexBuilder(
            model_path=Path(args.model_path) if args.model_path else None,
            model_config={'model_name': args.model_name} if not args.model_path else None,
            index_type=args.index_type,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )

        # Load model
        with TimedOperation(logger, "Loading bi-encoder model"):
            builder.load_model()

        # Load documents
        with TimedOperation(logger, "Loading documents"):
            documents, doc_ids = builder.load_documents(Path(args.documents_file))

        # Create index
        with TimedOperation(logger, "Creating dense index"):
            builder.create_index(documents, doc_ids)

        # Save index
        with TimedOperation(logger, "Saving index"):
            builder.save_index(output_dir)

        logger.info("\n" + "=" * 60)
        logger.info("DENSE INDEX CREATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Index saved to: {output_dir}")
        logger.info(f"Indexed {len(documents)} documents")
        logger.info(f"Index type: {args.index_type}")
        logger.info("\nNext steps:")
        logger.info("1. Test the index:")
        logger.info(f"   python bi_encoder/scripts/evaluate_retrieval.py --index-dir {output_dir}")
        logger.info("2. Use for retrieval in your applications")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Dense index creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()