#!/usr/bin/env python3
"""
ColBERT Index Creation Script

Creates a ColBERT index from ir_datasets collections.
"""

import os
import argparse
import logging
from pathlib import Path
import re
import unicodedata
import sys

import pyterrier as pt
import ir_datasets
from tqdm import tqdm

# This assumes your script is 4 levels deep from the root as shown in the screenshot
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_colbert_index(dataset_name: str,
                         checkpoint_path: str,
                         index_root: str,
                         index_name: str,
                         chunksize: float = 6.0):
    """
    Create ColBERT index from ir_datasets collection.

    Args:
        dataset_name: ir_datasets name (e.g., 'disks45/nocr/trec-robust-2004')
        checkpoint_path: Path to ColBERT checkpoint
        index_root: Directory where index will be stored
        index_name: Name of the index
        chunksize: Chunk size for indexing (GB)
    """

    # Initialize PyTerrier
    if not pt.started():
        pt.java.init()

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset = ir_datasets.load(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Check if dataset has documents
    if not hasattr(dataset, 'docs_iter'):
        logger.error(f"Dataset {dataset_name} does not have documents")
        return

    # Create index directory
    os.makedirs(index_root, exist_ok=True)

    # Document generator for ColBERT indexer
    def document_generator():
        """Document generator with a workaround to skip a problematic data range."""
        logger.info("Generating documents for ColBERT indexing...")

        doc_count = 0
        skipped_count = 0

        # We now need the index, so we use enumerate
        for i, doc in enumerate(
                tqdm(dataset.docs_iter(), desc="Processing documents", total=dataset.docs_count(), unit="docs")):
            try:
                doc_id = getattr(doc, 'doc_id', None)
                if not doc_id:
                    logger.warning("Found a document with no doc_id. Skipping.")
                    skipped_count += 1
                    continue

                doc_text = extract_document_text(doc)

                if not doc_text or not doc_text.strip():
                    logger.warning(f"Skipping doc_id {doc_id} because it resulted in empty text.")
                    skipped_count += 1
                    continue

                yield {
                    'docno': doc_id,
                    'text': doc_text
                }
                doc_count += 1

            except Exception as e:
                logger.error(f"Critical error processing document at index {i}. Skipping it. Error: {e}")
                skipped_count += 1
                continue

        logger.info(f"Generator completed: {doc_count:,} documents yielded.")
        logger.warning(f"Skipped a total of {skipped_count:,} documents (including the problem zone).")

    # Create ColBERT indexer
    try:
        from pyterrier_colbert.indexing import ColBERTIndexer

        logger.info("Creating ColBERT indexer...")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Index root: {index_root}")
        logger.info(f"  Index name: {index_name}")
        logger.info(f"  Chunk size: {chunksize} GB")

        indexer = ColBERTIndexer(
            checkpoint=checkpoint_path,
            index_root=index_root,
            index_name=index_name,
            chunksize=chunksize,
            ids=True,
            nbits=2,
            fp16=True

        )

        # Override the batch size after creation
        indexer.args.bsize = 32  # Much smaller batch size

    except ImportError as e:
        logger.error(f"Failed to import ColBERT indexer: {e}")
        logger.error("Please install: pip install git+https://github.com/terrierteam/pyterrier_colbert.git")
        raise

    # Start indexing
    logger.info("Starting ColBERT indexing...")
    try:
        indexer.index(document_generator())
        logger.info("✅ ColBERT indexing completed successfully!")

        # Print index location
        index_path = Path(index_root) / index_name
        logger.info(f"Index created at: {index_path}")

        # List index files
        if index_path.exists():
            index_files = list(index_path.glob("*"))
            logger.info(f"Index contains {len(index_files)} files")

    except Exception as e:
        logger.error(f"ColBERT indexing failed: {e}")
        raise


def extract_document_text(doc) -> str:
    """
    Extract, truncate, and clean text from an ir_datasets document object.
    """
    try:
        text_parts = []
        # Handle different document field formats
        if hasattr(doc, 'text'):
            text_parts.append(doc.text)
        elif hasattr(doc, 'body'):
            title = getattr(doc, 'title', '')
            body = doc.body
            if title:
                text_parts.append(f"{title} {body}")
            else:
                text_parts.append(body)
        else:
            # Common document fields to check
            text_fields = ['title', 'content', 'abstract', 'summary']
            for field in text_fields:
                if hasattr(doc, field):
                    field_value = getattr(doc, field)
                    if field_value and str(field_value).strip():
                        text_parts.append(str(field_value).strip())

        # Combine text parts into a single string
        combined_text = " ".join(text_parts) if text_parts else ""

        # **KEY FIX**: Truncate the text *before* cleaning to avoid processing huge documents.
        # ColBERT has a maximum token limit anyway, so this is safe.
        max_length = 50000
        if len(combined_text) > max_length:
            logger.warning(
                f"Truncating doc {getattr(doc, 'doc_id', 'unknown')} from {len(combined_text)} to {max_length} chars.")
            combined_text = combined_text[:max_length]

        # Now, clean the potentially much smaller text.
        return clean_text(combined_text)

    except Exception as e:
        logger.error(f"Error processing document {getattr(doc, 'doc_id', 'unknown')}: {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean text by removing problematic characters and forcing it into ASCII,
    which is safe for most standard ColBERT models.
    """
    if not text:
        return ""

    # Convert to string to be safe
    text = str(text)

    # 1. Normalize unicode characters to their base form (e.g., 'é' becomes 'e' + '´')
    # This is a crucial first step before ASCII conversion.
    text = unicodedata.normalize('NFKD', text)

    # 2. **THE DEFINITIVE FIX**: Force the text into ASCII.
    # This encodes the string into raw ASCII bytes, ignoring any character
    # that isn't a standard ASCII character (this will remove the Thai text).
    # It then decodes the bytes back into a clean Python string.
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # 3. Clean up multiple whitespaces that may have been introduced
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Create ColBERT index from ir_datasets collection")

    parser.add_argument("--dataset", required=True,
                        help="ir_datasets name (e.g., 'disks45/nocr/trec-robust-2004')")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to ColBERT checkpoint")
    parser.add_argument("--index-root", required=True,
                        help="Directory where index will be stored")
    parser.add_argument("--index-name", required=True,
                        help="Name of the index")
    parser.add_argument("--chunksize", type=float, default=6.0,
                        help="Chunk size for indexing in GB (default: 6.0)")

    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Create index
    create_colbert_index(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        index_root=args.index_root,
        index_name=args.index_name,
        chunksize=args.chunksize
    )

    print(f"✅ ColBERT index '{args.index_name}' created successfully!")
    print(f"   Location: {args.index_root}/{args.index_name}")
    print(f"   Use with: --index-root {args.index_root} --index-name {args.index_name}")


if __name__ == "__main__":
    main()
