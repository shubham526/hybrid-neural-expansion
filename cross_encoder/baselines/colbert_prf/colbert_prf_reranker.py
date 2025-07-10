#!/usr/bin/env python3
"""
BM25 Retrieval + ColBERT-PRF Reranking Script
Outputs TREC-formatted run file.
"""

import argparse
import pyterrier as pt
import pandas as pd
import time
import gc
import sys
import unicodedata
from pathlib import Path
import re
import logging

# Add the project's root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info
logger = logging.getLogger(__name__)

# Move imports to top following Python conventions
try:
    logger.info("Step 1: Initializing PyTerrier...")
    if not pt.started():
        pt.init()
    logger.info("--- PyTerrier Initialized ---")
    import pyterrier_colbert
    from pyterrier_colbert.ranking import ColBERTFactory, ColbertPRF
except ImportError as e:
    logger.error(f"Error importing ColBERT modules: {e}")
    logger.error("Please ensure pyterrier_colbert is properly installed")
    sys.exit(1)


def get_query_column_name(topics_df):
    """
    Intelligently detect the appropriate query column name from the topics DataFrame.

    Args:
        topics_df: DataFrame containing topic/query data

    Returns:
        str: The column name to use for queries

    Raises:
        ValueError: If no suitable query column is found
    """
    # Priority order for query fields
    query_field_priority = [
        'query',  # Already standardized
        'text',  # MS MARCO, some BEIR datasets
        'title',  # TREC Robust, many TREC collections
        'question',  # Natural Questions, some QA datasets
        'body',  # Some datasets use 'body'
        'description'  # Some datasets have description as main query
    ]

    available_columns = set(topics_df.columns)
    logger.info(f"Available topic columns: {sorted(available_columns)}")

    # Check each field in priority order
    for field in query_field_priority:
        if field in available_columns:
            logger.info(f"Using '{field}' as query field")
            return field

    # If no standard field found, look for any column containing 'query' or 'text'
    for col in available_columns:
        if 'query' in col.lower() or 'text' in col.lower():
            logger.info(f"Using '{col}' as query field (contains 'query' or 'text')")
            return col

    # Last resort: use the first string column that's not 'qid' or 'docno'
    for col in topics_df.columns:
        if col not in ['qid', 'docno', 'query_id', 'topic_id'] and topics_df[col].dtype == 'object':
            logger.info(f"Using '{col}' as query field (first string column)")
            return col

    raise ValueError(f"Could not identify query column in: {list(available_columns)}")


def get_query_text(query_obj):
    """
    Extract query text from ir_datasets query object.

    Args:
        query_obj: Query object from ir_datasets

    Returns:
        str: The query text
    """
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        # For TREC topics, optionally combine title and description
        if hasattr(query_obj, 'description') and query_obj.description and query_obj.description.strip():
            return f"{query_obj.title} {query_obj.description}"
        return query_obj.title
    elif hasattr(query_obj, 'question'):
        return query_obj.question
    elif hasattr(query_obj, 'body'):
        return query_obj.body
    elif hasattr(query_obj, 'description'):
        return query_obj.description
    else:
        # Convert to string as fallback
        return str(query_obj)


def load_and_standardize_topics(dataset_name, use_title_desc=False):
    """
    Load topics from ir_datasets and standardize to have 'query' column.

    Args:
        dataset_name: Name of the dataset (e.g., 'disks45/nocr/trec-robust-2004')
        use_title_desc: Whether to combine title and description for TREC topics

    Returns:
        pd.DataFrame: DataFrame with 'qid' and 'query' columns
    """
    logger.info(f"Loading dataset: {dataset_name}")

    try:
        dataset = pt.get_dataset(f"irds:{dataset_name}")
        topics = dataset.get_topics()

        # Check if we already have a 'query' column
        if 'query' in topics.columns:
            logger.info("Dataset already has 'query' column")
            return topics

        # Detect the appropriate query column
        query_column = get_query_column_name(topics)

        # Handle special case for TREC topics with title/description combination
        if use_title_desc and 'title' in topics.columns and 'description' in topics.columns:
            logger.info("Combining title and description for TREC topics")
            topics['query'] = topics.apply(lambda row:
                                           f"{row['title']} {row['description']}" if pd.notna(row['description']) and
                                                                                     row['description'].strip()
                                           else row['title'], axis=1)
        else:
            # Standard rename
            topics = topics.rename(columns={query_column: "query"})

        # Validate the result
        if 'query' not in topics.columns:
            raise ValueError("Failed to create 'query' column")

        if 'qid' not in topics.columns:
            raise ValueError("Dataset must have 'qid' column")

        logger.info(f"Successfully loaded {len(topics)} topics")
        logger.info(f"Sample query: {topics['query'].iloc[0][:100]}...")

        return topics[['qid', 'query']]  # Return only essential columns

    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_name}': {e}")
        raise


def clean_text(text):
    """
    Clean text by removing problematic characters and forcing it into ASCII,
    which is safe for most standard ColBERT models.
    """
    if not text or pd.isna(text):
        return ""

    # Convert to string to be safe
    text = str(text)

    # 1. Normalize unicode characters to their base form (e.g., 'é' becomes 'e' + '´')
    # This is a crucial first step before ASCII conversion.
    text = unicodedata.normalize('NFKD', text)

    # 2. **THE DEFINITIVE FIX**: Force the text into ASCII.
    # This encodes the string into raw ASCII bytes, ignoring any character
    # that isn't a standard ASCII character (this will remove problematic text).
    # It then decodes the bytes back into a clean Python string.
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # 3. Clean up multiple whitespaces that may have been introduced
    text = re.sub(r'\s+', ' ', text)

    # 4. Remove any remaining control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

    # 5. Truncate very long text to avoid memory issues
    max_length = 10000  # Reasonable limit for document text
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def preprocess_documents(df):
    """
    Preprocess document text to handle problematic characters.
    """
    if 'text' not in df.columns:
        return df

    logger.info("Preprocessing document text...")
    original_count = len(df)

    # Clean the text column
    df = df.copy()
    df['text'] = df['text'].apply(clean_text)

    # Remove documents with empty text after cleaning
    df = df[df['text'].str.len() > 0]
    removed_count = original_count - len(df)

    if removed_count > 0:
        logger.warning(f"  Removed {removed_count} documents with empty/problematic text")

    logger.info(f"  Processed {len(df)} documents")
    return df


def safe_index_scorer(colbert_factory, query_encoded=True, add_ranks=True, batch_size=100):
    """
    Wrapper around index_scorer that handles problematic documents gracefully.
    """

    def safe_scorer(df):
        try:
            # Try the normal index_scorer first
            return colbert_factory.index_scorer(query_encoded=query_encoded, add_ranks=add_ranks,
                                                batch_size=batch_size).transform(df)
        except Exception as e:
            logger.error(f"    ⚠️ Error in batch scoring: {e}")
            logger.error(f"    Trying smaller batch sizes...")

            # Try progressively smaller batch sizes
            for smaller_batch in [50, 20, 10, 5, 1]:
                try:
                    result = colbert_factory.index_scorer(query_encoded=query_encoded, add_ranks=add_ranks,
                                                          batch_size=smaller_batch).transform(df)
                    logger.info(f"    ✅ Succeeded with batch_size={smaller_batch}")
                    return result
                except Exception as e2:
                    logger.error(f"    ❌ batch_size={smaller_batch} failed: {e2}")
                    continue

            # If all batch sizes fail, try processing documents one by one and skip problematic ones
            logger.info(f"    Trying document-by-document processing...")
            successful_docs = []

            for idx, (_, row) in enumerate(df.iterrows()):
                single_doc = pd.DataFrame([row])
                try:
                    single_result = colbert_factory.index_scorer(query_encoded=query_encoded, add_ranks=False,
                                                                 batch_size=1).transform(single_doc)
                    successful_docs.append(single_result)
                except Exception as e3:
                    logger.error(
                        f"    ❌ Skipping problematic document {idx + 1}/{len(df)}: docno={row.get('docno', 'unknown')}")
                    continue

            if successful_docs:
                result = pd.concat(successful_docs, ignore_index=True)
                if add_ranks:
                    result = pt.model.add_ranks(result)
                logger.info(f"    ✅ Processed {len(successful_docs)}/{len(df)} documents successfully")
                return result
            else:
                logger.warning(f"    ❌ No documents could be processed for this query")
                return pd.DataFrame(columns=df.columns.tolist() + ['score', 'rank'])

    return pt.apply.by_query(safe_scorer)


def sort_trec_run(input_file, output_file):
    """
    Post-process TREC run file to ensure proper ranking by score.
    Much faster than DataFrame operations.
    """
    from collections import defaultdict

    # Read all lines
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, q0, docno, rank, score, tag = parts[:6]
                lines.append((qid, q0, docno, float(score), tag))

    # Group by qid and sort by score
    by_qid = defaultdict(list)
    for qid, q0, docno, score, tag in lines:
        by_qid[qid].append((q0, docno, score, tag))

    # Write sorted results
    with open(output_file, 'w') as f:
        for qid in sorted(by_qid.keys(), key=int):
            docs = sorted(by_qid[qid], key=lambda x: x[2], reverse=True)
            for rank, (q0, docno, score, tag) in enumerate(docs):
                f.write(f"{qid} {q0} {docno} {rank} {score} {tag}\n")


def preprocess_queries(queries_df):
    """
    Preprocess queries to handle special characters that cause parsing errors.

    Args:
        queries_df: DataFrame with 'qid' and 'query' columns

    Returns:
        DataFrame with cleaned queries
    """

    def clean_query(query_text):
        """Clean individual query text"""
        if pd.isna(query_text):
            return ""

        # Convert to string
        query = str(query_text)

        # More aggressive cleaning for Terrier compatibility
        import re

        # Remove all punctuation except spaces, hyphens, and periods in numbers
        # Keep only: letters, numbers, spaces, hyphens
        query = re.sub(r'[^\w\s\-]', ' ', query)

        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)

        # Remove leading/trailing whitespace
        query = query.strip()

        # Ensure query is not empty
        if not query:
            return "empty query"

        return query

    logger.info("Preprocessing queries for Terrier compatibility...")

    # Store original queries for logging
    original_queries = queries_df['query'].tolist()

    # Clean queries
    queries_df = queries_df.copy()
    queries_df['query'] = queries_df['query'].apply(clean_query)

    # Log changes
    changes_count = 0
    for i, (orig, cleaned) in enumerate(zip(original_queries, queries_df['query'].tolist())):
        if orig != cleaned:
            changes_count += 1
            if changes_count <= 5:  # Show first 5 changes
                qid = queries_df.iloc[i]['qid']
                logger.info(f"  Query {qid}: '{orig}' → '{cleaned}'")

    if changes_count > 5:
        logger.info(f"  ... and {changes_count - 5} more queries were cleaned")
    elif changes_count > 0:
        logger.info(f"  Total queries cleaned: {changes_count}")
    else:
        logger.info("  No queries needed cleaning")

    return queries_df


def main():
    parser = argparse.ArgumentParser(description="BM25 + ColBERT-PRF Reranking for Robust04")

    parser.add_argument("--index-path", required=True,
                        help="Path to Terrier index directory (the folder, not data.properties)")
    parser.add_argument("--dataset", required=True,
                     help="ir_datasets name (e.g., 'disks45/nocr/trec-robust-2004')")
    parser.add_argument("--use-title-desc", action="store_true", default=False,
                        help="For TREC topics, combine title and description fields (default: False)")
    parser.add_argument("--colbert-checkpoint", required=True,
                        help="Path to ColBERT checkpoint (.dnn)")
    parser.add_argument("--colbert-index-root", required=True,
                        help="Path to ColBERT index root directory")
    parser.add_argument("--colbert-index-name", required=True,
                        help="ColBERT index name (folder name under index root)")
    parser.add_argument("--output", required=True,
                        help="Path to save TREC run file (e.g., runs/bm25_colbert_prf.txt)")
    parser.add_argument("--log-file", default="colbert_prf_reranker.log",
                        help="Path to save log file")
    parser.add_argument("--top-k", type=int, default=1000,
                        help="Number of BM25 documents to retrieve (default: 1000)")
    parser.add_argument("--final-k", type=int, default=1000,
                        help="Final number of documents to output (default: 1000)")
    parser.add_argument("--fb-docs", type=int, default=3,
                        help="Number of feedback documents for PRF (default: 3)")
    parser.add_argument("--fb-embs", type=int, default=10,
                        help="Number of expansion embeddings for PRF (default: 10)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight of expansion embeddings (default: 1.0)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU for ColBERT operations (default: True)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Force CPU-only operation")
    parser.add_argument("--doc-batch-size", type=int, default=100,
                        help="Batch size for ColBERT scoring (documents) (default: 100)")
    parser.add_argument("--query-batch-size", type=int, default=10,
                        help="Batch size for ColBERT scoring (queries) (default: 10)")

    parser.add_argument("--k-clusters", type=int, default=24,
                        help="Number of clusters for PRF (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test pipeline on first 5 queries only")
    parser.add_argument('--log-level', type=str, default='INFO')


    args = parser.parse_args()

    logger = setup_experiment_logging("colbert_prf_reranker", args.log_level,
                                      str(args.log_file))
    log_experiment_info(logger, **vars(args))
    
    logger.info(f"Processing {args.doc_batch_size} documents per batch for ColBERT scoring...")
    logger.info(f"Processing {args.query_batch_size} queries per batch for ColBERT scoring...")

    logger.info("--- SCRIPT START ---")
    script_start_time = time.time()

    try:
        logger.info("Step 2: Loading dataset topics...")
        topics = load_and_standardize_topics(args.dataset, use_title_desc=args.use_title_desc)
        topics = preprocess_queries(topics)
        logger.info(f"--- Loaded {len(topics)} topics ---")
        if args.dry_run:
            topics = topics.head(5)
            logger.info(f"--- DRY RUN: Using only {len(topics)} topics ---")

        logger.info("Step 3: Loading Terrier index...")
        try:
            index_ref = pt.IndexFactory.of(args.index_path)
            logger.info("--- Terrier index loaded ---")
        except Exception as e:
            logger.error(f"Error loading Terrier index: {e}")
            logger.error(f"Please check that {args.index_path} contains a valid Terrier index")
            return

        logger.info("Step 4: Building BM25 retriever...")
        bm25 = pt.BatchRetrieve(
            index_ref,
            wmodel="BM25",
            metadata=["docno", "text"],
            controls={"qemodel": "QE"},
            verbose=True
        ) % args.top_k
        logger.info("--- BM25 retriever built ---")

        logger.info("Step 4.1: Adding document text preprocessing...")
        # Add document preprocessing to clean problematic text
        bm25_with_preprocessing = bm25 >> pt.apply.generic(preprocess_documents)

        logger.info("Step 5: Initializing ColBERTFactory (this is the most likely step to hang)...")
        try:
            colbert_factory = ColBERTFactory(
                colbert_model=args.colbert_checkpoint,
                index_root=args.colbert_index_root,
                index_name=args.colbert_index_name,
                faiss_partitions=100,
                gpu=args.gpu  # Use GPU setting from args
            )
            logger.info("--- ColBERTFactory Initialized ---")
        except Exception as e:
            logger.error(f"Error initializing ColBERT: {e}")
            logger.error("Please check:")
            logger.error(f"  - ColBERT checkpoint exists: {args.colbert_checkpoint}")
            logger.error(f"  - ColBERT index exists: {args.colbert_index_root}/{args.colbert_index_name}")
            return

        logger.info("Step 6: Building ColBERT PRF reranker...")
        colbert_prf_transformer = ColbertPRF(
            colbert_factory,
            k=args.k_clusters,
            fb_docs=args.fb_docs,
            fb_embs=args.fb_embs,
            beta=args.beta,
            return_docs=True
        )
        logger.info("--- ColBERT PRF transformer built ---")

        logger.info("Step 7: Building the final pipeline...")
        # Enable verbose mode on ColBERT factory for internal progress tracking
        colbert_factory.verbose = True

        # pipeline = (
        #         bm25_with_preprocessing >>  # Use the version with preprocessing
        #         colbert_factory.query_encoder() >>
        #         colbert_factory.index_scorer(query_encoded=True, add_ranks=True, batch_size=args.batch_size) >>
        #         colbert_prf_transformer >>
        #         colbert_factory.index_scorer(query_encoded=True, add_ranks=True,
        #                                      batch_size=args.batch_size) % args.final_k
        # )
        pipeline = (
                bm25_with_preprocessing >>
                colbert_factory.query_encoder() >>
                safe_index_scorer(colbert_factory, query_encoded=True, add_ranks=True, batch_size=args.doc_batch_size) >>
                colbert_prf_transformer >>
                safe_index_scorer(colbert_factory, query_encoded=True, add_ranks=True,
                                  batch_size=args.doc_batch_size) % args.final_k
        )
        logger.info("--- Pipeline built ---")

        logger.info("Step 8: Starting pipeline.transform() on all topics...")
        logger.info("Progress will be shown by individual pipeline components...")

        try:
            pipeline_start_time = time.time()

            # Process queries in smaller batches to isolate problematic ones
            all_results = []

            for i in range(0, len(topics), args.query_batch_size):
                batch_topics = topics.iloc[i:i + args.query_batch_size]
                batch_qids = batch_topics['qid'].tolist()
                logger.info(f"Processing batch {i // args.query_batch_size + 1}: queries {batch_qids}")

                try:
                    batch_results = pipeline.transform(batch_topics)
                    all_results.append(batch_results)
                    logger.info(f"  ✅ Batch completed successfully")
                except Exception as e:
                    logger.error(f"  ❌ Error in batch {batch_qids}: {e}")

                    # Try individual queries in this batch
                    for _, topic_row in batch_topics.iterrows():
                        single_topic = pd.DataFrame([topic_row])
                        qid = topic_row['qid']
                        try:
                            single_result = pipeline.transform(single_topic)
                            all_results.append(single_result)
                            logger.info(f"    ✅ Query {qid} processed individually")
                        except Exception as single_e:
                            logger.error(f"    ❌ Query {qid} failed: {single_e}")
                            logger.error(f"    Query text: '{topic_row['query']}'")
                            # Continue with other queries
                            continue

            if not all_results:
                logger.warning("❌ No queries processed successfully!")
                return

            results = pd.concat(all_results, ignore_index=True)
            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"--- Pipeline completed in {pipeline_time:.2f} seconds ---")

            # Result validation and statistics
            logger.info("Step 8.1: Validating results...")
            if len(results) == 0:
                logger.warning("⚠️ Warning: No results returned from pipeline!")
                return

            results_per_query = results.groupby('qid').size()
            logger.info(f"Results per query statistics:")
            logger.info(f"  Mean: {results_per_query.mean():.1f}")
            logger.info(f"  Min: {results_per_query.min()}")
            logger.info(f"  Max: {results_per_query.max()}")
            logger.info(f"  Total results: {len(results)}")
            logger.info(f"  Unique queries: {results['qid'].nunique()}")
            logger.info(f"  Successfully processed queries: {sorted(results['qid'].unique())}")

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            return

        # Memory cleanup for large datasets
        logger.info("Step 8.2: Cleaning up memory...")
        gc.collect()

        logger.info("Step 9: Saving TREC run file...")
        try:
            # Save raw results first
            temp_output = args.output + ".temp"
            pt.io.write_results(results, temp_output, format="trec")

            # Post-process to ensure proper ranking
            logger.info("Step 9.1: Post-processing rankings...")
            sort_trec_run(temp_output, args.output)

            # Clean up temp file
            import os
            os.remove(temp_output)

            logger.info("✅ Retrieval and reranking completed successfully.")
            logger.info(f"Results saved to: {args.output}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return

    except KeyboardInterrupt:
        logger.error("\n❌ Script interrupted by user")
        return
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.logger.info_exc()
        return

    finally:
        total_time = time.time() - script_start_time
        logger.info(f"--- Total runtime: {total_time:.2f} seconds ---")
        logger.info("--- SCRIPT END ---")


if __name__ == "__main__":
    main()