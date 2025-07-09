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
from tqdm import tqdm

# Move imports to top following Python conventions
try:
    print("Step 1: Initializing PyTerrier...")
    if not pt.started():
        pt.init()
    print("--- PyTerrier Initialized ---")
    import pyterrier_colbert
    from pyterrier_colbert.ranking import ColBERTFactory, ColbertPRF
except ImportError as e:
    print(f"Error importing ColBERT modules: {e}")
    print("Please ensure pyterrier_colbert is properly installed")
    sys.exit(1)


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

    print("Preprocessing queries for Terrier compatibility...")

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
                print(f"  Query {qid}: '{orig}' → '{cleaned}'")

    if changes_count > 5:
        print(f"  ... and {changes_count - 5} more queries were cleaned")
    elif changes_count > 0:
        print(f"  Total queries cleaned: {changes_count}")
    else:
        print("  No queries needed cleaning")

    return queries_df


def main():
    parser = argparse.ArgumentParser(description="BM25 + ColBERT-PRF Reranking for Robust04")

    parser.add_argument("--index-path", required=True,
                        help="Path to Terrier index directory (the folder, not data.properties)")
    parser.add_argument("--colbert-checkpoint", required=True,
                        help="Path to ColBERT checkpoint (.dnn)")
    parser.add_argument("--colbert-index-root", required=True,
                        help="Path to ColBERT index root directory")
    parser.add_argument("--colbert-index-name", required=True,
                        help="ColBERT index name (folder name under index root)")
    parser.add_argument("--output", required=True,
                        help="Path to save TREC run file (e.g., runs/bm25_colbert_prf.txt)")
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
    parser.add_argument("--k-clusters", type=int, default=24,
                        help="Number of clusters for PRF (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test pipeline on first 5 queries only")

    args = parser.parse_args()

    print("--- SCRIPT START ---")
    script_start_time = time.time()

    try:

        print("Step 2: Loading dataset topics...")
        dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
        topics = dataset.get_topics()
        topics = topics.rename(columns={"title": "query"})
        topics = preprocess_queries(topics)
        print(f"--- Loaded {len(topics)} topics ---")
        if args.dry_run:
            topics = topics.head(5)
            print(f"--- DRY RUN: Using only {len(topics)} topics ---")

        print("Step 3: Loading Terrier index...")
        try:
            index_ref = pt.IndexFactory.of(args.index_path)
            print("--- Terrier index loaded ---")
        except Exception as e:
            print(f"Error loading Terrier index: {e}")
            print(f"Please check that {args.index_path} contains a valid Terrier index")
            return

        print("Step 4: Building BM25 retriever...")
        bm25 = pt.BatchRetrieve(
            index_ref,
            wmodel="BM25",
            metadata=["docno", "text"],
            controls={"qemodel": "QE"},
            verbose=True
        ) % args.top_k
        print("--- BM25 retriever built ---")

        print("Step 5: Initializing ColBERTFactory (this is the most likely step to hang)...")
        try:
            colbert_factory = ColBERTFactory(
                colbert_model=args.colbert_checkpoint,
                index_root=args.colbert_index_root,
                index_name=args.colbert_index_name,
                faiss_partitions=100
            )
            print("--- ColBERTFactory Initialized ---")
        except Exception as e:
            print(f"Error initializing ColBERT: {e}")
            print("Please check:")
            print(f"  - ColBERT checkpoint exists: {args.colbert_checkpoint}")
            print(f"  - ColBERT index exists: {args.colbert_index_root}/{args.colbert_index_name}")
            return

        print("Step 6: Building ColBERT PRF reranker...")
        colbert_prf_transformer = ColbertPRF(
            colbert_factory,
            k=args.k_clusters,
            fb_docs=args.fb_docs,
            fb_embs=args.fb_embs,
            beta=args.beta,
            return_docs=True
        )
        print("--- ColBERT PRF transformer built ---")

        print("Step 7: Building the final pipeline...")
        # Enable verbose mode on ColBERT factory for internal progress tracking
        colbert_factory.verbose = True

        pipeline = (
                bm25 >>
                colbert_factory.query_encoder() >>
                colbert_factory.index_scorer(query_encoded=True, add_ranks=True) >>
                colbert_prf_transformer >>
                colbert_factory.index_scorer(query_encoded=True, add_ranks=True) % args.final_k
        )
        print("--- Pipeline built ---")

        print("Step 8: Starting pipeline.transform() on all topics...")
        print("Note: Progress will be shown by individual pipeline components...")

        try:
            pipeline_start_time = time.time()
            results = pipeline.transform(topics)
            pipeline_time = time.time() - pipeline_start_time
            print(f"--- Pipeline completed in {pipeline_time:.2f} seconds ---")

            # Result validation and statistics
            print("Step 8.1: Validating results...")
            if len(results) == 0:
                print("⚠️ Warning: No results returned from pipeline!")
                return

            results_per_query = results.groupby('qid').size()
            print(f"Results per query statistics:")
            print(f"  Mean: {results_per_query.mean():.1f}")
            print(f"  Min: {results_per_query.min()}")
            print(f"  Max: {results_per_query.max()}")
            print(f"  Total results: {len(results)}")
            print(f"  Unique queries: {results['qid'].nunique()}")

        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            return

        # Memory cleanup for large datasets
        print("Step 8.2: Cleaning up memory...")
        gc.collect()

        print("Step 9: Saving TREC run file...")
        try:
            pt.io.write_results(results, args.output, format="trec")
            print("✅ Retrieval and reranking completed successfully.")
            print(f"Results saved to: {args.output}")

        except Exception as e:
            print(f"Error saving results: {e}")
            return

    except KeyboardInterrupt:
        print("\n❌ Script interrupted by user")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return

    finally:
        total_time = time.time() - script_start_time
        print(f"--- Total runtime: {total_time:.2f} seconds ---")
        print("--- SCRIPT END ---")


if __name__ == "__main__":
    main()