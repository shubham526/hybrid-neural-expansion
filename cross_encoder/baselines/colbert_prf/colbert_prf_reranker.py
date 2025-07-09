#!/usr/bin/env python3
"""
BM25 Retrieval + ColBERT-PRF Reranking Script
Outputs TREC-formatted run file.
"""

import argparse
import pyterrier as pt
import pandas as pd
from tqdm import tqdm

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
    parser.add_argument("--fb-docs", type=int, default=3,
                        help="Number of feedback documents for PRF (default: 3)")
    parser.add_argument("--fb-embs", type=int, default=10,
                        help="Number of expansion embeddings for PRF (default: 10)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight of expansion embeddings (default: 1.0)")
    parser.add_argument("--k-clusters", type=int, default=24,
                        help="Number of clusters for PRF (default: 24)")

    args = parser.parse_args()

    print("--- SCRIPT START ---")

    print("Step 1: Initializing PyTerrier...")
    if not pt.started():
        pt.init()
    print("--- PyTerrier Initialized ---")

    from pyt_colbert import ColBERTFactory, ColbertPRF

    print("Step 2: Loading dataset topics...")
    dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
    topics = dataset.get_topics()
    topics = topics.rename(columns={"title": "query"})
    topics = preprocess_queries(topics)
    print(f"--- Loaded {len(topics)} topics ---")

    print("Step 3: Loading Terrier index...")
    index_ref = pt.IndexFactory.of(args.index_path)
    print("--- Terrier index loaded ---")

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
    colbert_factory = ColBERTFactory(
        colbert_model=args.colbert_checkpoint,
        index_root=args.colbert_index_root,
        index_name=args.colbert_index_name,
        faiss_partitions=100
    )
    print("--- ColBERTFactory Initialized ---")

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
    pipeline = (
        bm25 >>
        colbert_factory.query_encoder() >>
        colbert_factory.index_scorer(query_encoded=True, add_ranks=True) >>
        colbert_prf_transformer >>
        colbert_factory.index_scorer(query_encoded=True, add_ranks=True) % 1000
    )
    print("--- Pipeline built ---")

    print("Step 8: Starting pipeline.transform() on all topics...")
    results = pipeline.transform(topics)

    print("Step 9: Saving TREC run file...")
    pt.io.write_run(results, args.output, format="trec")

    print("✅ Retrieval and reranking completed successfully.")
    print(f"Results saved to: {args.output}")
    print(f"Total results: {len(results)}")

if __name__ == "__main__":
    main()