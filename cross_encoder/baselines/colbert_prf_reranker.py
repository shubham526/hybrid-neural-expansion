#!/usr/bin/env python3
"""
ColBERT-PRF Reranking Script

Reranks test data using ColBERT-PRF with per-query indexing.
Outputs results in TREC run format.
"""

import json
import os
import argparse
import pandas as pd
import pyterrier as pt
from pathlib import Path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_jsonl(file_path):
    """Load JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_colbert_index_for_query(candidates, index_path, checkpoint_path):
    """Create ColBERT index for a single query's candidates."""
    from pyterrier_colbert.indexing import ColBERTIndexer

    # Prepare documents
    documents = []
    for candidate in candidates:
        if 'doc_text' in candidate and candidate['doc_text'].strip():
            documents.append({
                'docno': candidate['doc_id'],
                'text': candidate['doc_text']
            })

    if not documents:
        raise ValueError("No valid documents to index")

    # Create ColBERT index
    indexer = ColBERTIndexer(checkpoint_path, index_path, chunksize=3.0)
    indexer.index(documents)

    return len(documents)


def process_single_query_colbert(query_data, args, work_dir):
    """Process a single query with ColBERT PRF."""
    query_id = query_data['query_id']
    query_text = query_data['query_text']
    candidates = query_data['candidates']

    query_work_dir = work_dir / f"query_{query_id}"
    query_work_dir.mkdir(exist_ok=True)

    try:
        # Create ColBERT index for this query's candidates
        index_path = query_work_dir / "colbert_index"
        num_docs = create_colbert_index_for_query(candidates, str(index_path), args.checkpoint_path)

        # Load ColBERT factory
        from pyterrier_colbert.ranking import ColBERTFactory

        colbert = ColBERTFactory.from_dataset(
            dataset=str(index_path),
            variant="colbert",
            checkpoint=args.checkpoint_path
        )

        # Create query DataFrame
        query_df = pd.DataFrame([{
            'qid': query_id,
            'query': query_text
        }])

        # Create initial ranking DataFrame from candidates
        initial_ranking = []
        for i, candidate in enumerate(candidates):
            initial_ranking.append({
                'qid': query_id,
                'docno': candidate['doc_id'],
                'score': candidate['score'],
                'rank': i + 1
            })
        initial_ranking_df = pd.DataFrame(initial_ranking)

        # Create pipeline: initial retrieval -> ColBERT PRF -> ColBERT scoring
        initial_retriever = pt.Transformer.from_df(initial_ranking_df, uniform=True)

        # ColBERT PRF pipeline
        colbert_prf = colbert.prf(
            k=args.prf_k,  # Number of docs for PRF
            alpha=args.alpha,  # Weight for original query
            beta=args.beta  # Weight for expansion
        )

        pipeline = initial_retriever >> colbert_prf >> colbert.text_scorer()

        # Run pipeline
        results = pipeline(query_df)

        # Convert to our format
        query_results = []
        for _, row in results.iterrows():
            query_results.append({
                'query_id': str(row['qid']),
                'doc_id': str(row['docno']),
                'rank': int(row['rank']),
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
        return {
            'query_id': query_id,
            'success': False,
            'error': str(e),
            'results': []
        }

    finally:
        if query_work_dir.exists():
            shutil.rmtree(query_work_dir, ignore_errors=True)


def write_trec_run(all_results, output_file, run_name):
    """Write results in TREC run format."""
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['query_id']} Q0 {result['doc_id']} {result['rank']} {result['score']} {run_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rerank test data using ColBERT-PRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test.jsonl file from create_train_test_data.py')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to ColBERT checkpoint directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for TREC run file')

    # Optional arguments
    parser.add_argument('--run-name', type=str, default='colbert_prf',
                        help='Run name for TREC output')
    parser.add_argument('--prf-k', type=int, default=5,
                        help='Number of documents to use for PRF')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Weight for original query in PRF')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Weight for expansion terms in PRF')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Number of parallel workers (ColBERT is memory intensive)')

    args = parser.parse_args()

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    # Validate inputs
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} queries")

    # Create working directory
    work_dir = output_dir / "temp_work"
    work_dir.mkdir(exist_ok=True)

    print(f"Starting ColBERT-PRF reranking...")
    print(f"  PRF k: {args.prf_k}")
    print(f"  Alpha: {args.alpha}, Beta: {args.beta}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Run name: {args.run_name}")

    all_query_results = []
    successful_queries = 0
    failed_queries = 0

    try:
        # Process queries in parallel (fewer workers for ColBERT due to memory)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_query = {
                executor.submit(process_single_query_colbert, query_data, args, work_dir): query_data['query_id']
                for query_data in test_data
            }

            for future in as_completed(future_to_query):
                query_id = future_to_query[future]

                try:
                    result = future.result()
                    all_query_results.append(result)

                    if result['success']:
                        successful_queries += 1
                        print(f"✓ {query_id}: {result['num_results']} results")
                    else:
                        failed_queries += 1
                        print(f"✗ {query_id}: {result['error']}")

                except Exception as e:
                    failed_queries += 1
                    print(f"✗ {query_id}: {e}")

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    # Collect all results and write TREC run file
    all_results = []
    for query_result in all_query_results:
        if query_result['success']:
            all_results.extend(query_result['results'])

    if all_results:
        # Sort by query_id, then by rank
        all_results.sort(key=lambda x: (x['query_id'], x['rank']))

        # Write TREC run file
        trec_file = output_dir / f"{args.run_name}.trec"
        write_trec_run(all_results, trec_file, args.run_name)

        print(f"\n✅ ColBERT-PRF reranking completed!")
        print(f"   Successful queries: {successful_queries}")
        print(f"   Failed queries: {failed_queries}")
        print(f"   Total results: {len(all_results)}")
        print(f"   TREC run file: {trec_file}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()