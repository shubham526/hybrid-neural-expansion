#!/usr/bin/env python3
"""
ANCE-PRF Reranking Script

Reranks test data using ANCE-PRF with per-query indexing.
Outputs results in TREC run format.
"""

import json
import os
import argparse
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile


def load_jsonl(file_path):
    """Load JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_candidate_index(candidates, index_path):
    """Create Lucene index with candidate documents and --storeRaw."""
    collection_dir = Path(index_path).parent / f"{Path(index_path).name}_collection"
    collection_dir.mkdir(exist_ok=True, parents=True)

    collection_file = collection_dir / "docs.jsonl"
    doc_count = 0

    with open(collection_file, 'w', encoding='utf-8') as f:
        for candidate in candidates:
            if 'doc_text' in candidate and candidate['doc_text'].strip():
                doc = {
                    'id': candidate['doc_id'],
                    'contents': candidate['doc_text']
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                doc_count += 1

    if doc_count == 0:
        raise ValueError("No valid documents to index")

    # Create index with --storeRaw (required for ANCE-PRF)
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(collection_dir),
        "--index", str(index_path),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storeRaw"  # Essential for PRF
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        shutil.rmtree(collection_dir)
        return doc_count
    except subprocess.CalledProcessError as e:
        shutil.rmtree(collection_dir, ignore_errors=True)
        raise RuntimeError(f"Indexing failed: {e.stderr}")


def process_single_query(query_data, args, work_dir):
    """Process a single query with ANCE-PRF."""
    query_id = query_data['query_id']
    query_text = query_data['query_text']
    candidates = query_data['candidates']

    query_work_dir = work_dir / f"query_{query_id}"
    query_work_dir.mkdir(exist_ok=True)

    try:
        # Create index for this query's candidates
        index_path = query_work_dir / "index"
        num_docs = create_candidate_index(candidates, str(index_path))

        # Create topics file
        topics_file = query_work_dir / "topics.tsv"
        with open(topics_file, 'w') as f:
            f.write(f"{query_id}\t{query_text}\n")

        # Run ANCE-PRF
        output_file = query_work_dir / "results.trec"

        cmd = [
            "python", "-m", "pyserini.dsearch",
            "--topics", str(topics_file),
            "--index", str(index_path),
            "--encoder", args.ance_encoder,
            "--batch-size", str(args.batch_size),
            "--output", str(output_file),
            "--prf-depth", str(args.prf_depth),
            "--prf-method", "ance-prf",
            "--threads", "1",
            "--sparse-index", args.sparse_index,
            "--ance-prf-encoder", args.ance_prf_encoder,
            "--hits", str(len(candidates))
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse results
        query_results = []
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        query_results.append({
                            'query_id': parts[0],
                            'doc_id': parts[2],
                            'rank': int(parts[3]),
                            'score': float(parts[4])
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
        description="Rerank test data using ANCE-PRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test.jsonl file from create_train_test_data.py')
    parser.add_argument('--ance-encoder', type=str, required=True,
                        help='ANCE encoder model (e.g., castorini/ance-msmarco-passage)')
    parser.add_argument('--ance-prf-encoder', type=str, required=True,
                        help='Path to ANCE-PRF checkpoint directory')
    parser.add_argument('--sparse-index', type=str, required=True,
                        help='Path to sparse Lucene index (MS MARCO with --storeRaw)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for TREC run file')

    # Optional arguments
    parser.add_argument('--run-name', type=str, default='ance_prf',
                        help='Run name for TREC output')
    parser.add_argument('--prf-depth', type=int, default=3,
                        help='PRF depth (must match checkpoint, usually 3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    if not os.path.exists(args.ance_prf_encoder):
        raise FileNotFoundError(f"ANCE-PRF encoder not found: {args.ance_prf_encoder}")
    if not os.path.exists(args.sparse_index):
        raise FileNotFoundError(f"Sparse index not found: {args.sparse_index}")

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

    print(f"Starting ANCE-PRF reranking...")
    print(f"  PRF depth: {args.prf_depth}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Run name: {args.run_name}")

    all_query_results = []
    successful_queries = 0
    failed_queries = 0

    try:
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_query = {
                executor.submit(process_single_query, query_data, args, work_dir): query_data['query_id']
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

        print(f"\n✅ ANCE-PRF reranking completed!")
        print(f"   Successful queries: {successful_queries}")
        print(f"   Failed queries: {failed_queries}")
        print(f"   Total results: {len(all_results)}")
        print(f"   TREC run file: {trec_file}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()