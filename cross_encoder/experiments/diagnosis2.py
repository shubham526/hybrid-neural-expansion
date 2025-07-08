#!/usr/bin/env python3
"""
Comprehensive Query Filtering Debug Script

This script analyzes why queries are being filtered out during train/test data creation.
It checks all possible filtering steps and provides detailed statistics.
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Any

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import ir_datasets
    from cross_encoder.src.utils.file_utils import load_features_file, load_trec_run
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    print("Some functionality may be limited")


def load_query_subset(subset_file: str) -> Set[str]:
    """Load query IDs from subset file."""
    print(f"Loading query subset from: {subset_file}")

    try:
        # Try pandas first
        df = pd.read_csv(subset_file, sep='\t', dtype=str)

        if 'query_id' in df.columns:
            query_ids = set(df['query_id'].astype(str))
        else:
            # Assume first column is query_id
            query_ids = set(df.iloc[:, 0].astype(str))

    except Exception as e:
        print(f"Pandas failed ({e}), trying manual parsing...")

        query_ids = set()
        with open(subset_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    query_ids.add(parts[0])

    print(f"Loaded {len(query_ids)} query IDs from subset file")
    return query_ids


def load_features_safe(features_file: str) -> Dict[str, Any]:
    """Safely load features file."""
    print(f"Loading features from: {features_file}")

    try:
        features = load_features_file(features_file)
        print(f"Loaded features for {len(features)} queries")
        return features
    except Exception as e:
        print(f"Error loading features: {e}")
        return {}


def load_run_safe(run_file: str) -> Dict[str, List]:
    """Safely load run file."""
    print(f"Loading run file from: {run_file}")

    try:
        runs = load_trec_run(run_file)
        print(f"Loaded runs for {len(runs)} queries")
        return runs
    except Exception as e:
        print(f"Error loading run file: {e}")
        return {}


def analyze_features_quality(features: Dict[str, Any], query_subset: Set[str]) -> Dict[str, Any]:
    """Analyze the quality and completeness of features."""
    print("\n=== FEATURES ANALYSIS ===")

    results = {
        'total_features': len(features),
        'empty_features': 0,
        'invalid_features': 0,
        'missing_term_features': 0,
        'empty_term_features': 0,
        'sample_feature_keys': [],
        'sample_empty_queries': [],
        'feature_sizes': []
    }

    for qid, feature_data in features.items():
        # Check if query is in our subset
        if qid not in query_subset:
            continue

        # Check for completely empty features
        if not feature_data:
            results['empty_features'] += 1
            if len(results['sample_empty_queries']) < 5:
                results['sample_empty_queries'].append(qid)
            continue

        # Check for missing term_features key
        if 'term_features' not in feature_data:
            results['missing_term_features'] += 1
            continue

        # Check for empty term_features
        term_features = feature_data['term_features']
        if not term_features:
            results['empty_term_features'] += 1
            continue

        # Record feature size
        results['feature_sizes'].append(len(term_features))

        # Sample feature keys
        if len(results['sample_feature_keys']) < 10:
            results['sample_feature_keys'].extend(list(term_features.keys())[:3])

    print(f"Total features: {results['total_features']}")
    print(f"Empty features: {results['empty_features']}")
    print(f"Missing term_features: {results['missing_term_features']}")
    print(f"Empty term_features: {results['empty_term_features']}")

    if results['feature_sizes']:
        avg_size = sum(results['feature_sizes']) / len(results['feature_sizes'])
        print(f"Average feature size: {avg_size:.1f}")
        print(f"Min/Max feature sizes: {min(results['feature_sizes'])}/{max(results['feature_sizes'])}")

    if results['sample_empty_queries']:
        print(f"Sample empty queries: {results['sample_empty_queries'][:5]}")

    return results


def analyze_runs_quality(runs: Dict[str, List], query_subset: Set[str]) -> Dict[str, Any]:
    """Analyze the quality and completeness of run data."""
    print("\n=== RUNS ANALYSIS ===")

    results = {
        'total_runs': len(runs),
        'empty_runs': 0,
        'run_sizes': [],
        'sample_empty_queries': []
    }

    for qid, candidates in runs.items():
        # Check if query is in our subset
        if qid not in query_subset:
            continue

        if not candidates:
            results['empty_runs'] += 1
            if len(results['sample_empty_queries']) < 5:
                results['sample_empty_queries'].append(qid)
        else:
            results['run_sizes'].append(len(candidates))

    print(f"Total runs: {results['total_runs']}")
    print(f"Empty runs: {results['empty_runs']}")

    if results['run_sizes']:
        avg_size = sum(results['run_sizes']) / len(results['run_sizes'])
        print(f"Average candidates per query: {avg_size:.1f}")
        print(f"Min/Max candidates: {min(results['run_sizes'])}/{max(results['run_sizes'])}")

    if results['sample_empty_queries']:
        print(f"Sample empty run queries: {results['sample_empty_queries'][:5]}")

    return results


def analyze_dataset_qrels(dataset_name: str, query_subset: Set[str]) -> Dict[str, Any]:
    """Analyze qrels from dataset."""
    print(f"\n=== QRELS ANALYSIS ({dataset_name}) ===")

    try:
        dataset = ir_datasets.load(dataset_name)

        qrels = defaultdict(dict)
        positive_counts = Counter()

        for qrel in dataset.qrels_iter():
            if qrel.query_id in query_subset:
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
                if qrel.relevance > 0:
                    positive_counts[qrel.query_id] += 1

        results = {
            'total_qrels_queries': len(qrels),
            'queries_with_positives': len(positive_counts),
            'queries_without_positives': len(qrels) - len(positive_counts),
            'avg_positives_per_query': sum(positive_counts.values()) / len(positive_counts) if positive_counts else 0,
            'sample_no_positive_queries': []
        }

        # Find queries without positive examples
        no_positive = [qid for qid in qrels.keys() if qid not in positive_counts]
        results['sample_no_positive_queries'] = no_positive[:10]

        print(f"Queries with qrels: {results['total_qrels_queries']}")
        print(f"Queries with positive examples: {results['queries_with_positives']}")
        print(f"Queries without positive examples: {results['queries_without_positives']}")
        print(f"Average positives per query: {results['avg_positives_per_query']:.2f}")

        if results['sample_no_positive_queries']:
            print(f"Sample queries without positives: {results['sample_no_positive_queries'][:5]}")

        return results

    except Exception as e:
        print(f"Error analyzing dataset qrels: {e}")
        return {}


def analyze_document_coverage(dataset_name: str, runs: Dict[str, List], query_subset: Set[str],
                              sample_size: int = 100) -> Dict[str, Any]:
    """Analyze document text coverage for a sample of queries."""
    print(f"\n=== DOCUMENT COVERAGE ANALYSIS ({dataset_name}) ===")

    try:
        dataset = ir_datasets.load(dataset_name)

        # Load a sample of documents for analysis
        print("Loading document sample...")
        documents = {}
        doc_count = 0

        for doc in dataset.docs_iter():
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                title = getattr(doc, 'title', '')
                body = doc.body
                doc_text = f"{title} {body}".strip() if title else body
            else:
                doc_text = str(doc)

            documents[doc.doc_id] = doc_text
            doc_count += 1

            # Limit for analysis
            if doc_count >= 100000:  # Load first 100K docs for analysis
                break

        print(f"Loaded {len(documents)} documents for analysis")

        # Analyze coverage for sample queries
        sample_queries = list(query_subset)[:sample_size]

        total_candidates = 0
        candidates_with_text = 0
        queries_without_any_docs = 0

        for qid in sample_queries:
            if qid not in runs:
                continue

            query_candidates = runs[qid]
            query_has_docs = False

            for doc_id, score in query_candidates:
                total_candidates += 1
                if doc_id in documents:
                    candidates_with_text += 1
                    query_has_docs = True

            if not query_has_docs:
                queries_without_any_docs += 1

        coverage = candidates_with_text / total_candidates if total_candidates > 0 else 0

        results = {
            'sample_queries_analyzed': len(sample_queries),
            'total_candidates_sampled': total_candidates,
            'candidates_with_text': candidates_with_text,
            'document_coverage': coverage,
            'queries_without_any_docs': queries_without_any_docs
        }

        print(f"Sample queries analyzed: {results['sample_queries_analyzed']}")
        print(f"Total candidates: {results['total_candidates_sampled']}")
        print(f"Candidates with document text: {results['candidates_with_text']}")
        print(f"Document coverage: {coverage:.2%}")
        print(f"Queries without any document text: {results['queries_without_any_docs']}")

        return results

    except Exception as e:
        print(f"Error analyzing document coverage: {e}")
        return {}


def cross_reference_analysis(query_subset: Set[str], features: Dict[str, Any], runs: Dict[str, List]) -> Dict[str, Any]:
    """Cross-reference all data sources to find overlaps and gaps."""
    print("\n=== CROSS-REFERENCE ANALYSIS ===")

    features_qids = set(features.keys())
    runs_qids = set(runs.keys())

    # Find overlaps
    subset_features = query_subset & features_qids
    subset_runs = query_subset & runs_qids
    subset_both = query_subset & features_qids & runs_qids

    # Find gaps
    subset_missing_features = query_subset - features_qids
    subset_missing_runs = query_subset - runs_qids
    subset_missing_both = query_subset - (features_qids & runs_qids)

    results = {
        'query_subset_size': len(query_subset),
        'features_size': len(features_qids),
        'runs_size': len(runs_qids),
        'subset_with_features': len(subset_features),
        'subset_with_runs': len(subset_runs),
        'subset_with_both': len(subset_both),
        'missing_features': len(subset_missing_features),
        'missing_runs': len(subset_missing_runs),
        'missing_both': len(subset_missing_both),
        'sample_missing_features': list(subset_missing_features)[:10],
        'sample_missing_runs': list(subset_missing_runs)[:10],
        'expected_final_count': len(subset_both)
    }

    print(f"Query subset size: {results['query_subset_size']:,}")
    print(f"Features file size: {results['features_size']:,}")
    print(f"Runs file size: {results['runs_size']:,}")
    print()
    print(f"Subset queries with features: {results['subset_with_features']:,}")
    print(f"Subset queries with runs: {results['subset_with_runs']:,}")
    print(f"Subset queries with BOTH: {results['subset_with_both']:,}")
    print()
    print(f"Missing features: {results['missing_features']:,}")
    print(f"Missing runs: {results['missing_runs']:,}")
    print(f"Missing both: {results['missing_both']:,}")

    if results['sample_missing_features']:
        print(f"Sample missing features: {results['sample_missing_features'][:5]}")

    if results['sample_missing_runs']:
        print(f"Sample missing runs: {results['sample_missing_runs'][:5]}")

    print(f"\n*** EXPECTED FINAL COUNT: {results['expected_final_count']:,} ***")

    return results


def detailed_query_analysis(query_subset: Set[str], features: Dict[str, Any], runs: Dict[str, List],
                            sample_size: int = 20):
    """Analyze a sample of specific queries in detail."""
    print(f"\n=== DETAILED QUERY ANALYSIS (Sample of {sample_size}) ===")

    # Get sample queries that should be included
    valid_queries = query_subset & set(features.keys()) & set(runs.keys())
    sample_queries = list(valid_queries)[:sample_size]

    print(f"Analyzing {len(sample_queries)} sample queries...")

    for i, qid in enumerate(sample_queries):
        print(f"\n--- Query {i + 1}: {qid} ---")

        # Check features
        feature_data = features.get(qid, {})
        term_features = feature_data.get('term_features', {})
        print(f"  Features: {len(term_features)} terms")
        if term_features:
            sample_terms = list(term_features.keys())[:3]
            print(f"  Sample terms: {sample_terms}")

        # Check runs
        candidates = runs.get(qid, [])
        print(f"  Candidates: {len(candidates)}")
        if candidates:
            print(f"  Top candidate: {candidates[0][0]} (score: {candidates[0][1]:.4f})")

        # This query should be included unless other filters apply
        print(f"  Status: Should be included (has both features and runs)")


def main():
    parser = argparse.ArgumentParser(description="Debug query filtering in train/test data creation")

    parser.add_argument('--query-subset-file', required=True,
                        help='Path to query subset TSV file')
    parser.add_argument('--features-file', required=True,
                        help='Path to features file')
    parser.add_argument('--run-file', required=True,
                        help='Path to run file')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g., msmarco-passage/train/judged)')

    parser.add_argument('--output-file',
                        help='Save detailed report to JSON file')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Sample size for document coverage analysis')
    parser.add_argument('--detailed-sample', type=int, default=20,
                        help='Number of queries for detailed analysis')

    args = parser.parse_args()

    print("=" * 60)
    print("COMPREHENSIVE QUERY FILTERING DEBUG")
    print("=" * 60)

    # Load all data sources
    print("Loading data sources...")
    query_subset = load_query_subset(args.query_subset_file)
    features = load_features_safe(args.features_file)
    runs = load_run_safe(args.run_file)

    if not query_subset or not features or not runs:
        print("ERROR: Failed to load one or more data sources!")
        sys.exit(1)

    # Run all analyses
    results = {}

    results['features_analysis'] = analyze_features_quality(features, query_subset)
    results['runs_analysis'] = analyze_runs_quality(runs, query_subset)
    results['qrels_analysis'] = analyze_dataset_qrels(args.dataset, query_subset)
    results['document_coverage'] = analyze_document_coverage(args.dataset, runs, query_subset, args.sample_size)
    results['cross_reference'] = cross_reference_analysis(query_subset, features, runs)

    # Detailed query analysis
    detailed_query_analysis(query_subset, features, runs, args.detailed_sample)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    expected_count = results['cross_reference']['expected_final_count']
    print(f"Expected queries in train.jsonl: {expected_count:,}")
    print(f"Actual queries in your train.jsonl: 43,459")
    print(f"Difference: {expected_count - 43459:,}")

    if expected_count > 43459:
        print("\nPossible reasons for remaining difference:")
        print("1. --ensure-positive-training flag (filters queries without positive examples)")
        print("2. Document text coverage (queries where ALL candidates lack document text)")
        print("3. Other filtering logic in the data creation script")

        if results['qrels_analysis'].get('queries_without_positives', 0) > 0:
            print(f"   - {results['qrels_analysis']['queries_without_positives']} queries have no positive examples")

        if results['document_coverage'].get('queries_without_any_docs', 0) > 0:
            estimated_no_docs = (results['document_coverage']['queries_without_any_docs'] /
                                 results['document_coverage']['sample_queries_analyzed'] * expected_count)
            print(f"   - Estimated ~{estimated_no_docs:.0f} queries have no document text")

    # Save detailed report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {args.output_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()