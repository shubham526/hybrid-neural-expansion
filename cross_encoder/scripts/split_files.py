#!/usr/bin/env python3
"""
Split files for fold-based intrinsic evaluation

This script splits features files and per-query evaluation files to specific fold queries.
"""

import argparse
import json
from pathlib import Path

def load_fold_queries(folds_file: str, fold_id: str, split: str):
    """Load query IDs for a specific fold and split."""
    with open(folds_file, 'r') as f:
        folds = json.load(f)
    
    if str(fold_id) not in folds:
        raise ValueError(f"Fold {fold_id} not found in folds file")
    
    if split not in folds[str(fold_id)]:
        raise ValueError(f"Split '{split}' not found in fold {fold_id}")
    
    return set(folds[str(fold_id)][split])

def filter_features_file(input_features: str, output_features: str, query_ids: set):
    """Filter features JSONL file to specific queries."""
    with open(input_features, 'r') as f_in, open(output_features, 'w') as f_out:
        count = 0
        for line in f_in:
            if line.strip():
                data = json.loads(line)
                if data['query_id'] in query_ids:
                    f_out.write(line)
                    count += 1
    
    print(f"Filtered features file: {count} queries -> {output_features}")
    return count

def filter_trec_eval_file(input_eval: str, output_eval: str, query_ids: set, metric: str):
    """
    Filter trec_eval per-query results file to specific queries.
    
    Expected format: metric query_id score
    e.g., "recall.1000 301 0.4567"
    """
    with open(input_eval, 'r') as f_in, open(output_eval, 'w') as f_out:
        count = 0
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == metric:
                query_id = parts[1]
                if query_id in query_ids:
                    f_out.write(line)
                    count += 1
    
    print(f"Filtered {metric} file: {count} queries -> {output_eval}")
    return count

def main():
    parser = argparse.ArgumentParser(description="Split files for fold evaluation")
    parser.add_argument('--folds-file', type=str, required=True, help='Path to folds.json')
    parser.add_argument('--fold-id', type=str, required=True, help='Fold ID (e.g., "0")')
    parser.add_argument('--split', type=str, required=True, choices=['training', 'testing'], 
                        help='Which split to use')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory for filtered files')
    
    # Input files
    parser.add_argument('--features-file', type=str, required=True, help='Input features.jsonl file')
    parser.add_argument('--baseline-eval', type=str, required=True, help='Baseline per-query eval file')
    parser.add_argument('--expanded-eval', type=str, required=True, help='Expanded per-query eval file')
    parser.add_argument('--qrels-file', type=str, required=True, help='Qrels file')
    
    # Optional parameters
    parser.add_argument('--metric', type=str, default='recall.1000', help='Metric to filter')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load query IDs for the specified fold and split
    query_ids = load_fold_queries(args.folds_file, args.fold_id, args.split)
    print(f"Loaded {len(query_ids)} query IDs for fold {args.fold_id} {args.split}")
    print(f"Sample query IDs: {list(query_ids)[:5]}")
    
    # Filter features file
    output_features = output_dir / f"fold_{args.fold_id}_{args.split}_features.jsonl"
    features_count = filter_features_file(args.features_file, str(output_features), query_ids)
    
    # Filter baseline eval file
    output_baseline = output_dir / f"fold_{args.fold_id}_{args.split}_baseline_eval.txt"
    baseline_count = filter_trec_eval_file(args.baseline_eval, str(output_baseline), query_ids, args.metric)
    
    # Filter expanded eval file
    output_expanded = output_dir / f"fold_{args.fold_id}_{args.split}_expanded_eval.txt"
    expanded_count = filter_trec_eval_file(args.expanded_eval, str(output_expanded), query_ids, args.metric)
    
    # Filter qrels file
    output_qrels = output_dir / f"fold_{args.fold_id}_{args.split}_qrels.txt"
    with open(args.qrels_file, 'r') as f_in, open(output_qrels, 'w') as f_out:
        qrels_count = 0
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                if query_id in query_ids:
                    f_out.write(line)
                    qrels_count += 1
    
    print(f"Filtered qrels file: {qrels_count} judgments -> {output_qrels}")
    
    # Create summary file
    summary = {
        'fold_id': args.fold_id,
        'split': args.split,
        'total_queries': len(query_ids),
        'files_created': {
            'features': str(output_features),
            'baseline_eval': str(output_baseline),
            'expanded_eval': str(output_expanded),
            'qrels': str(output_qrels)
        },
        'counts': {
            'features': features_count,
            'baseline_eval': baseline_count,
            'expanded_eval': expanded_count,
            'qrels': qrels_count
        }
    }
    
    summary_file = output_dir / f"fold_{args.fold_id}_{args.split}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Fold {args.fold_id} {args.split}: {len(query_ids)} queries")
    print(f"  Features: {features_count} queries")
    print(f"  Baseline eval: {baseline_count} scores")
    print(f"  Expanded eval: {expanded_count} scores")
    print(f"  Qrels: {qrels_count} judgments")
    print(f"  Summary saved to: {summary_file}")
    print(f"\nFiles ready for intrinsic evaluation!")

if __name__ == "__main__":
    main()
