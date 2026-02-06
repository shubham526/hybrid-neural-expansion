from __future__ import print_function
from collections import defaultdict
import os
import math
import json
import numpy as np
from scipy import stats
from argparse import ArgumentParser

__author__ = 'dietz, enhanced by chatterjee'


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


tooldescription = """
Enhanced hurts/helps analysis for IR papers. 
Computes helps/hurts statistics, magnitudes, statistical significance, 
and generates paper-ready output.
"""

parser = ArgumentParser(description=tooldescription)
parser.add_argument('--metric', help='metric for comparison', required=True)
parser.add_argument('--delta', help='Minimum difference to be considered', type=float, default=0.00)
parser.add_argument('--format', help='trec_eval output or galago_eval output', default='trec_eval')
parser.add_argument('--output-json', help='Output detailed JSON results', type=str, default=None)
parser.add_argument('--output-csv', help='Output CSV with per-query details', type=str, default=None)
parser.add_argument('--show-magnitudes', help='Show average improvements/degradations', action='store_true')
parser.add_argument('--show-stats', help='Show statistical significance (paired t-test)', action='store_true')
parser.add_argument('--show-worst', help='Show N worst failures', type=int, default=0)
parser.add_argument('--show-best', help='Show N best improvements', type=int, default=0)
parser.add_argument('--paper-text', help='Generate paper-ready text', action='store_true')
parser.add_argument('--verbose', help='Verbose output with detailed per-query info', action='store_true')
parser.add_argument(dest='runs', nargs='+', type=lambda x: is_valid_file(parser, x))


def read_ssv(fname, format_type):
    """Read space-separated value file (trec_eval or galago_eval format)"""
    lines = [line.split() for line in open(fname, 'r')]
    if format_type.lower() == 'galago_eval':
        return lines
    elif format_type.lower() == 'trec_eval':
        return [[line[1], line[0]] + line[2:] for line in lines]
    return lines


def fetchValues(run, metric, format_type):
    """Extract metric values for all queries from a run"""
    tsv = read_ssv(run, format_type)
    data = {row[0]: float(row[2]) for row in tsv if row[1] == metric}
    return data


def findQueriesWithNanValues(run, format_type):
    """Find queries with no relevance judgments"""
    tsv = read_ssv(run, format_type)
    queriesWithNan = {row[0] for row in tsv
                      if row[1] == 'num_rel' and
                      (float(row[2]) == 0.0 or math.isnan(float(row[2])))}
    return queriesWithNan


def compute_statistics(baseline_values, run_values, queries):
    """Compute detailed statistics for comparison"""
    baseline_scores = [baseline_values[q] for q in queries]
    run_scores = [run_values[q] for q in queries]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(run_scores, baseline_scores)

    # Effect size (Cohen's d)
    differences = [run_scores[i] - baseline_scores[i] for i in range(len(queries))]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_difference': mean_diff,
        'std_difference': std_diff
    }


def analyze_improvements(baseline_data, run_data, queries, delta=0.0):
    """Analyze improvements, degradations, and unchanged queries"""
    helps = []
    hurts = []
    unchanged = []
    improvements = []

    for query_id in queries:
        base_value = baseline_data[query_id]
        run_value = run_data[query_id]
        difference = run_value - base_value

        if difference > delta:
            helps.append(query_id)
            improvements.append({
                'query_id': query_id,
                'baseline': base_value,
                'run': run_value,
                'delta': difference,
                'type': 'improvement'
            })
        elif difference < -delta:
            hurts.append(query_id)
            improvements.append({
                'query_id': query_id,
                'baseline': base_value,
                'run': run_value,
                'delta': difference,
                'type': 'degradation'
            })
        else:
            unchanged.append(query_id)
            improvements.append({
                'query_id': query_id,
                'baseline': base_value,
                'run': run_value,
                'delta': difference,
                'type': 'unchanged'
            })

    return helps, hurts, unchanged, improvements


def compute_magnitude_statistics(improvements):
    """Compute magnitude statistics for improvements and degradations"""
    helped = [imp for imp in improvements if imp['type'] == 'improvement']
    hurt = [imp for imp in improvements if imp['type'] == 'degradation']

    stats_dict = {
        'helped': {
            'count': len(helped),
            'mean_improvement': np.mean([imp['delta'] for imp in helped]) if helped else 0,
            'median_improvement': np.median([imp['delta'] for imp in helped]) if helped else 0,
            'max_improvement': max([imp['delta'] for imp in helped]) if helped else 0,
            'min_improvement': min([imp['delta'] for imp in helped]) if helped else 0,
        },
        'hurt': {
            'count': len(hurt),
            'mean_degradation': np.mean([imp['delta'] for imp in hurt]) if hurt else 0,
            'median_degradation': np.median([imp['delta'] for imp in hurt]) if hurt else 0,
            'max_degradation': max([imp['delta'] for imp in hurt]) if hurt else 0,  # most negative
            'min_degradation': min([imp['delta'] for imp in hurt]) if hurt else 0,  # least negative
        }
    }

    return stats_dict


def generate_paper_text(run_name, total_queries, helps_count, hurts_count,
                        magnitude_stats, metric_name):
    """Generate paper-ready text"""
    pct_helped = (helps_count / total_queries * 100) if total_queries > 0 else 0
    pct_hurt = (hurts_count / total_queries * 100) if total_queries > 0 else 0

    text = f"""
=== PAPER-READY TEXT ===

Per-query analysis reveals the method improves {pct_helped:.1f}% of queries 
({helps_count}/{total_queries}) on {metric_name}, with {pct_hurt:.1f}% 
showing degradation.
"""

    if magnitude_stats:
        helped = magnitude_stats['helped']
        hurt = magnitude_stats['hurt']

        text += f"""
For queries where the method improves performance, the average gain is 
{helped['mean_improvement']:+.3f} {metric_name} (median: {helped['median_improvement']:+.3f}). 
For the {pct_hurt:.1f}% of queries with degradation, the average loss is 
{hurt['mean_degradation']:.3f} {metric_name} (median: {hurt['median_degradation']:.3f}), 
indicating favorable risk/reward characteristics.
"""

    return text


def print_worst_best_queries(improvements, n_worst, n_best):
    """Print worst failures and best improvements"""
    if n_worst > 0:
        print("\n=== WORST FAILURES ===")
        worst = sorted([imp for imp in improvements if imp['type'] == 'degradation'],
                       key=lambda x: x['delta'])[:n_worst]
        print(f"{'Query ID':<15} {'Baseline':<12} {'Run':<12} {'Delta':<12}")
        print("-" * 55)
        for imp in worst:
            print(f"{imp['query_id']:<15} {imp['baseline']:<12.4f} "
                  f"{imp['run']:<12.4f} {imp['delta']:<12.4f}")

    if n_best > 0:
        print("\n=== BEST IMPROVEMENTS ===")
        best = sorted([imp for imp in improvements if imp['type'] == 'improvement'],
                      key=lambda x: x['delta'], reverse=True)[:n_best]
        print(f"{'Query ID':<15} {'Baseline':<12} {'Run':<12} {'Delta':<12}")
        print("-" * 55)
        for imp in best:
            print(f"{imp['query_id']:<15} {imp['baseline']:<12.4f} "
                  f"{imp['run']:<12.4f} {imp['delta']:<12.4f}")


def save_json_output(results, output_file):
    """Save detailed results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed JSON results saved to: {output_file}")


def save_csv_output(improvements, output_file):
    """Save per-query details to CSV"""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'baseline', 'run', 'delta', 'type'])
        writer.writeheader()
        writer.writerows(improvements)

    print(f"✅ Per-query CSV saved to: {output_file}")


def main():
    args = parser.parse_args()

    # Load all runs
    datas = {run: fetchValues(run, args.metric, args.format) for run in args.runs}

    # Find queries with no judgments
    queriesWithNanValues = {'all'}.union(
        *[findQueriesWithNanValues(run, args.format) for run in args.runs]
    )

    # Get valid queries
    basedata = datas[args.runs[0]]
    queries = sorted(set(basedata.keys()).difference(queriesWithNanValues))

    print(f"\n{'=' * 80}")
    print(f"HELPS/HURTS ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Baseline: {args.runs[0]}")
    print(f"Metric: {args.metric}")
    print(f"Delta threshold: {args.delta}")
    print(f"Total valid queries: {len(queries)}")
    print(f"{'=' * 80}\n")

    # Store all results
    all_results = {}

    # Basic output (original functionality)
    print(f"{'Run':<50} {'Helps':<10} {'Hurts':<10} {'Unchanged':<10}")
    print("-" * 80)

    for run in args.runs[1:]:  # Skip baseline
        run_data = datas[run]

        # Analyze improvements
        helps, hurts, unchanged, improvements = analyze_improvements(
            basedata, run_data, queries, args.delta
        )

        # Store results
        result_dict = {
            'run_name': run,
            'baseline_name': args.runs[0],
            'metric': args.metric,
            'total_queries': len(queries),
            'helps': {
                'count': len(helps),
                'queries': helps
            },
            'hurts': {
                'count': len(hurts),
                'queries': hurts
            },
            'unchanged': {
                'count': len(unchanged),
                'queries': unchanged
            },
            'improvements': improvements
        }

        # Print basic stats
        print(f"{run:<50} {len(helps):<10} {len(hurts):<10} {len(unchanged):<10}")

        # Magnitude analysis
        if args.show_magnitudes:
            magnitude_stats = compute_magnitude_statistics(improvements)
            result_dict['magnitude_statistics'] = magnitude_stats

            print(f"\n--- Magnitude Statistics for {run} ---")
            helped = magnitude_stats['helped']
            hurt = magnitude_stats['hurt']

            if helped['count'] > 0:
                print(f"  Helped queries ({helped['count']}):")
                print(f"    Mean improvement: {helped['mean_improvement']:+.4f}")
                print(f"    Median improvement: {helped['median_improvement']:+.4f}")
                print(f"    Max improvement: {helped['max_improvement']:+.4f}")

            if hurt['count'] > 0:
                print(f"  Hurt queries ({hurt['count']}):")
                print(f"    Mean degradation: {hurt['mean_degradation']:.4f}")
                print(f"    Median degradation: {hurt['median_degradation']:.4f}")
                print(f"    Worst degradation: {hurt['max_degradation']:.4f}")

            print()

        # Statistical significance
        if args.show_stats:
            sig_stats = compute_statistics(basedata, run_data, queries)
            result_dict['statistical_significance'] = sig_stats

            print(f"\n--- Statistical Significance for {run} ---")
            print(f"  Paired t-test:")
            print(f"    t-statistic: {sig_stats['t_statistic']:.4f}")
            print(f"    p-value: {sig_stats['p_value']:.6f}")
            print(f"    Significant: {'Yes' if sig_stats['p_value'] < 0.05 else 'No'} (p<0.05)")
            print(f"  Effect size:")
            print(f"    Cohen's d: {sig_stats['cohens_d']:.4f}")
            print(f"    Mean difference: {sig_stats['mean_difference']:+.4f}")
            print()

        # Show worst/best queries
        if args.show_worst > 0 or args.show_best > 0:
            print_worst_best_queries(improvements, args.show_worst, args.show_best)
            print()

        # Paper-ready text
        if args.paper_text:
            magnitude_stats = compute_magnitude_statistics(improvements)
            paper_text = generate_paper_text(
                run, len(queries), len(helps), len(hurts),
                magnitude_stats, args.metric
            )
            print(paper_text)
            result_dict['paper_text'] = paper_text

        # Verbose per-query output
        if args.verbose:
            print(f"\n--- Detailed Per-Query Results for {run} ---")
            print(f"{'Query':<15} {'Baseline':<12} {'Run':<12} {'Delta':<12} {'Status':<12}")
            print("-" * 65)
            for imp in sorted(improvements, key=lambda x: x['delta'], reverse=True):
                print(f"{imp['query_id']:<15} {imp['baseline']:<12.4f} "
                      f"{imp['run']:<12.4f} {imp['delta']:<+12.4f} {imp['type']:<12}")
            print()

        # Store in all_results
        all_results[run] = result_dict

        # Original output format (for backward compatibility)
        if not (args.show_magnitudes or args.show_stats or args.paper_text or args.verbose):
            print(f"\nHelps: {' '.join(helps)}")
            print(f"Hurts: {' '.join(hurts)}\n")

    # Save JSON output
    if args.output_json:
        save_json_output(all_results, args.output_json)

    # Save CSV output
    if args.output_csv and len(args.runs) == 2:  # Only works for single comparison
        run_name = args.runs[1]
        save_csv_output(all_results[run_name]['improvements'], args.output_csv)

    print(f"\n{'=' * 80}")
    print("Analysis complete!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()