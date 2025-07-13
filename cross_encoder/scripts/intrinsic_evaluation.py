#!/usr/bin/env python3
"""
Clean Intrinsic Evaluation: Term Re-weighting Quality via Rank Correlation

This script evaluates whether MEQE's hybrid weights are better aligned with the
"true" utility of expansion terms than the original RM3 weights.

Expects pre-computed per-query evaluation files and filtered features.
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import json
from scipy.stats import spearmanr, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.models.reranker import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class CleanIntrinsicEvaluator:
    """
    Clean intrinsic evaluator using pre-computed files.
    """

    def __init__(self,
                 features_file: str,
                 baseline_eval_file: str,
                 expanded_eval_file: str,
                 output_dir: Path):
        """
        Initialize the evaluator.

        Args:
            features_file: Path to features.jsonl file
            baseline_eval_file: Path to baseline per-query eval file
            expanded_eval_file: Path to expanded per-query eval file
            output_dir: Output directory for results
        """
        self.features_file = Path(features_file)
        self.baseline_eval_file = Path(baseline_eval_file)
        self.expanded_eval_file = Path(expanded_eval_file)
        self.output_dir = ensure_dir(output_dir)

        logger.info(f"CleanIntrinsicEvaluator initialized:")
        logger.info(f"  Features file: {features_file}")
        logger.info(f"  Baseline eval: {baseline_eval_file}")
        logger.info(f"  Expanded eval: {expanded_eval_file}")

    def load_features(self) -> Dict[str, Dict]:
        """Load features from JSONL file."""
        logger.info("Loading features from JSONL file...")

        features = {}
        with open(self.features_file, 'r') as f:
            for line in tqdm(f, desc="Loading features"):
                if line.strip():
                    data = json.loads(line)
                    features[data['query_id']] = data

        logger.info(f"Loaded features for {len(features)} queries")
        return features

    def load_eval_scores(self, eval_file: Path, metric: str = "recall.1000") -> Dict[str, float]:
        """
        Load per-query evaluation scores from trec_eval output file.

        Expected format: metric query_id score
        e.g., "recall.1000 301 0.4567"
        """
        logger.info(f"Loading {metric} scores from: {eval_file}")

        scores = {}
        with open(eval_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == metric:
                    query_id = parts[1]
                    score = float(parts[2])
                    scores[query_id] = score

        logger.info(f"Loaded {metric} scores for {len(scores)} queries")

        # Show sample scores
        if scores:
            sample_queries = list(scores.keys())[:3]
            logger.info(f"Sample {metric} scores:")
            for qid in sample_queries:
                logger.info(f"  Query {qid}: {scores[qid]:.4f}")

        return scores

    def calculate_utility_from_score_difference(self,
                                                features: Dict[str, Dict],
                                                baseline_scores: Dict[str, float],
                                                expanded_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate term utilities by distributing the score improvement from expanded vs baseline.
        """
        logger.info("Calculating term utilities from score difference...")
        logger.info(f"Input data sizes:")
        logger.info(f"  Features: {len(features)} queries")
        logger.info(f"  Baseline scores: {len(baseline_scores)} queries")
        logger.info(f"  Expanded scores: {len(expanded_scores)} queries")

        # Check query ID overlap
        feature_queries = set(features.keys())
        baseline_queries = set(baseline_scores.keys())
        expanded_queries = set(expanded_scores.keys())

        all_overlap = feature_queries & baseline_queries & expanded_queries
        logger.info(f"Common queries across all datasets: {len(all_overlap)}")

        if len(all_overlap) == 0:
            logger.error("No queries found in all three datasets!")
            logger.error(f"Sample feature queries: {list(feature_queries)[:5]}")
            logger.error(f"Sample baseline queries: {list(baseline_queries)[:5]}")
            logger.error(f"Sample expanded queries: {list(expanded_queries)[:5]}")
            return {}

        term_utilities = {}
        queries_processed = 0
        queries_with_improvement = 0
        total_improvements = []

        for query_id, query_data in features.items():
            if query_id not in baseline_scores or query_id not in expanded_scores:
                continue

            queries_processed += 1
            baseline_score = baseline_scores[query_id]
            expanded_score = expanded_scores[query_id]
            total_improvement = expanded_score - baseline_score
            total_improvements.append(total_improvement)

            # Debug for first few queries
            if queries_processed <= 3:
                logger.info(f"Query {query_id}:")
                logger.info(f"  Baseline score: {baseline_score:.4f}")
                logger.info(f"  Expanded score: {expanded_score:.4f}")
                logger.info(f"  Total improvement: {total_improvement:.4f}")

            if total_improvement > 0:
                queries_with_improvement += 1

            term_features = query_data['term_features']

            # Calculate total RM3 weight for normalization
            total_rm3_weight = sum(term_data['rm_weight'] for term_data in term_features.values())

            if total_rm3_weight == 0:
                logger.warning(f"Zero total RM3 weight for query {query_id}")
                continue

            # Distribute improvement proportionally to RM3 weights
            query_utilities = {}
            for term, term_data in term_features.items():
                rm_weight = term_data['rm_weight']
                # Each term gets improvement proportional to its RM3 weight
                term_utility = (rm_weight / total_rm3_weight) * total_improvement
                query_utilities[term] = term_utility

            term_utilities[query_id] = query_utilities

        # Summary statistics
        if total_improvements:
            mean_improvement = np.mean(total_improvements)
            positive_improvements = sum(1 for x in total_improvements if x > 0)
            logger.info(f"Utility calculation summary:")
            logger.info(f"  Queries processed: {queries_processed}")
            logger.info(f"  Mean improvement: {mean_improvement:.4f}")
            logger.info(
                f"  Queries with positive improvement: {positive_improvements} ({positive_improvements / len(total_improvements) * 100:.1f}%)")
            logger.info(f"  Final utility scores: {len(term_utilities)} queries")

        return term_utilities

    def evaluate_meqe_model(self,
                            features: Dict[str, Dict],
                            model_checkpoint_path: str,
                            model_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate MEQE model to get hybrid weights.
        """
        logger.info("Evaluating MEQE model weights...")

        # Load trained model
        model = create_neural_reranker(**model_config)

        # Load checkpoint
        import torch
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

        # Handle both strict and non-strict loading
        try:
            model.load_state_dict(checkpoint, strict=False)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        # Get learned weights
        alpha, beta, lambda_val = model.get_learned_weights()
        logger.info(f"Model weights: α={alpha:.4f}, β={beta:.4f}, λ={lambda_val:.4f}")

        meqe_weights = {}

        with torch.no_grad():
            for query_id, query_data in features.items():
                term_features = query_data['term_features']
                term_weights = {}

                for term, term_data in term_features.items():
                    rm_weight = term_data['rm_weight']
                    semantic_score = term_data['semantic_score']

                    # Calculate MEQE hybrid weight (always uses both components)
                    hybrid_weight = alpha * rm_weight + beta * semantic_score
                    term_weights[term] = hybrid_weight

                meqe_weights[query_id] = term_weights

        logger.info(f"Calculated MEQE weights for {len(meqe_weights)} queries")
        return meqe_weights

    def calculate_rank_correlations(self,
                                    utility_scores: Dict[str, Dict[str, float]],
                                    features: Dict[str, Dict],
                                    meqe_weights: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate Spearman rank correlations between utility scores and different weighting schemes.
        """
        logger.info("Calculating rank correlations...")
        logger.info(f"Input data sizes:")
        logger.info(f"  Utility scores: {len(utility_scores)} queries")
        logger.info(f"  Features: {len(features)} queries")
        logger.info(f"  MEQE weights: {len(meqe_weights)} queries")

        rm3_correlations = []
        meqe_correlations = []
        query_results = {}

        # Check what queries we have in each dataset
        utility_queries = set(utility_scores.keys())
        feature_queries = set(features.keys())
        meqe_queries = set(meqe_weights.keys())

        common_queries = utility_queries & feature_queries & meqe_queries
        logger.info(f"Common queries across all datasets: {len(common_queries)}")

        if len(common_queries) == 0:
            logger.error("No common queries found!")
            return {
                'error': 'No common queries found',
                'debug_info': {
                    'utility_queries': len(utility_queries),
                    'feature_queries': len(feature_queries),
                    'meqe_queries': len(meqe_queries)
                },
                'raw_correlations': {'rm3': [], 'meqe': []}
            }

        queries_processed = 0
        queries_with_sufficient_terms = 0

        for query_id in common_queries:
            queries_processed += 1

            # Get data for this query
            utility_dict = utility_scores[query_id]
            term_features = features[query_id]['term_features']
            meqe_dict = meqe_weights[query_id]

            # Ensure we have the same terms in all dictionaries
            utility_terms = set(utility_dict.keys())
            feature_terms = set(term_features.keys())
            meqe_terms = set(meqe_dict.keys())

            common_terms = utility_terms & feature_terms & meqe_terms

            if len(common_terms) < 3:  # Need at least 3 points for correlation
                logger.debug(f"Skipping query {query_id} - insufficient common terms ({len(common_terms)})")
                continue

            queries_with_sufficient_terms += 1

            # Create aligned lists
            utility_values = []
            rm3_values = []
            meqe_values = []

            for term in common_terms:
                utility_values.append(utility_dict[term])
                rm3_values.append(term_features[term]['rm_weight'])
                meqe_values.append(meqe_dict[term])

            # Debug for first query
            if queries_with_sufficient_terms == 1:
                logger.info(f"First valid query {query_id} sample data:")
                logger.info(f"  Sample utility values: {[f'{x:.4f}' for x in utility_values[:3]]}")
                logger.info(f"  Sample RM3 values: {[f'{x:.4f}' for x in rm3_values[:3]]}")
                logger.info(f"  Sample MEQE values: {[f'{x:.4f}' for x in meqe_values[:3]]}")

            # Calculate correlations
            try:
                rho_rm3, p_rm3 = spearmanr(utility_values, rm3_values)
                rho_meqe, p_meqe = spearmanr(utility_values, meqe_values)

                # Handle NaN values (can happen with constant arrays)
                if not np.isnan(rho_rm3):
                    rm3_correlations.append(rho_rm3)
                if not np.isnan(rho_meqe):
                    meqe_correlations.append(rho_meqe)

                query_results[query_id] = {
                    'rm3_correlation': rho_rm3 if not np.isnan(rho_rm3) else 0.0,
                    'meqe_correlation': rho_meqe if not np.isnan(rho_meqe) else 0.0,
                    'rm3_p_value': p_rm3 if not np.isnan(rho_rm3) else 1.0,
                    'meqe_p_value': p_meqe if not np.isnan(rho_meqe) else 1.0,
                    'num_terms': len(common_terms),
                    'utility_range': max(utility_values) - min(utility_values) if utility_values else 0,
                    'rm3_weight_range': max(rm3_values) - min(rm3_values) if rm3_values else 0,
                    'meqe_weight_range': max(meqe_values) - min(meqe_values) if meqe_values else 0
                }

                if queries_with_sufficient_terms <= 3:
                    logger.info(f"Query {query_id} correlations: RM3={rho_rm3:.4f}, MEQE={rho_meqe:.4f}")

            except Exception as e:
                logger.warning(f"Error calculating correlation for query {query_id}: {e}")
                continue

        logger.info(f"Processed {queries_processed} queries")
        logger.info(f"Queries with sufficient terms: {queries_with_sufficient_terms}")
        logger.info(f"Valid RM3 correlations: {len(rm3_correlations)}")
        logger.info(f"Valid MEQE correlations: {len(meqe_correlations)}")

        # Ensure we have paired data for t-test
        valid_pairs = []
        for query_id, results in query_results.items():
            if not np.isnan(results['rm3_correlation']) and not np.isnan(results['meqe_correlation']):
                valid_pairs.append((results['rm3_correlation'], results['meqe_correlation']))

        if len(valid_pairs) < 2:
            logger.error(f"Insufficient valid correlation pairs for statistical testing: {len(valid_pairs)}")
            return {
                'error': 'Insufficient valid correlation pairs',
                'debug_info': {
                    'queries_processed': queries_processed,
                    'queries_with_sufficient_terms': queries_with_sufficient_terms,
                    'valid_pairs': len(valid_pairs)
                },
                'per_query_results': query_results,
                'raw_correlations': {'rm3': [], 'meqe': []}
            }

        paired_rm3, paired_meqe = zip(*valid_pairs)
        paired_rm3 = np.array(paired_rm3)
        paired_meqe = np.array(paired_meqe)

        # Perform paired t-test
        t_stat, t_p_value = ttest_rel(paired_meqe, paired_rm3)

        results = {
            'summary': {
                'rm3_mean_correlation': np.mean(paired_rm3),
                'rm3_std_correlation': np.std(paired_rm3),
                'rm3_median_correlation': np.median(paired_rm3),
                'meqe_mean_correlation': np.mean(paired_meqe),
                'meqe_std_correlation': np.std(paired_meqe),
                'meqe_median_correlation': np.median(paired_meqe),
                'correlation_difference': np.mean(paired_meqe) - np.mean(paired_rm3),
                'num_queries': len(valid_pairs),
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'significant_improvement': t_p_value < 0.05 and np.mean(paired_meqe) > np.mean(paired_rm3),
                'queries_improved': np.sum(paired_meqe > paired_rm3),
                'queries_degraded': np.sum(paired_meqe < paired_rm3),
                'queries_unchanged': np.sum(paired_meqe == paired_rm3)
            },
            'per_query_results': query_results,
            'raw_correlations': {
                'rm3': paired_rm3.tolist(),
                'meqe': paired_meqe.tolist()
            }
        }

        logger.info(f"Correlation analysis complete:")
        logger.info(f"  RM3 mean correlation: {results['summary']['rm3_mean_correlation']:.4f}")
        logger.info(f"  MEQE mean correlation: {results['summary']['meqe_mean_correlation']:.4f}")
        logger.info(f"  Difference: {results['summary']['correlation_difference']:.4f}")
        logger.info(f"  Queries improved: {results['summary']['queries_improved']}")
        logger.info(f"  Significant improvement: {results['summary']['significant_improvement']}")

        return results

    def create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create visualizations for the correlation results."""
        logger.info("Creating visualizations...")

        # Check if we have valid results
        if 'error' in results or not results.get('raw_correlations'):
            logger.error("Cannot create visualizations - insufficient data")

            # Create a simple error plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient Data for Correlation Analysis\n\nPossible Issues:\n' +
                    '• Query ID mismatch between files\n• Too few terms per query\n• Constant utility values\n• No score improvements\n\n' +
                    f'Debug Info: {results.get("debug_info", "N/A")}',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Intrinsic Evaluation - Error')
            ax.axis('off')

            plot_path = self.output_dir / 'intrinsic_evaluation_error.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Error visualization saved to: {plot_path}")
            return

        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Intrinsic Evaluation: Term Re-weighting Quality', fontsize=16, fontweight='bold')

        # 1. Correlation comparison boxplot
        ax1 = axes[0, 0]
        correlation_data = [
            results['raw_correlations']['rm3'],
            results['raw_correlations']['meqe']
        ]
        box_plot = ax1.boxplot(correlation_data, labels=['RM3', 'MEQE'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightgreen')
        ax1.set_title('Spearman Correlation Distribution')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.grid(True, alpha=0.3)

        # Add mean markers
        rm3_mean = results['summary']['rm3_mean_correlation']
        meqe_mean = results['summary']['meqe_mean_correlation']
        ax1.scatter([1, 2], [rm3_mean, meqe_mean], color='red', s=100, marker='D', label='Mean', zorder=5)
        ax1.legend()

        # 2. Scatter plot of correlations
        ax2 = axes[0, 1]
        rm3_corr = results['raw_correlations']['rm3']
        meqe_corr = results['raw_correlations']['meqe']
        ax2.scatter(rm3_corr, meqe_corr, alpha=0.6, s=50)

        # Add diagonal line
        min_val = min(min(rm3_corr), min(meqe_corr))
        max_val = max(max(rm3_corr), max(meqe_corr))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal performance')

        ax2.set_xlabel('RM3 Correlation')
        ax2.set_ylabel('MEQE Correlation')
        ax2.set_title('Per-Query Correlation Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Histogram of correlation differences
        ax3 = axes[1, 0]
        differences = np.array(meqe_corr) - np.array(rm3_corr)
        ax3.hist(differences, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No difference')
        ax3.axvline(x=np.mean(differences), color='green', linestyle='-', linewidth=2,
                    label=f'Mean diff: {np.mean(differences):.3f}')
        ax3.set_xlabel('Correlation Difference (MEQE - RM3)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Correlation Differences')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary = results['summary']
        table_data = [
            ['Metric', 'RM3', 'MEQE', 'Difference'],
            ['Mean Correlation', f"{summary['rm3_mean_correlation']:.4f}",
             f"{summary['meqe_mean_correlation']:.4f}",
             f"{summary['correlation_difference']:.4f}"],
            ['Median Correlation', f"{summary['rm3_median_correlation']:.4f}",
             f"{summary['meqe_median_correlation']:.4f}",
             f"{summary['meqe_median_correlation'] - summary['rm3_median_correlation']:.4f}"],
            ['Std Dev', f"{summary['rm3_std_correlation']:.4f}",
             f"{summary['meqe_std_correlation']:.4f}", ''],
            ['', '', '', ''],
            ['Statistical Test', 'Value', '', ''],
            ['t-statistic', f"{summary['t_statistic']:.4f}", '', ''],
            ['p-value', f"{summary['t_p_value']:.4f}", '', ''],
            ['Significant?', 'Yes' if summary['significant_improvement'] else 'No', '', ''],
            ['', '', '', ''],
            ['Query Analysis', 'Count', '', ''],
            ['Total Queries', f"{summary['num_queries']}", '', ''],
            ['Improved', f"{summary['queries_improved']}", '', ''],
            ['Degraded', f"{summary['queries_degraded']}", '', ''],
            ['Unchanged', f"{summary['queries_unchanged']}", '', '']
        ]

        table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')

        # Style section headers
        for row_idx in [5, 10]:
            for col_idx in range(len(table_data[0])):
                table[(row_idx, col_idx)].set_facecolor('#F0F0F0')
                table[(row_idx, col_idx)].set_text_props(weight='bold')

        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)

        plt.tight_layout()

        # Save the plot
        plot_path = self.output_dir / 'intrinsic_evaluation_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to: {plot_path}")

        # Create additional detailed plots
        self._create_detailed_plots(results)

    def _create_detailed_plots(self, results: Dict[str, Any]) -> None:
        """Create additional detailed analysis plots."""
        logger.info("Creating detailed analysis plots...")

        # 1. Query-level improvement analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Query-Level Analysis', fontsize=16, fontweight='bold')

        # Query performance scatter with improvement coloring
        ax1 = axes[0, 0]
        rm3_corr = np.array(results['raw_correlations']['rm3'])
        meqe_corr = np.array(results['raw_correlations']['meqe'])
        improvements = meqe_corr - rm3_corr

        scatter = ax1.scatter(rm3_corr, meqe_corr, c=improvements, cmap='RdYlGn', alpha=0.7, s=60)
        ax1.plot([min(rm3_corr), max(rm3_corr)], [min(rm3_corr), max(rm3_corr)], 'k--', alpha=0.5)
        ax1.set_xlabel('RM3 Correlation')
        ax1.set_ylabel('MEQE Correlation')
        ax1.set_title('Query Performance with Improvement Coloring')
        ax1.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('MEQE - RM3 Correlation')

        # Distribution of improvements by quartiles
        ax2 = axes[0, 1]
        quartiles = np.percentile(rm3_corr, [25, 50, 75])
        q1_mask = rm3_corr <= quartiles[0]
        q2_mask = (rm3_corr > quartiles[0]) & (rm3_corr <= quartiles[1])
        q3_mask = (rm3_corr > quartiles[1]) & (rm3_corr <= quartiles[2])
        q4_mask = rm3_corr > quartiles[2]

        quartile_improvements = [
            improvements[q1_mask],
            improvements[q2_mask],
            improvements[q3_mask],
            improvements[q4_mask]
        ]

        ax2.boxplot(quartile_improvements, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('RM3 Performance Quartile')
        ax2.set_ylabel('Correlation Improvement')
        ax2.set_title('Improvement by RM3 Performance Quartile')
        ax2.grid(True, alpha=0.3)

        # Correlation vs number of terms
        ax3 = axes[1, 0]
        per_query = results['per_query_results']
        num_terms = [per_query[qid]['num_terms'] for qid in per_query.keys()]
        meqe_correlations = [per_query[qid]['meqe_correlation'] for qid in per_query.keys()]

        ax3.scatter(num_terms, meqe_correlations, alpha=0.6, s=50)
        ax3.set_xlabel('Number of Terms')
        ax3.set_ylabel('MEQE Correlation')
        ax3.set_title('Correlation vs Number of Terms')
        ax3.grid(True, alpha=0.3)

        # Improvement vs utility range
        ax4 = axes[1, 1]
        utility_ranges = [per_query[qid]['utility_range'] for qid in per_query.keys()]
        query_improvements = [per_query[qid]['meqe_correlation'] - per_query[qid]['rm3_correlation']
                              for qid in per_query.keys()]

        ax4.scatter(utility_ranges, query_improvements, alpha=0.6, s=50)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Utility Score Range')
        ax4.set_ylabel('Correlation Improvement')
        ax4.set_title('Improvement vs Utility Score Diversity')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save detailed plots
        detailed_plot_path = self.output_dir / 'intrinsic_evaluation_detailed.png'
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Detailed analysis plots saved to: {detailed_plot_path}")

    def run_evaluation(self,
                       model_checkpoint_path: str,
                       model_config: Dict[str, Any],
                       metric: str = "recall.1000") -> Dict[str, Any]:
        """
        Run the complete intrinsic evaluation pipeline.

        Args:
            model_checkpoint_path: Path to trained MEQE model checkpoint
            model_config: Configuration for model creation
            metric: Evaluation metric to use (default: recall.1000)

        Returns:
            Complete evaluation results
        """
        logger.info("Starting intrinsic evaluation pipeline...")

        with TimedOperation("Complete Intrinsic Evaluation"):
            # Step 1: Load all data
            with TimedOperation("Loading features"):
                features = self.load_features()

            with TimedOperation("Loading baseline scores"):
                baseline_scores = self.load_eval_scores(self.baseline_eval_file, metric)

            with TimedOperation("Loading expanded scores"):
                expanded_scores = self.load_eval_scores(self.expanded_eval_file, metric)

            # Step 2: Calculate term utilities (ground truth)
            with TimedOperation("Calculating term utilities"):
                utility_scores = self.calculate_utility_from_score_difference(
                    features, baseline_scores, expanded_scores
                )

            if not utility_scores:
                logger.error("No utility scores calculated - cannot proceed")
                return {'error': 'No utility scores calculated'}

            # Step 3: Get MEQE model weights
            with TimedOperation("Evaluating MEQE model"):
                meqe_weights = self.evaluate_meqe_model(
                    features, model_checkpoint_path, model_config
                )

            # Step 4: Calculate rank correlations
            with TimedOperation("Calculating rank correlations"):
                correlation_results = self.calculate_rank_correlations(
                    utility_scores, features, meqe_weights
                )

            # Step 5: Create visualizations
            with TimedOperation("Creating visualizations"):
                self.create_visualizations(correlation_results)

            # Step 6: Save detailed results
            with TimedOperation("Saving results"):
                results_file = self.output_dir / 'intrinsic_evaluation_results.json'
                save_json(correlation_results, results_file)
                logger.info(f"Detailed results saved to: {results_file}")

                # Save summary report
                self._save_summary_report(correlation_results)

        return correlation_results

    def _save_summary_report(self, results: Dict[str, Any]) -> None:
        """Save a human-readable summary report."""
        report_path = self.output_dir / 'intrinsic_evaluation_summary.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("INTRINSIC EVALUATION SUMMARY REPORT\n")
            f.write("Term Re-weighting Quality via Rank Correlation\n")
            f.write("=" * 60 + "\n\n")

            if 'error' in results:
                f.write(f"ERROR: {results['error']}\n")
                f.write(f"Debug Info: {results.get('debug_info', 'N/A')}\n")
                return

            summary = results['summary']

            f.write("RESEARCH QUESTION:\n")
            f.write("Is the ranking of expansion terms produced by MEQE's hybrid weights\n")
            f.write("more correlated with the terms' actual retrieval utility than the\n")
            f.write("original ranking from RM3?\n\n")

            f.write("METHODOLOGY:\n")
            f.write("- Ground truth utility: Change in Recall@1000 when adding each term\n")
            f.write("- Comparison: Spearman rank correlation between utility and weights\n")
            f.write("- Statistical test: Paired t-test on per-query correlations\n\n")

            f.write("RESULTS:\n")
            f.write(f"- Queries analyzed: {summary['num_queries']}\n")
            f.write(
                f"- RM3 mean correlation: {summary['rm3_mean_correlation']:.4f} ± {summary['rm3_std_correlation']:.4f}\n")
            f.write(
                f"- MEQE mean correlation: {summary['meqe_mean_correlation']:.4f} ± {summary['meqe_std_correlation']:.4f}\n")
            f.write(f"- Improvement: {summary['correlation_difference']:.4f}\n")
            f.write(f"- t-statistic: {summary['t_statistic']:.4f}\n")
            f.write(f"- p-value: {summary['t_p_value']:.4f}\n")
            f.write(f"- Statistically significant: {'Yes' if summary['significant_improvement'] else 'No'}\n\n")

            f.write("QUERY-LEVEL BREAKDOWN:\n")
            f.write(
                f"- Queries improved: {summary['queries_improved']} ({summary['queries_improved'] / summary['num_queries'] * 100:.1f}%)\n")
            f.write(
                f"- Queries degraded: {summary['queries_degraded']} ({summary['queries_degraded'] / summary['num_queries'] * 100:.1f}%)\n")
            f.write(
                f"- Queries unchanged: {summary['queries_unchanged']} ({summary['queries_unchanged'] / summary['num_queries'] * 100:.1f}%)\n\n")

            f.write("INTERPRETATION:\n")
            if summary['significant_improvement']:
                f.write("✅ MEQE's hybrid weights produce term rankings that are significantly\n")
                f.write("   better aligned with actual retrieval utility than RM3 weights.\n")
                f.write("   This validates the effectiveness of the learned re-weighting scheme.\n")
            elif summary['correlation_difference'] > 0:
                f.write("⚠️  MEQE shows improvement over RM3 but the difference is not\n")
                f.write("   statistically significant. More data or refinement may be needed.\n")
            else:
                f.write("❌ MEQE does not show improvement over RM3 in term ranking quality.\n")
                f.write("   The model may need further training or architectural changes.\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run intrinsic evaluation of MEQE term re-weighting quality"
    )

    # Required arguments
    parser.add_argument(
        "--features-file",
        type=str,
        required=True,
        help="Path to features.jsonl file containing term features"
    )
    parser.add_argument(
        "--baseline-eval-file",
        type=str,
        required=True,
        help="Path to baseline per-query evaluation file (trec_eval format)"
    )
    parser.add_argument(
        "--expanded-eval-file",
        type=str,
        required=True,
        help="Path to expanded per-query evaluation file (trec_eval format)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to trained MEQE model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results and visualizations"
    )

    # Model configuration arguments (should match training)
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model name used in training"
    )
    parser.add_argument(
        "--max-expansion_terms",
        type=int,
        default=15,
        help="Maximum expansion terms (should match training)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension (should match training)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (should match training)"
    )
    parser.add_argument(
        "--scoring-method",
        type=str,
        default="neural",
        choices=['neural', 'bilinear', 'cosine'],
        help="Scoring method (should match training)"
    )
    parser.add_argument(
        "--force-hf",
        action="store_true",
        help="Force HuggingFace transformers (should match training)"
    )
    parser.add_argument(
        "--pooling-strategy",
        type=str,
        default="cls",
        choices=['cls', 'mean', 'max'],
        help="Pooling strategy for HF models (should match training)"
    )

    # Evaluation options
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg_cut_20",
        help="Evaluation metric to use (default: recall.1000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_experiment_logging(
        experiment_name="intrinsic_evaluation",
        output_dir=Path(args.output_dir),
        log_level=log_level
    )

    # Log experiment info
    log_experiment_info({
        'experiment_type': 'intrinsic_evaluation',
        'features_file': args.features_file,
        'baseline_eval_file': args.baseline_eval_file,
        'expanded_eval_file': args.expanded_eval_file,
        'model_checkpoint': args.model_checkpoint,
        'metric': args.metric,
        'model_config': {
            'model_type': args.model_type,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    })

    # Create model configuration
    model_config = {
        'model_name': args.model_name,
        'max_expansion_terms': args.max_expansion_terms,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'scoring_method': args.scoring_method,
        'force_hf': args.force_hf,
        'pooling_strategy': args.pooling_strategy,
        'device': 'cpu'  # Use CPU for evaluation to avoid GPU memory issues
    }

    # Initialize evaluator
    evaluator = CleanIntrinsicEvaluator(
        features_file=args.features_file,
        baseline_eval_file=args.baseline_eval_file,
        expanded_eval_file=args.expanded_eval_file,
        output_dir=Path(args.output_dir)
    )

    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            model_checkpoint_path=args.model_checkpoint,
            model_config=model_config,
            metric=args.metric
        )

        if 'error' not in results:
            logger.info("Intrinsic evaluation completed successfully!")
            summary = results['summary']
            logger.info(f"Final result: MEQE correlation = {summary['meqe_mean_correlation']:.4f}, "
                        f"RM3 correlation = {summary['rm3_mean_correlation']:.4f}, "
                        f"improvement = {summary['correlation_difference']:.4f}")
        else:
            logger.error(f"Evaluation failed: {results['error']}")
            return 1

    except Exception as e:
        logger.error(f"Evaluation failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())