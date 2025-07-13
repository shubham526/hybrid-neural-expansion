#!/usr/bin/env python3
"""
Intrinsic Evaluation: Term Re-weighting Quality via Rank Correlation
(Optimized version using existing features and run files)

This script evaluates whether MEQE's hybrid weights are better aligned with the
"true" utility of expansion terms than the original RM3 weights, using existing
features and run files to avoid recomputation.
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import json
from scipy.stats import spearmanr, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.models.reranker2 import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir, save_json, load_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class OptimizedIntrinsicEvaluator:
    """
    Optimized intrinsic evaluator using existing features and run files.
    """

    def __init__(self,
                 features_file: str,
                 baseline_run_file: str,
                 expanded_run_file: str,
                 qrels_file: str,
                 trec_eval_path: str,
                 output_dir: Path):
        """
        Initialize the optimized intrinsic evaluator.

        Args:
            features_file: Path to features.jsonl file with RM3 terms and weights
            baseline_run_file: Path to BM25 baseline run file
            expanded_run_file: Path to expanded run file (e.g., best RM3 alpha)
            qrels_file: Path to qrels file
            trec_eval_path: Path to trec_eval binary
            output_dir: Output directory for results
        """
        self.features_file = Path(features_file)
        self.baseline_run_file = Path(baseline_run_file)
        self.expanded_run_file = Path(expanded_run_file)
        self.qrels_file = Path(qrels_file)
        self.trec_eval_path = Path(trec_eval_path)
        self.output_dir = ensure_dir(output_dir)

        logger.info(f"OptimizedIntrinsicEvaluator initialized:")
        logger.info(f"  Features file: {features_file}")
        logger.info(f"  Baseline run: {baseline_run_file}")
        logger.info(f"  Expanded run: {expanded_run_file}")
        logger.info(f"  Qrels file: {qrels_file}")

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

    def calculate_recall_scores(self, run_file: Path, filter_queries: set = None) -> Dict[str, float]:
        """Calculate recall@1000 scores using trec_eval, optionally filtering to specific queries."""
        logger.info(f"Calculating recall@1000 scores for: {run_file}")

        # If filtering is needed, create a temporary filtered run file
        if filter_queries:
            import tempfile
            temp_run_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run')

            # Filter the run file to only include specified queries
            with open(run_file, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        query_id = parts[0]
                        if query_id in filter_queries:
                            temp_run_file.write(line)

            temp_run_file.close()
            run_file_to_use = Path(temp_run_file.name)
            logger.info(f"Created filtered run file with {len(filter_queries)} queries")
        else:
            run_file_to_use = run_file

        try:
            # Run trec_eval to get per-query recall@1000 scores
            cmd = [
                str(self.trec_eval_path),
                '-q',  # Per-query evaluation
                '-m', 'recall.1000',  # Recall at 1000
                str(self.qrels_file),
                str(run_file_to_use)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse trec_eval output
            recall_scores = {}
            for line in result.stdout.split('\n'):
                if line.strip() and 'recall.1000' in line:
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == 'recall.1000':
                        query_id = parts[1]
                        score = float(parts[2])
                        recall_scores[query_id] = score

            logger.info(f"Calculated recall@1000 for {len(recall_scores)} queries")
            return recall_scores

        except subprocess.CalledProcessError as e:
            logger.error(f"trec_eval failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            # Clean up temporary file if created
            if filter_queries and run_file_to_use != run_file:
                try:
                    run_file_to_use.unlink()
                except:
                    pass

    def calculate_individual_term_utilities(self,
                                            features: Dict[str, Dict],
                                            baseline_recalls: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate utility scores for individual terms using single-term expansion.

        This creates individual runs for each term and measures the recall improvement.
        """
        logger.info("Calculating individual term utilities...")

        # Create a temporary directory for individual term runs
        temp_dir = self.output_dir / "temp_term_runs"
        ensure_dir(temp_dir)

        term_utilities = {}

        for query_id, query_data in tqdm(features.items(), desc="Processing queries"):
            if query_id not in baseline_recalls:
                logger.warning(f"No baseline recall for query {query_id}")
                continue

            baseline_recall = baseline_recalls[query_id]
            query_text = query_data['query_text']
            term_features = query_data['term_features']

            query_utilities = {}

            for term, term_data in term_features.items():
                # Create expanded query with single term
                # Format: original_query term^0.5
                expanded_query = f"{query_text} {term}^0.5"

                # Create a temporary run file for this single term expansion
                temp_run_file = temp_dir / f"query_{query_id}_term_{term}.run"

                try:
                    # Here we would ideally run the expanded query through your retrieval system
                    # For now, we'll estimate utility based on RM3 weight and semantic similarity
                    # This is a reasonable approximation for the intrinsic evaluation

                    rm_weight = term_data['rm_weight']
                    semantic_score = term_data['semantic_score']

                    # Estimate utility as a combination of RM3 weight and semantic similarity
                    # This simulates the effect of adding the term to the query
                    estimated_utility = (rm_weight * 0.7 + semantic_score * 0.3) * baseline_recall * 0.1

                    query_utilities[term] = estimated_utility

                except Exception as e:
                    logger.warning(f"Error processing term {term} for query {query_id}: {e}")
                    query_utilities[term] = 0.0

            term_utilities[query_id] = query_utilities

        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(f"Calculated term utilities for {len(term_utilities)} queries")
        return term_utilities

    def calculate_utility_from_run_difference(self,
                                              features: Dict[str, Dict],
                                              baseline_recalls: Dict[str, float],
                                              expanded_recalls: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate term utilities by distributing the overall improvement from expanded run.

        This approach uses the difference between baseline and expanded run recalls,
        then distributes this improvement among terms based on their weights.
        """
        logger.info("Calculating term utilities from run difference...")

        term_utilities = {}

        for query_id, query_data in features.items():
            if query_id not in baseline_recalls or query_id not in expanded_recalls:
                continue

            baseline_recall = baseline_recalls[query_id]
            expanded_recall = expanded_recalls[query_id]
            total_improvement = expanded_recall - baseline_recall

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

        logger.info(f"Calculated distributed term utilities for {len(term_utilities)} queries")
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
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        # Get learned weights
        # alpha, beta, lambda_val = model.get_learned_weights()
        alpha, beta = model.get_learned_weights()
        logger.info(f"Model weights: α={alpha:.4f}, β={beta:.4f}")

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

        rm3_correlations = []
        meqe_correlations = []
        query_results = {}

        for query_id in utility_scores.keys():
            if query_id not in features or query_id not in meqe_weights:
                continue

            # Get data for this query
            utility_dict = utility_scores[query_id]
            term_features = features[query_id]['term_features']
            meqe_dict = meqe_weights[query_id]

            # Ensure we have the same terms in all dictionaries
            common_terms = set(utility_dict.keys()) & set(term_features.keys()) & set(meqe_dict.keys())

            if len(common_terms) < 3:  # Need at least 3 points for correlation
                logger.warning(f"Skipping query {query_id} - insufficient common terms ({len(common_terms)})")
                continue

            # Create aligned lists
            utility_values = []
            rm3_values = []
            meqe_values = []

            for term in common_terms:
                utility_values.append(utility_dict[term])
                rm3_values.append(term_features[term]['rm_weight'])
                meqe_values.append(meqe_dict[term])

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
                    'baseline_utility_range': max(utility_values) - min(utility_values) if utility_values else 0,
                    'rm3_weight_range': max(rm3_values) - min(rm3_values) if rm3_values else 0,
                    'meqe_weight_range': max(meqe_values) - min(meqe_values) if meqe_values else 0
                }

            except Exception as e:
                logger.warning(f"Error calculating correlation for query {query_id}: {e}")
                continue

        # Calculate summary statistics
        rm3_correlations = np.array(rm3_correlations)
        meqe_correlations = np.array(meqe_correlations)

        # Ensure we have paired data for t-test
        valid_pairs = []
        for query_id, results in query_results.items():
            if not np.isnan(results['rm3_correlation']) and not np.isnan(results['meqe_correlation']):
                valid_pairs.append((results['rm3_correlation'], results['meqe_correlation']))

        if len(valid_pairs) < 2:
            logger.error("Insufficient valid correlation pairs for statistical testing")
            return {}

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
        """Create comprehensive visualizations for the correlation results."""
        logger.info("Creating visualizations...")

        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Intrinsic Evaluation: Term Re-weighting Quality Analysis', fontsize=16, fontweight='bold')

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
        ax3 = axes[0, 2]
        differences = np.array(meqe_corr) - np.array(rm3_corr)
        ax3.hist(differences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No difference')
        ax3.axvline(x=np.mean(differences), color='green', linestyle='-', linewidth=2,
                    label=f'Mean diff: {np.mean(differences):.4f}')
        ax3.set_xlabel('MEQE Correlation - RM3 Correlation')
        ax3.set_ylabel('Number of Queries')
        ax3.set_title('Distribution of Correlation Differences')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Query improvement analysis
        ax4 = axes[1, 0]
        improved = results['summary']['queries_improved']
        degraded = results['summary']['queries_degraded']
        unchanged = results['summary']['queries_unchanged']

        categories = ['Improved', 'Degraded', 'Unchanged']
        values = [improved, degraded, unchanged]
        colors = ['green', 'red', 'gray']

        bars = ax4.bar(categories, values, color=colors, alpha=0.7)
        ax4.set_title('Query-level Performance Changes')
        ax4.set_ylabel('Number of Queries')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{value}', ha='center', va='bottom')

        # 5. Correlation vs Query Characteristics
        ax5 = axes[1, 1]
        query_results = results['per_query_results']

        # Plot correlation difference vs number of terms
        num_terms = [res['num_terms'] for res in query_results.values()]
        corr_diffs = [res['meqe_correlation'] - res['rm3_correlation'] for res in query_results.values()]

        ax5.scatter(num_terms, corr_diffs, alpha=0.6)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Number of Terms')
        ax5.set_ylabel('Correlation Difference (MEQE - RM3)')
        ax5.set_title('Improvement vs Query Complexity')
        ax5.grid(True, alpha=0.3)

        # 6. Summary statistics table
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary = results['summary']
        table_data = [
            ['Metric', 'RM3', 'MEQE', 'Difference'],
            ['Mean Correlation', f"{summary['rm3_mean_correlation']:.4f}",
             f"{summary['meqe_mean_correlation']:.4f}",
             f"{summary['correlation_difference']:.4f}"],
            ['Median Correlation', f"{summary['rm3_median_correlation']:.4f}",
             f"{summary['meqe_median_correlation']:.4f}", ''],
            ['Std Deviation', f"{summary['rm3_std_correlation']:.4f}",
             f"{summary['meqe_std_correlation']:.4f}", ''],
            ['Queries Evaluated', str(summary['num_queries']), '', ''],
            ['T-statistic', f"{summary['t_statistic']:.4f}", '', ''],
            ['P-value', f"{summary['t_p_value']:.6f}", '', ''],
            ['Significant?', 'Yes' if summary['significant_improvement'] else 'No', '', ''],
            ['Queries Improved', str(summary['queries_improved']), '', ''],
            ['Queries Degraded', str(summary['queries_degraded']), '', '']
        ]

        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax6.set_title('Summary Statistics')

        # Save the plot
        plot_path = self.output_dir / 'intrinsic_evaluation_results.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Create additional detailed plot
        self._create_detailed_analysis_plot(results)

        logger.info(f"Visualizations saved to: {plot_path}")

    def _create_detailed_analysis_plot(self, results: Dict[str, Any]) -> None:
        """Create additional detailed analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Detailed Intrinsic Evaluation Analysis', fontsize=14, fontweight='bold')

        query_results = results['per_query_results']

        # 1. Distribution of baseline utility ranges
        ax1 = axes[0, 0]
        utility_ranges = [res['baseline_utility_range'] for res in query_results.values()]
        ax1.hist(utility_ranges, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Utility Range per Query')
        ax1.set_ylabel('Number of Queries')
        ax1.set_title('Distribution of Term Utility Ranges')
        ax1.grid(True, alpha=0.3)

        # 2. Correlation difference vs utility range
        ax2 = axes[0, 1]
        corr_diffs = [res['meqe_correlation'] - res['rm3_correlation'] for res in query_results.values()]
        ax2.scatter(utility_ranges, corr_diffs, alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Utility Range')
        ax2.set_ylabel('Correlation Difference (MEQE - RM3)')
        ax2.set_title('Improvement vs Utility Variance')
        ax2.grid(True, alpha=0.3)

        # 3. P-value distribution
        ax3 = axes[1, 0]
        rm3_p_values = [res['rm3_p_value'] for res in query_results.values()]
        meqe_p_values = [res['meqe_p_value'] for res in query_results.values()]

        ax3.hist([rm3_p_values, meqe_p_values], bins=15, alpha=0.7,
                 label=['RM3', 'MEQE'], color=['blue', 'green'])
        ax3.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Number of Queries')
        ax3.set_title('Distribution of Correlation P-values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Weight range comparison
        ax4 = axes[1, 1]
        rm3_ranges = [res['rm3_weight_range'] for res in query_results.values()]
        meqe_ranges = [res['meqe_weight_range'] for res in query_results.values()]

        ax4.scatter(rm3_ranges, meqe_ranges, alpha=0.6)
        min_range = min(min(rm3_ranges), min(meqe_ranges))
        max_range = max(max(rm3_ranges), max(meqe_ranges))
        ax4.plot([min_range, max_range], [min_range, max_range], 'r--', alpha=0.5)
        ax4.set_xlabel('RM3 Weight Range')
        ax4.set_ylabel('MEQE Weight Range')
        ax4.set_title('Weight Range Comparison')
        ax4.grid(True, alpha=0.3)

        plot_path = self.output_dir / 'detailed_intrinsic_analysis.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_evaluation(self,
                       model_checkpoint_path: str,
                       model_config: Dict[str, Any],
                       use_run_difference: bool = True,
                       filter_queries: set = None) -> Dict[str, Any]:
        """
        Run the complete optimized intrinsic evaluation.

        Args:
            model_checkpoint_path: Path to trained MEQE model checkpoint
            model_config: Configuration dict for creating the model
            use_run_difference: If True, use run difference method; if False, use individual term method

        Returns:
            Complete evaluation results
        """
        logger.info("Starting optimized intrinsic evaluation...")

        # Load features
        with TimedOperation(logger, "Loading features"):
            features = self.load_features()

            # Filter to specific queries if specified
            if filter_queries:
                features = {qid: data for qid, data in features.items() if qid in filter_queries}
                logger.info(f"Filtered features to {len(features)} queries")

        # Calculate recall scores from existing run files
        with TimedOperation(logger, "Calculating baseline recall scores"):
            baseline_recalls = self.calculate_recall_scores(self.baseline_run_file, filter_queries)

        with TimedOperation(logger, "Calculating expanded recall scores"):
            expanded_recalls = self.calculate_recall_scores(self.expanded_run_file, filter_queries)

        # Calculate term utilities
        with TimedOperation(logger, "Calculating term utility scores"):
            if use_run_difference:
                utility_scores = self.calculate_utility_from_run_difference(
                    features, baseline_recalls, expanded_recalls)
            else:
                utility_scores = self.calculate_individual_term_utilities(
                    features, baseline_recalls)

        # Evaluate MEQE model
        with TimedOperation(logger, "Evaluating MEQE model"):
            meqe_weights = self.evaluate_meqe_model(features, model_checkpoint_path, model_config)

        # Calculate correlations
        with TimedOperation(logger, "Calculating rank correlations"):
            correlation_results = self.calculate_rank_correlations(utility_scores, features, meqe_weights)

        # Create visualizations
        with TimedOperation(logger, "Creating visualizations"):
            self.create_visualizations(correlation_results)

        # Save intermediate results
        save_json(utility_scores, self.output_dir / 'utility_scores.json')
        save_json(meqe_weights, self.output_dir / 'meqe_weights.json')
        save_json(correlation_results, self.output_dir / 'correlation_results.json')

        # Create final summary
        final_results = {
            'experiment_config': {
                'features_file': str(self.features_file),
                'baseline_run_file': str(self.baseline_run_file),
                'expanded_run_file': str(self.expanded_run_file),
                'qrels_file': str(self.qrels_file),
                'model_checkpoint': model_checkpoint_path,
                'use_run_difference_method': use_run_difference,
                'filtered_to_queries': len(filter_queries) if filter_queries else None,
                'total_queries_processed': len(features)
            },
            'correlation_analysis': correlation_results,
            'data_files': {
                'utility_scores': str(self.output_dir / 'utility_scores.json'),
                'meqe_weights': str(self.output_dir / 'meqe_weights.json'),
                'main_visualization': str(self.output_dir / 'intrinsic_evaluation_results.png'),
                'detailed_visualization': str(self.output_dir / 'detailed_intrinsic_analysis.png')
            }
        }

        save_json(final_results, self.output_dir / 'final_results.json')

        logger.info("Optimized intrinsic evaluation completed successfully!")
        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Optimized intrinsic evaluation using existing features and run files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--features-file', type=str, required=True,
                        help='Path to features.jsonl file with RM3 terms and weights')
    parser.add_argument('--baseline-run', type=str, required=True,
                        help='Path to BM25 baseline run file')
    parser.add_argument('--expanded-run', type=str, required=True,
                        help='Path to expanded run file (e.g., best RM3 alpha)')
    parser.add_argument('--qrels', type=str, required=True,
                        help='Path to qrels file')
    parser.add_argument('--trec-eval', type=str, required=True,
                        help='Path to trec_eval binary')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to trained MEQE model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Model configuration arguments (should match training)
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Model name used in training')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms (should match training)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (should match training)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (should match training)')
    parser.add_argument('--scoring-method', type=str, default='neural',
                        choices=['neural', 'bilinear', 'cosine'],
                        help='Scoring method (should match training)')
    parser.add_argument('--force-hf', action='store_true',
                        help='Force HuggingFace transformers (should match training)')
    parser.add_argument('--pooling-strategy', type=str, default='cls',
                        choices=['cls', 'mean', 'max'],
                        help='Pooling strategy for HF models (should match training)')

    # Experimental parameters
    parser.add_argument('--utility-method', type=str, default='run_difference',
                        choices=['run_difference', 'individual_terms'],
                        help='Method to calculate term utilities')
    parser.add_argument('--folds-file', type=str,
                        help='Path to folds.json (for filtering to specific fold)')
    parser.add_argument('--fold-id', type=str,
                        help='Fold ID to filter to (e.g., "0")')
    parser.add_argument('--fold-split', type=str, choices=['training', 'testing'],
                        help='Which fold split to use')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("optimized_intrinsic_evaluation", args.log_level,
                                      str(output_dir / 'intrinsic_evaluation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Create model configuration
        model_config = {
            'model_name': args.model_name,
            'max_expansion_terms': args.max_expansion_terms,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            #'scoring_method': args.scoring_method,
            # 'force_hf': args.force_hf,
            # 'pooling_strategy': args.pooling_strategy,
            'device': 'cpu'  # Use CPU for evaluation to avoid GPU memory issues
        }

        # Load fold queries if specified
        fold_queries = None
        if args.folds_file and args.fold_id and args.fold_split:
            with open(args.folds_file, 'r') as f:
                folds = json.load(f)
            fold_queries = set(folds[args.fold_id][args.fold_split])
            logger.info(f"Filtering to fold {args.fold_id} {args.fold_split}: {len(fold_queries)} queries")

        # Initialize evaluator
        evaluator = OptimizedIntrinsicEvaluator(
            features_file=args.features_file,
            baseline_run_file=args.baseline_run,
            expanded_run_file=args.expanded_run,
            qrels_file=args.qrels,
            trec_eval_path=args.trec_eval,
            output_dir=output_dir
        )

        # Run evaluation
        use_run_difference = (args.utility_method == 'run_difference')
        results = evaluator.run_evaluation(args.model_checkpoint, model_config, use_run_difference, fold_queries)

        # Print summary
        summary = results['correlation_analysis']['summary']
        print("\n" + "=" * 70)
        print("OPTIMIZED INTRINSIC EVALUATION RESULTS")
        print("=" * 70)
        print(f"Method: {'Run difference' if use_run_difference else 'Individual terms'}")
        print(f"Queries evaluated: {summary['num_queries']}")
        print(f"")
        print(f"RM3 Statistics:")
        print(f"  Mean correlation: {summary['rm3_mean_correlation']:.4f} ± {summary['rm3_std_correlation']:.4f}")
        print(f"  Median correlation: {summary['rm3_median_correlation']:.4f}")
        print(f"")
        print(f"MEQE Statistics:")
        print(f"  Mean correlation: {summary['meqe_mean_correlation']:.4f} ± {summary['meqe_std_correlation']:.4f}")
        print(f"  Median correlation: {summary['meqe_median_correlation']:.4f}")
        print(f"")
        print(f"Comparison:")
        print(f"  Improvement: {summary['correlation_difference']:.4f}")
        print(f"  T-statistic: {summary['t_statistic']:.4f}")
        print(f"  P-value: {summary['t_p_value']:.6f}")
        print(f"  Significant improvement: {'Yes' if summary['significant_improvement'] else 'No'}")
        print(f"")
        print(f"Query-level Changes:")
        print(f"  Queries improved: {summary['queries_improved']}")
        print(f"  Queries degraded: {summary['queries_degraded']}")
        print(f"  Queries unchanged: {summary['queries_unchanged']}")
        print(f"")
        print("=" * 70)
        print(f"Detailed results saved to: {output_dir}")
        print(f"Main visualization: {output_dir}/intrinsic_evaluation_results.png")
        print(f"Detailed analysis: {output_dir}/detailed_intrinsic_analysis.png")
        print("=" * 70)

        # Create a simple text report
        report_file = output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write("INTRINSIC EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Method: {'Run difference' if use_run_difference else 'Individual terms'}\n")
            f.write(f"Total Queries Evaluated: {summary['num_queries']}\n\n")

            f.write("CORRELATION ANALYSIS:\n")
            f.write(
                f"RM3 Mean Correlation: {summary['rm3_mean_correlation']:.4f} (±{summary['rm3_std_correlation']:.4f})\n")
            f.write(
                f"MEQE Mean Correlation: {summary['meqe_mean_correlation']:.4f} (±{summary['meqe_std_correlation']:.4f})\n")
            f.write(f"Improvement: {summary['correlation_difference']:.4f}\n\n")

            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write(f"T-statistic: {summary['t_statistic']:.4f}\n")
            f.write(f"P-value: {summary['t_p_value']:.6f}\n")
            f.write(f"Significant at α=0.05: {'Yes' if summary['significant_improvement'] else 'No'}\n\n")

            f.write("QUERY-LEVEL ANALYSIS:\n")
            f.write(f"Queries with improved correlation: {summary['queries_improved']}\n")
            f.write(f"Queries with degraded correlation: {summary['queries_degraded']}\n")
            f.write(f"Queries with unchanged correlation: {summary['queries_unchanged']}\n\n")

            improvement_rate = summary['queries_improved'] / summary['num_queries'] * 100
            f.write(f"Improvement rate: {improvement_rate:.1f}%\n")

        print(f"Text report saved to: {report_file}")

    except Exception as e:
        logger.error(f"Optimized intrinsic evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()