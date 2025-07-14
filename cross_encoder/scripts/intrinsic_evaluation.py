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
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.models.reranker import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

# Import ir_measures for metric calculation
import ir_measures
from ir_measures import *

logger = logging.getLogger(__name__)


class CleanIntrinsicEvaluator:
    """
    Clean intrinsic evaluator using pre-computed files.
    """

    def __init__(self,
                 features_file: str,
                 baseline_eval_file: str,
                 output_dir: Path,
                 qrels_file: str = None,
                 index_path: str = None):
        """
        Initialize the evaluator.

        Args:
            features_file: Path to features.jsonl file
            baseline_eval_file: Path to baseline per-query eval file
            output_dir: Output directory for results
            qrels_file: Path to qrels file (for individual term retrieval)
            index_path: Path to Lucene index (for individual term retrieval)
        """
        self.features_file = Path(features_file)
        self.baseline_eval_file = Path(baseline_eval_file)
        self.output_dir = ensure_dir(output_dir)
        self.qrels_file = qrels_file
        self.index_path = index_path

        logger.info(f"CleanIntrinsicEvaluator initialized:")
        logger.info(f"  Features file: {features_file}")
        logger.info(f"  Baseline eval: {baseline_eval_file}")
        logger.info(f"  QRels file: {qrels_file}")
        logger.info(f"  Index path: {index_path}")

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
        Load per-query evaluation scores using ir_measures.

        Supports both trec_eval format files and TREC run files.
        If it's a run file, we'll need qrels to calculate the metric.
        """
        logger.info(f"Loading {metric} scores from: {eval_file}")

        # Check if this is a pre-computed trec_eval results file or a run file
        try:
            # Try to read as trec_eval results first (format: metric query_id score)
            scores = {}
            with open(eval_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0] == metric:
                        query_id = parts[1]
                        score = float(parts[2])
                        scores[query_id] = score
                    else:
                        logger.error(f"Metric {parts[0]} in file does not match metric {metric} provided. Exiting.")
                        sys.exit(1)

            if scores:
                logger.info(f"Loaded {metric} scores for {len(scores)} queries from trec_eval format")
                # Show sample scores
                sample_queries = list(scores.keys())[:3]
                logger.info(f"Sample {metric} scores:")
                for qid in sample_queries:
                    logger.info(f"  Query {qid}: {scores[qid]:.4f}")
                return scores
        except:
            pass

        # If not trec_eval format, treat as run file and compute metric using ir_measures
        logger.info(f"Treating {eval_file} as TREC run file, computing {metric} using ir_measures")

        if not self.qrels_file:
            logger.error(f"Need qrels file to compute {metric} from run file {eval_file}")
            return {}

        try:
            # Load qrels and run using ir_measures
            qrels = ir_measures.read_trec_qrels(self.qrels_file)
            run = ir_measures.read_trec_run(str(eval_file))

            # Parse the metric - convert from trec_eval format to ir_measures format
            if metric == "recall.1000":
                ir_metric = R@1000
            elif metric.startswith("recall."):
                cutoff = int(metric.split(".")[1])
                ir_metric = ir_measures.parse_measure(f"R@{cutoff}")
            elif metric.startswith("ndcg_cut_"):
                cutoff = int(metric.split("_")[-1])
                ir_metric = ir_measures.parse_measure(f"nDCG@{cutoff}")
            elif metric == "map":
                ir_metric = AP
            else:
                # Try to parse as ir_measures format
                ir_metric = ir_measures.parse_measure(metric)

            # Calculate per-query results
            scores = {}
            for result in ir_measures.iter_calc([ir_metric], qrels, run):
                scores[result.query_id] = result.value

            logger.info(f"Computed {metric} for {len(scores)} queries using ir_measures")

            # Show sample scores
            if scores:
                sample_queries = list(scores.keys())[:3]
                logger.info(f"Sample {metric} scores:")
                for qid in sample_queries:
                    logger.info(f"  Query {qid}: {scores[qid]:.4f}")

            return scores

        except Exception as e:
            logger.error(f"Failed to compute {metric} using ir_measures: {e}")
            return {}

        return {}

    def calculate_utility_from_individual_term_retrieval(self,
                                                         features: Dict[str, Dict],
                                                         baseline_scores: Dict[str, float],
                                                         top_k_retrieval: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Calculate TRUE term utilities by running individual retrieval for each expansion term.
        This implements the correct experimental methodology using ir_measures.
        """
        if not self.index_path or not self.qrels_file:
            logger.error("Individual term retrieval requires both index_path and qrels_file")
            return {}

        logger.info("Calculating individual term utilities via separate retrievals...")
        logger.info("This is the CORRECT implementation of the experimental plan")

        # Initialize Lucene components
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes

        lucene_classes = get_lucene_classes()
        FSDirectory = lucene_classes['FSDirectory']
        Paths = lucene_classes['Path']
        DirectoryReader = lucene_classes['DirectoryReader']
        IndexSearcher = lucene_classes['IndexSearcher']
        BM25Similarity = lucene_classes['BM25Similarity']
        BooleanQueryBuilder = lucene_classes['BooleanQueryBuilder']
        TermQuery = lucene_classes['TermQuery']
        Term = lucene_classes['Term']
        Occur = lucene_classes['BooleanClauseOccur']
        BoostQuery = lucene_classes['BoostQuery']

        # Open index
        directory = FSDirectory.open(Paths.get(self.index_path))
        reader = DirectoryReader.open(directory)
        reader_context = reader.getContext()
        searcher = IndexSearcher(reader_context)
        searcher.setSimilarity(BM25Similarity())

        logger.info(f"Lucene index loaded: {self.index_path}")

        # Load qrels using ir_measures
        qrels = ir_measures.read_trec_qrels(self.qrels_file)
        logger.info(f"Loaded qrels using ir_measures from: {self.qrels_file}")

        # Check query overlap
        feature_queries = set(features.keys())
        baseline_queries = set(baseline_scores.keys())
        qrels_queries = set(qrel.query_id for qrel in qrels)

        all_overlap = feature_queries & baseline_queries & qrels_queries
        logger.info(f"Common queries across all datasets: {len(all_overlap)}")

        if len(all_overlap) == 0:
            logger.error("No queries found in all datasets!")
            return {}

        term_utilities = {}
        queries_processed = 0

        # Create temporary files for ir_measures evaluation
        import tempfile
        import os

        try:
            for query_id, query_data in tqdm(features.items(), desc="Processing queries"):
                if query_id not in baseline_scores or query_id not in qrels_queries:
                    continue

                queries_processed += 1
                baseline_score = baseline_scores[query_id]
                query_text = query_data['query_text']
                term_features = query_data['term_features']

                # Debug for first few queries
                if queries_processed <= 3:
                    logger.info(f"Query {query_id}: '{query_text}'")
                    logger.info(f"  Baseline nDCG@20: {baseline_score:.4f}")
                    logger.info(f"  Processing {len(term_features)} expansion terms")

                query_utilities = {}

                # Process each expansion term individually
                for term_idx, (term, term_data) in enumerate(term_features.items()):
                    # Get the original (unstemmed) term for query expansion
                    original_term = term_data.get('original_term', term)

                    try:
                        # Build Lucene query for expanded query
                        builder = BooleanQueryBuilder()

                        # Add original query terms
                        original_terms = self._tokenize_query_lucene(query_text, searcher)
                        for orig_term in original_terms:
                            term_query = TermQuery(Term("contents", orig_term))
                            builder.add(term_query, Occur.SHOULD)

                        # Add expansion term with weight 0.5
                        expansion_term_query = TermQuery(Term("contents", original_term.lower()))
                        boosted_expansion = BoostQuery(expansion_term_query, 0.5)
                        builder.add(boosted_expansion, Occur.SHOULD)

                        final_query = builder.build()

                        # Execute search
                        top_docs = searcher.search(final_query, top_k_retrieval)
                        # if len(top_docs.scoreDocs) == 0:
                        #     logger.warning(
                        #         f"    [DIAGNOSTIC] Term '{original_term}' resulted in 0 documents from Lucene.")
                        # else:
                        #     logger.warning(
                        #         f"    [DIAGNOSTIC] Term '{original_term}' resulted in '{len(top_docs.scoreDocs)}' documents from Lucene.")


                        # Create temporary run file for this expanded query
                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as temp_run:
                            for rank, score_doc in enumerate(top_docs.scoreDocs, 1):
                                doc_id = reader.storedFields().document(score_doc.doc).get("id")
                                if doc_id:
                                    temp_run.write(f'{query_id} Q0 {doc_id} {rank} {score_doc.score:.6f} expanded\n')
                                    # print(f'{query_id} Q0 {doc_id} {rank} {score_doc.score:.6f} expanded\n')
                            temp_run_path = temp_run.name


                        # In your calculate_utility_from_individual_term_retrieval function...

                        try:
                            # Load the run using ir_measures and immediately convert to a list
                            # This prevents the generator from being exhausted.
                            expanded_run = list(ir_measures.read_trec_run(temp_run_path))

                            # Default the recall to 0.0. This handles empty run files gracefully.
                            expanded_recall = 0.0

                            # Only proceed if the Lucene search actually returned results
                            if expanded_run:
                                # Use iter_calc to get a dictionary of {query_id: score}
                                # This is the correct way to get per-query results.
                                per_query_scores = {m.query_id: m.value for m in
                                                    ir_measures.iter_calc([nDCG@20], qrels, expanded_run)}

                                # Get the recall for the specific query we are currently processing
                                expanded_recall = per_query_scores.get(query_id, 0.0)

                            # Calculate the final utility for the term
                            term_utility = expanded_recall - baseline_score
                            query_utilities[term] = term_utility

                            # Debug for first query, first few terms
                            if queries_processed == 1 and term_idx < 3:
                                logger.info(
                                    f"    Term '{original_term}' -> nDCG@20: {expanded_recall:.4f}, utility: {term_utility:.4f}")

                        finally:
                            # Clean up temporary file
                            os.unlink(temp_run_path)

                    except Exception as e:
                        logger.warning(f"Error processing term '{original_term}' for query {query_id}: {e}")
                        query_utilities[term] = 0.0

                term_utilities[query_id] = query_utilities

                # Log progress for first few queries
                if queries_processed <= 3:
                    avg_utility = np.mean(list(query_utilities.values())) if query_utilities else 0.0
                    logger.info(f"  Average term utility: {avg_utility:.4f}")

        finally:
            # Clean up Lucene resources
            reader.close()
            directory.close()

        logger.info(f"Individual term utility calculation complete:")
        logger.info(f"  Queries processed: {queries_processed}")
        logger.info(f"  Final utility scores: {len(term_utilities)} queries")

        return term_utilities

    def _load_qrels_for_evaluation(self) -> Dict[str, Dict[str, int]]:
        """Load qrels using ir_measures (preferred) with fallback to manual parsing."""
        if not self.qrels_file:
            logger.warning("No qrels file provided - cannot calculate individual term utilities")
            return {}

        qrels_file = Path(self.qrels_file)

        try:
            # Try using ir_measures first (preferred method)
            logger.info(f"Loading qrels using ir_measures from: {qrels_file}")
            qrels_ir = ir_measures.read_trec_qrels(str(qrels_file))

            # Convert to the expected dictionary format
            qrels = {}
            for qrel in qrels_ir:
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = {}
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

            logger.info(f"Loaded qrels for {len(qrels)} queries using ir_measures")
            return qrels

        except Exception as e:
            logger.warning(f"ir_measures loading failed ({e}), trying manual parsing...")

            # Fallback to manual parsing
            qrels = {}
            try:
                with open(qrels_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            query_id = parts[0]
                            doc_id = parts[2]
                            relevance = int(parts[3])

                            if query_id not in qrels:
                                qrels[query_id] = {}
                            qrels[query_id][doc_id] = relevance

                logger.info(f"Loaded qrels for {len(qrels)} queries using manual parsing")
                return qrels

            except Exception as file_error:
                logger.error(f"Could not load qrels from {qrels_file}: {file_error}")
                return {}

    def _tokenize_query_lucene(self, query_text: str, searcher) -> List[str]:
        """Tokenize query using Lucene's analyzer."""
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes

        lucene_classes = get_lucene_classes()
        EnglishAnalyzer = lucene_classes['EnglishAnalyzer']
        CharTermAttribute = lucene_classes['CharTermAttribute']

        analyzer = EnglishAnalyzer()
        tokens = []

        try:
            token_stream = analyzer.tokenStream("contents", query_text)
            char_term_attr = token_stream.addAttribute(CharTermAttribute)
            token_stream.reset()

            while token_stream.incrementToken():
                tokens.append(char_term_attr.toString())

            token_stream.close()
        except Exception as e:
            logger.warning(f"Error tokenizing query '{query_text}': {e}")
            # Fallback to simple splitting
            tokens = query_text.lower().split()

        return tokens

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
        This version explicitly handles cases of constant utility to avoid NaN correlations.
        """
        logger.info("Calculating rank correlations...")
        logger.info(f"Input data sizes:")
        logger.info(f"  Utility scores: {len(utility_scores)} queries")
        logger.info(f"  Features: {len(features)} queries")
        logger.info(f"  MEQE weights: {len(meqe_weights)} queries")

        # --- Find common queries across all data sources ---
        common_queries = set(utility_scores.keys()) & set(features.keys()) & set(meqe_weights.keys())
        logger.info(f"Common queries across all datasets: {len(common_queries)}")

        if not common_queries:
            logger.error("No common queries found!")
            return {'error': 'No common queries to process.'}

        query_results = {}
        queries_with_sufficient_terms = 0

        # --- Process each query individually ---
        for query_id in sorted(list(common_queries)):  # Sorting for consistent order
            utility_dict = utility_scores.get(query_id, {})
            term_features = features.get(query_id, {}).get('term_features', {})
            meqe_dict = meqe_weights.get(query_id, {})

            # Find common terms for this specific query
            common_terms = set(utility_dict.keys()) & set(term_features.keys()) & set(meqe_dict.keys())

            if len(common_terms) < 3:  # Spearman correlation needs at least a few data points
                continue

            queries_with_sufficient_terms += 1

            # --- Create aligned lists of values for correlation ---
            utility_values = np.array([utility_dict[term] for term in common_terms], dtype=float)
            rm3_values = np.array([term_features[term]['rm_weight'] for term in common_terms], dtype=float)
            meqe_values = np.array([meqe_dict[term] for term in common_terms], dtype=float)

            # --- Key Fix: Check for constant utility values before calculating ---
            # The correlation is undefined if one of the inputs has no variance.
            if np.all(utility_values == utility_values[0]):
                logger.warning(
                    f"Query {query_id}: Skipped. All utility scores are constant ({utility_values[0]:.4f}), so correlation is undefined.")
                # Record as NaN to indicate it couldn't be calculated
                rho_rm3, p_rm3 = np.nan, np.nan
                rho_meqe, p_meqe = np.nan, np.nan
            else:
                # Proceed with calculation only if utility varies
                rho_rm3, p_rm3 = spearmanr(utility_values, rm3_values)
                rho_meqe, p_meqe = spearmanr(utility_values, meqe_values)

            query_results[query_id] = {
                'rm3_correlation': rho_rm3,
                'meqe_correlation': rho_meqe,
                'rm3_p_value': p_rm3,
                'meqe_p_value': p_meqe,
                'num_terms': len(common_terms),
                'utility_variance': np.var(utility_values)
            }

        # --- Aggregate results and perform statistical tests ---
        logger.info(f"Queries with sufficient terms: {queries_with_sufficient_terms}")

        # Create paired lists for T-test, excluding any queries where correlation could not be calculated.
        valid_pairs = [(res['rm3_correlation'], res['meqe_correlation']) for res in query_results.values()
                       if not np.isnan(res['rm3_correlation']) and not np.isnan(res['meqe_correlation'])]

        logger.info(f"Valid correlation pairs for comparison: {len(valid_pairs)}")

        if len(valid_pairs) < 2:
            logger.error("Insufficient valid data for a paired t-test.")
            return {'summary': {'error': 'Insufficient valid data'}, 'per_query_results': query_results}

        # Unzip the pairs for testing
        paired_rm3, paired_meqe = zip(*valid_pairs)

        # Perform paired t-test
        t_stat, t_p_value = ttest_rel(paired_meqe, paired_rm3)

        final_results = {
            'summary': {
                'rm3_mean_correlation': np.mean(paired_rm3),
                'meqe_mean_correlation': np.mean(paired_meqe),
                'correlation_difference': np.mean(paired_meqe) - np.mean(paired_rm3),
                'num_queries_analyzed': len(valid_pairs),
                't_statistic': t_stat,
                't_p_value': t_p_value,
                'significant_improvement': t_p_value < 0.05 and np.mean(paired_meqe) > np.mean(paired_rm3),
                'queries_improved': int(np.sum(np.array(paired_meqe) > np.array(paired_rm3))),
            },
            'per_query_results': query_results
        }

        logger.info("Correlation analysis complete:")
        logger.info(f"  RM3 mean correlation: {final_results['summary']['rm3_mean_correlation']:.4f}")
        logger.info(f"  MEQE mean correlation: {final_results['summary']['meqe_mean_correlation']:.4f}")
        logger.info(f"  Queries improved: {final_results['summary']['queries_improved']} / {len(valid_pairs)}")
        logger.info(f"  P-value: {final_results['summary']['t_p_value']:.4f}")
        logger.info(f"  Significant improvement: {final_results['summary']['significant_improvement']}")

        return final_results

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
                       metric: str = "recall.1000",
                       use_individual_retrieval: bool = False) -> Dict[str, Any]:
        """
        Run the complete intrinsic evaluation pipeline.

        Args:
            model_checkpoint_path: Path to trained MEQE model checkpoint
            model_config: Configuration for model creation
            metric: Evaluation metric to use (default: recall.1000)
            use_individual_retrieval: If True, use individual term retrieval method

        Returns:
            Complete evaluation results
        """
        logger.info("Starting intrinsic evaluation pipeline...")

        with TimedOperation(logger, "Complete Intrinsic Evaluation"):
            # Step 1: Load all data
            with TimedOperation(logger, "Loading features"):
                features = self.load_features()

            with TimedOperation(logger,"Loading baseline scores"):
                baseline_scores = self.load_eval_scores(self.baseline_eval_file, metric)

            # Step 2: Calculate term utilities (ground truth)
            if use_individual_retrieval and self.index_path and self.qrels_file:
                with TimedOperation(logger,"Calculating individual term utilities"):
                    utility_scores = self.calculate_utility_from_individual_term_retrieval(
                        features, baseline_scores
                    )
            else:
                with TimedOperation(logger,"Loading expanded scores"):
                    expanded_scores = self.load_eval_scores(self.expanded_eval_file, metric)

                with TimedOperation(logger,"Calculating term utilities from score difference"):
                    utility_scores = self.calculate_utility_from_score_difference(
                        features, baseline_scores, expanded_scores
                    )

            if not utility_scores:
                logger.error("No utility scores calculated - cannot proceed")
                return {'error': 'No utility scores calculated'}

            # Step 3: Get MEQE model weights
            with TimedOperation(logger,"Evaluating MEQE model"):
                meqe_weights = self.evaluate_meqe_model(
                    features, model_checkpoint_path, model_config
                )

            # Step 4: Calculate rank correlations
            with TimedOperation(logger,"Calculating rank correlations"):
                correlation_results = self.calculate_rank_correlations(
                    utility_scores, features, meqe_weights
                )

            # Step 5: Create visualizations
            with TimedOperation(logger,"Creating visualizations"):
                self.create_visualizations(correlation_results)

            # Step 6: Save detailed results
            with TimedOperation(logger,"Saving results"):
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

    # Optional arguments for individual retrieval
    parser.add_argument(
        "--index-path",
        type=str,
        help="Path to Lucene index (required for individual term retrieval)"
    )
    parser.add_argument(
        "--qrels-file",
        type=str,
        help="Path to qrels file (required for individual term retrieval)"
    )
    parser.add_argument(
        "--use-individual-retrieval",
        action="store_true",
        help="Use individual term retrieval method (requires index_path and qrels_file)"
    )
    parser.add_argument(
        "--lucene-path",
        type=str,
        help="Path to Lucene JAR files (required if using individual retrieval)"
    )

    # Model configuration arguments (should match training)
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model name used in training"
    )
    parser.add_argument(
        "--max-expansion-terms",
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
        default="recall.1000",
        help="Evaluation metric to use (default: recall.1000)"
    )
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    setup_experiment_logging(
        experiment_name="intrinsic_evaluation",
        log_level=args.log_level,
        log_file=str(Path(args.output_dir) / 'intrinsic_evaluation.log')
    )

    # Get logger instance
    logger = logging.getLogger(__name__)

    # Initialize Lucene if using individual retrieval
    if args.use_individual_retrieval:
        if not args.lucene_path:
            logger.error("--lucene_path is required when using individual retrieval")
            sys.exit(1)
        if not args.index_path or not args.qrels_file:
            logger.error("--index_path and --qrels_file are required when using individual retrieval")
            sys.exit(1)

        from cross_encoder.src.utils.lucene_utils import initialize_lucene
        if not initialize_lucene(args.lucene_path):
            logger.error("Failed to initialize Lucene")
            sys.exit(1)


    logger = setup_experiment_logging(
        experiment_name="intrinsic_evaluation",
        log_level=args.log_level,
        log_file=str(Path(args.output_dir) / 'intrinsic_evaluation.log')
    )

    # Log experiment info
    log_experiment_info(
        logger,
        experiment_type='intrinsic_evaluation',
        features_file=args.features_file,
        baseline_eval_file=args.baseline_eval_file,
        model_checkpoint=args.model_checkpoint,
        metric=args.metric,
        use_individual_retrieval=args.use_individual_retrieval,
        index_path=args.index_path,
        qrels_file=args.qrels_file,
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        scoring_method=args.scoring_method
    )

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
        output_dir=Path(args.output_dir),
        qrels_file=args.qrels_file,
        index_path=args.index_path
    )

    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            model_checkpoint_path=args.model_checkpoint,
            model_config=model_config,
            metric=args.metric,
            use_individual_retrieval=args.use_individual_retrieval
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