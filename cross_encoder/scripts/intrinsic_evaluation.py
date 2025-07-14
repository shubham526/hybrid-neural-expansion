#!/usr/bin/env python3
"""
Intrinsic Evaluation: Term Ranking Quality via Incremental Build-Up

This script evaluates whether the term ordering from MEQE leads to better
query performance compared to the RM3 ordering when terms are added incrementally.
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import json
from scipy.stats import ttest_rel
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

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


class IncrementalBuildUpEvaluator:
    """
    Evaluates term rankings by incrementally building up a query and measuring performance.
    """

    def __init__(self,
                 features_file: str,
                 output_dir: Path,
                 qrels_file: str,
                 index_path: str):
        """
        Initialize the evaluator.
        """
        self.features_file = Path(features_file)
        self.output_dir = ensure_dir(output_dir)
        self.qrels_file = qrels_file
        self.index_path = index_path
        self.searcher = None  # To be initialized later
        self.reader = None
        self.qrels = None

        logger.info(f"IncrementalBuildUpEvaluator initialized:")
        logger.info(f"  Features file: {features_file}")
        logger.info(f"  QRels file: {qrels_file}")
        logger.info(f"  Index path: {index_path}")

    def _initialize_searcher(self):
        """Initializes the Lucene searcher and loads qrels."""
        if self.searcher:
            return

        logger.info("Initializing Lucene searcher and loading qrels...")
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        lucene_classes = get_lucene_classes()
        FSDirectory = lucene_classes['FSDirectory']
        Paths = lucene_classes['Path']
        DirectoryReader = lucene_classes['DirectoryReader']
        IndexSearcher = lucene_classes['IndexSearcher']
        BM25Similarity = lucene_classes['BM25Similarity']

        directory = FSDirectory.open(Paths.get(self.index_path))
        self.reader = DirectoryReader.open(directory)
        reader_context = self.reader.getContext()
        self.searcher = IndexSearcher(reader_context)
        self.searcher.setSimilarity(BM25Similarity())

        # directory = FSDirectory.open(Paths.get(self.index_path))
        # self.reader = DirectoryReader.open(directory)
        # self.searcher = IndexSearcher(self.reader)
        # self.searcher.setSimilarity(BM25Similarity())
        logger.info(f"Lucene index loaded: {self.index_path}")

        self.qrels = ir_measures.read_trec_qrels(self.qrels_file)
        logger.info(f"Loaded qrels for {len(set(q.query_id for q in self.qrels))} queries.")

    def _tokenize_query_lucene(self, query_text: str) -> List[str]:
        """Tokenize query using Lucene's analyzer for consistency."""
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        lucene_classes = get_lucene_classes()
        EnglishAnalyzer = lucene_classes['EnglishAnalyzer']
        CharTermAttribute = lucene_classes['CharTermAttribute']

        analyzer = EnglishAnalyzer()
        tokens = []
        try:
            stream = analyzer.tokenStream("contents", query_text)
            term_attr = stream.addAttribute(CharTermAttribute)
            stream.reset()
            while stream.incrementToken():
                tokens.append(term_attr.toString())
            stream.close()
        except Exception:
            tokens = query_text.lower().split()
        return tokens

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

    def evaluate_meqe_model(self, features: Dict[str, Dict], model_checkpoint_path: str,
                            model_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate MEQE model to get hybrid weights."""
        logger.info("Evaluating MEQE model weights...")
        model = create_neural_reranker(**model_config)
        import torch
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        alpha, beta, _ = model.get_learned_weights()
        logger.info(f"Model weights: α={alpha:.4f}, β={beta:.4f}")

        meqe_weights = {}
        with torch.no_grad():
            for query_id, query_data in features.items():
                term_weights = {
                    term: alpha * data['rm_weight'] + beta * data['semantic_score']
                    for term, data in query_data['term_features'].items()
                }
                meqe_weights[query_id] = term_weights
        logger.info(f"Calculated MEQE weights for {len(meqe_weights)} queries")
        return meqe_weights

    def _run_build_up_curve(self, query_id: str, original_query: str, ranked_terms: List[str],
                            metric: "ir_measures.Measure") -> List[float]:
        """Runs the incremental build-up for a single ranked list of terms."""
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        lucene_classes = get_lucene_classes()
        BooleanQueryBuilder = lucene_classes['BooleanQueryBuilder']
        TermQuery = lucene_classes['TermQuery']
        Term = lucene_classes['Term']
        Occur = lucene_classes['BooleanClauseOccur']

        performance_curve = []
        original_query_terms = self._tokenize_query_lucene(original_query)
        current_terms = list(original_query_terms)

        # **FIX**: Filter out expansion terms already in the original query
        original_terms_set = set(original_query_terms)
        filtered_ranked_terms = [term for term in ranked_terms if term not in original_terms_set]
        print(f"{len(filtered_ranked_terms)} terms remaining")
        print(filtered_ranked_terms)

        for i in range(len(filtered_ranked_terms) + 1):
            # **FIX**: Preserve order while handling duplicates for query building
            seen_terms = set()
            unique_query_terms = []
            for term in current_terms:
                if term not in seen_terms:
                    unique_query_terms.append(term)
                    seen_terms.add(term)

            builder = BooleanQueryBuilder()
            for term in unique_query_terms:
                builder.add(TermQuery(Term("contents", term)), Occur.SHOULD)
            final_query = builder.build()

            # Execute search
            top_docs = self.searcher.search(final_query, 1000)  # Only need top 20 for nDCG@20

            # Create a temporary run file for this single query
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as temp_run:
                for rank, score_doc in enumerate(top_docs.scoreDocs, 1):
                    doc_id = self.reader.storedFields().document(score_doc.doc).get("id")
                    if doc_id:
                        temp_run.write(f'{query_id} Q0 {doc_id} {rank} {score_doc.score:.6f} build_up\n')
                temp_run_path = temp_run.name

            # Calculate metric and clean up
            try:
                if os.path.getsize(temp_run_path) > 0:
                    run = list(ir_measures.read_trec_run(temp_run_path))  # Use list to be safe
                    score = ir_measures.calc_aggregate([metric], self.qrels, run).get(metric, 0.0)
                else:
                    score = 0.0  # No results returned
                performance_curve.append(score)
            finally:
                os.unlink(temp_run_path)

            # Add the next term for the next iteration
            if i < len(filtered_ranked_terms):
                current_terms.append(filtered_ranked_terms[i])

        return performance_curve

    def calculate_incremental_build_up_performance(self, features: Dict[str, Dict],
                                                   meqe_weights: Dict[str, Dict[str, float]], metric_str: str) -> Dict[
        str, Any]:
        """Main function to run the incremental build-up experiment."""
        self._initialize_searcher()
        logger.info("Starting incremental build-up experiment...")

        metric = ir_measures.parse_measure(metric_str)
        all_results = {}

        common_queries = set(features.keys()) & set(meqe_weights.keys())

        for query_id in tqdm(sorted(list(common_queries)), desc="Processing Queries"):
            query_data = features[query_id]
            original_query = query_data['query_text']

            # Get term rankings for both methods
            rm3_ranked_terms = sorted(query_data['term_features'].keys(),
                                      key=lambda t: query_data['term_features'][t]['rm_weight'], reverse=True)
            meqe_ranked_terms = sorted(meqe_weights[query_id].keys(), key=lambda t: meqe_weights[query_id][t],
                                       reverse=True)

            # Generate performance curves
            rm3_curve = self._run_build_up_curve(query_id, original_query, rm3_ranked_terms, metric)
            meqe_curve = self._run_build_up_curve(query_id, original_query, meqe_ranked_terms, metric)

            # Calculate metrics for this query
            # Ensure curves are of same length for AUC calculation, pad if necessary
            max_len = max(len(rm3_curve), len(meqe_curve))
            rm3_curve.extend([rm3_curve[-1]] * (max_len - len(rm3_curve)))
            meqe_curve.extend([meqe_curve[-1]] * (max_len - len(meqe_curve)))
            steps = np.arange(max_len)

            all_results[query_id] = {
                'rm3_curve': rm3_curve,
                'meqe_curve': meqe_curve,
                'rm3_auc': auc(steps, rm3_curve),
                'meqe_auc': auc(steps, meqe_curve),
                'rm3_perf_at_5': rm3_curve[5] if len(rm3_curve) > 5 else rm3_curve[-1],
                'meqe_perf_at_5': meqe_curve[5] if len(meqe_curve) > 5 else meqe_curve[-1],
                'rm3_perf_at_10': rm3_curve[10] if len(rm3_curve) > 10 else rm3_curve[-1],
                'meqe_perf_at_10': meqe_curve[10] if len(meqe_curve) > 10 else meqe_curve[-1],
                'rm3_peak_perf': max(rm3_curve),
                'meqe_peak_perf': max(meqe_curve)
            }

        # --- Aggregate final results and perform t-tests ---
        num_queries = len(all_results)
        if num_queries == 0:
            logger.error("No queries were processed successfully.")
            return {"error": "No queries processed."}

        summary = {}
        for key in ['auc', 'perf_at_5', 'perf_at_10', 'peak_perf']:
            rm3_scores = [res[f'rm3_{key}'] for res in all_results.values()]
            meqe_scores = [res[f'meqe_{key}'] for res in all_results.values()]

            t_stat, p_value = ttest_rel(meqe_scores, rm3_scores)

            summary[key] = {
                'rm3_mean': np.mean(rm3_scores),
                'meqe_mean': np.mean(meqe_scores),
                'mean_diff': np.mean(meqe_scores) - np.mean(rm3_scores),
                'queries_improved': int(np.sum(np.array(meqe_scores) > np.array(rm3_scores))),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_improvement': p_value < 0.05 and np.mean(meqe_scores) > np.mean(rm3_scores)
            }

        final_report = {
            'summary': summary,
            'per_query_results': all_results
        }

        # Log summary
        for key, data in summary.items():
            logger.info(f"--- {key.upper()} Results ---")
            logger.info(f"  RM3 Mean: {data['rm3_mean']:.4f}")
            logger.info(f"  MEQE Mean: {data['meqe_mean']:.4f}")
            logger.info(f"  Queries Improved: {data['queries_improved']} / {num_queries}")
            logger.info(
                f"  P-value: {data['p_value']:.4f} ({'Significant' if data['significant_improvement'] else 'Not Significant'})")

        return final_report

    def create_build_up_visualizations(self, results: Dict[str, Any], metric_str: str) -> None:
        """Create visualizations for the incremental build-up experiment."""
        logger.info("Creating build-up visualizations...")

        if 'error' in results or 'summary' not in results:
            logger.error("Cannot create visualizations due to error or missing data in results.")
            return

        per_query = results['per_query_results']

        plt.style.use('default')
        sns.set_palette("colorblind")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Incremental Build-Up Evaluation ({metric_str})', fontsize=18, fontweight='bold')

        # 1. Average Performance Curve
        ax1 = axes[0, 0]
        max_len = max(len(res['rm3_curve']) for res in per_query.values())

        # Pad curves to max_len for averaging
        rm3_curves_padded = [np.pad(res['rm3_curve'], (0, max_len - len(res['rm3_curve'])), 'edge') for res in
                             per_query.values()]
        meqe_curves_padded = [np.pad(res['meqe_curve'], (0, max_len - len(res['meqe_curve'])), 'edge') for res in
                              per_query.values()]

        avg_rm3_curve = np.mean(rm3_curves_padded, axis=0)
        avg_meqe_curve = np.mean(meqe_curves_padded, axis=0)
        steps = np.arange(len(avg_rm3_curve))

        ax1.plot(steps, avg_rm3_curve, marker='o', linestyle='--', label='RM3 Order', markersize=4)
        ax1.plot(steps, avg_meqe_curve, marker='s', linestyle='-', label='MEQE Order', markersize=4)
        ax1.set_xlabel("Number of Expansion Terms Added")
        ax1.set_ylabel(f"Average {metric_str}")
        ax1.set_title("Average Query Performance vs. Terms Added")
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        # 2. Boxplot of Area Under the Curve (AUC)
        ax2 = axes[0, 1]
        auc_data = [
            [res['rm3_auc'] for res in per_query.values()],
            [res['meqe_auc'] for res in per_query.values()]
        ]
        bplot1 = ax2.boxplot(auc_data, labels=['RM3', 'MEQE'], patch_artist=True)
        ax2.set_ylabel("Area Under Curve (AUC)")
        ax2.set_title("Distribution of AUC per Query")
        ax2.grid(True, alpha=0.4)

        # 3. Boxplot of Performance at 5 Terms
        ax3 = axes[1, 0]
        perf5_data = [
            [res['rm3_perf_at_5'] for res in per_query.values()],
            [res['meqe_perf_at_5'] for res in per_query.values()]
        ]
        bplot2 = ax3.boxplot(perf5_data, labels=['RM3', 'MEQE'], patch_artist=True)
        ax3.set_ylabel(f"{metric_str} after 5 terms")
        ax3.set_title("Performance after Adding Top 5 Terms")
        ax3.grid(True, alpha=0.4)

        # 4. Boxplot of Peak Performance
        ax4 = axes[1, 1]
        peak_perf_data = [
            [res['rm3_peak_perf'] for res in per_query.values()],
            [res['meqe_peak_perf'] for res in per_query.values()]
        ]
        bplot3 = ax4.boxplot(peak_perf_data, labels=['RM3', 'MEQE'], patch_artist=True)
        ax4.set_ylabel(f"Peak {metric_str} Achieved")
        ax4.set_title("Peak Performance per Query")
        ax4.grid(True, alpha=0.4)

        # Add colors to boxplots
        colors = ['lightblue', 'lightgreen']
        for bplot in (bplot1, bplot2, bplot3):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = self.output_dir / 'build_up_evaluation_results.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Build-up visualization saved to: {plot_path}")

    def run_evaluation(self, model_checkpoint_path: str, model_config: Dict[str, Any], metric: str):
        """Main entry point to run the evaluation."""
        logger.info("Starting incremental build-up evaluation pipeline...")

        with TimedOperation(logger, "Complete Incremental Build-Up Evaluation"):
            # Step 1: Load features
            with TimedOperation(logger, "Loading features"):
                features = self.load_features()

            # Step 2: Get MEQE weights
            with TimedOperation(logger, "Evaluating MEQE model"):
                meqe_weights = self.evaluate_meqe_model(features, model_checkpoint_path, model_config)

            # Step 3: Run the build-up experiment
            with TimedOperation(logger, "Calculating Incremental Build-Up Performance"):
                results = self.calculate_incremental_build_up_performance(features, meqe_weights, metric)

            # Step 4: Create visualizations
            with TimedOperation(logger, "Creating visualizations"):
                self.create_build_up_visualizations(results, metric)

            # Step 5: Save detailed results
            with TimedOperation(logger, "Saving results"):
                results_file = self.output_dir / 'build_up_evaluation_results.json'
                save_json(results, results_file)
                logger.info(f"Detailed results saved to: {results_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Run Incremental Build-Up evaluation of MEQE.")
    parser.add_argument("--features-file", type=str, required=True, help="Path to features.jsonl file.")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to trained MEQE model checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results.")
    parser.add_argument("--index-path", type=str, required=True, help="Path to Lucene index.")
    parser.add_argument("--qrels-file", type=str, required=True, help="Path to qrels file.")
    parser.add_argument("--lucene-path", type=str, required=True, help="Path to Lucene JAR files.")
    parser.add_argument("--metric", type=str, default="nDCG@20",
                        help="Metric for evaluation (e.g., nDCG@20, MAP, R@100).")

    # Model config args
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max-expansion-terms", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scoring-method", type=str, default="neural", choices=['neural', 'bilinear', 'cosine'])
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--pooling-strategy", type=str, default="cls", choices=['cls', 'mean', 'max'])
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    setup_experiment_logging(
        experiment_name="incremental_build_up",
        log_level=args.log_level,
        log_file=str(Path(args.output_dir) / 'build_up_evaluation.log')
    )

    # Initialize Lucene
    from cross_encoder.src.utils.lucene_utils import initialize_lucene
    if not initialize_lucene(args.lucene_path):
        logger.error("Failed to initialize Lucene.")
        sys.exit(1)

    model_config = {
        'model_name': args.model_name,
        'max_expansion_terms': args.max_expansion_terms,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'scoring_method': args.scoring_method,
        'force_hf': args.force_hf,
        'pooling_strategy': args.pooling_strategy,
        'device': 'cpu'
    }

    evaluator = IncrementalBuildUpEvaluator(
        features_file=args.features_file,
        output_dir=Path(args.output_dir),
        qrels_file=args.qrels_file,
        index_path=args.index_path
    )

    try:
        results = evaluator.run_evaluation(
            model_checkpoint_path=args.model_checkpoint,
            model_config=model_config,
            metric=args.metric
        )
        if 'error' not in results:
            logger.info("Incremental Build-Up evaluation completed successfully!")
        else:
            logger.error(f"Evaluation failed: {results['error']}")
            return 1
    except Exception as e:
        logger.error(f"Evaluation failed with an unhandled exception: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())