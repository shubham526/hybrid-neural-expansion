#!/usr/bin/env python3
"""
End-to-End Dense Retrieval Evaluation Script

This script provides comprehensive evaluation of dense retrieval systems using
trained bi-encoder models. It can evaluate single models or compare multiple
approaches including baselines.

Usage:
    # Evaluate single model
    python evaluate_retrieval.py \
        --model-path ./models/bi_encoder \
        --index-dir ./indices/documents \
        --queries-file queries.jsonl \
        --qrels-file qrels.txt \
        --output-dir ./results

    # Compare multiple models
    python evaluate_retrieval.py \
        --models-config models_config.json \
        --index-dir ./indices/documents \
        --dataset msmarco-passage/trec-dl-2019 \
        --output-dir ./comparison_results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from bi_encoder.src.models.bi_encoder import create_hybrid_bi_encoder
from bi_encoder.src.evaluation.dense_retrieval import (
    create_dense_retriever,
    create_dense_retrieval_evaluator,
    create_end_to_end_evaluator
)
from bi_encoder.src.utils.indexing import create_dense_index
from cross_encoder.src.utils.file_utils import ensure_dir, load_json, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, TimedOperation, log_experiment_info

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Comprehensive retrieval evaluation system.
    """

    def __init__(self,
                 output_dir: Path,
                 metrics: List[str] = None,
                 top_k: int = 1000):
        """
        Initialize retrieval evaluator.

        Args:
            output_dir: Directory to save results
            metrics: List of metrics to compute
            top_k: Number of documents to retrieve per query
        """
        self.output_dir = ensure_dir(output_dir)
        self.top_k = top_k

        if metrics is None:
            self.metrics = ['recall_10', 'recall_100', 'recall_1000',
                            'ndcg_cut_10', 'ndcg_cut_100', 'map']
        else:
            self.metrics = metrics

        self.evaluator = create_dense_retrieval_evaluator(self.metrics)

        # Storage for results
        self.models = {}
        self.retrievers = {}
        self.results = {}

        logger.info(f"RetrievalEvaluator initialized with metrics: {self.metrics}")

    def load_model(self, model_name: str, model_path: Path) -> None:
        """
        Load a bi-encoder model.

        Args:
            model_name: Name identifier for the model
            model_path: Path to model directory
        """
        logger.info(f"Loading model '{model_name}' from {model_path}")

        # Load model info
        model_info_file = model_path / 'model_info.json'
        if not model_info_file.exists():
            raise FileNotFoundError(f"Model info not found: {model_info_file}")

        model_info = load_json(model_info_file)

        # Create model
        model = create_hybrid_bi_encoder(
            model_name=model_info['model_name'],
            max_expansion_terms=model_info['max_expansion_terms'],
            expansion_weight=model_info.get('expansion_weight', 0.3),
            similarity_function=model_info.get('similarity_function', 'cosine'),
            force_hf=model_info.get('force_hf', False),
            pooling_strategy=model_info.get('pooling_strategy', 'cls')
        )

        # Load trained weights
        best_model_file = model_path / 'best_model.pt'
        final_model_file = model_path / 'final_model.pt'

        if best_model_file.exists():
            model.load_state_dict(torch.load(best_model_file, map_location=model.device))
            logger.info(f"Loaded best model weights for '{model_name}'")
        elif final_model_file.exists():
            model.load_state_dict(torch.load(final_model_file, map_location=model.device))
            logger.info(f"Loaded final model weights for '{model_name}'")
        else:
            logger.warning(f"No trained weights found for '{model_name}', using random initialization")

        model.eval()
        self.models[model_name] = {
            'model': model,
            'info': model_info
        }

        logger.info(f"Model '{model_name}' loaded successfully")

    def setup_retriever(self, model_name: str, index_dir: Path) -> None:
        """
        Setup dense retriever for a model.

        Args:
            model_name: Name of the model
            index_dir: Directory containing the dense index
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")

        model = self.models[model_name]['model']

        # Create retriever
        retriever = create_dense_retriever(
            bi_encoder_model=model,
            use_faiss=True,
            faiss_index_type="IndexFlatIP"
        )

        # Load index
        logger.info(f"Loading index for '{model_name}' from {index_dir}")
        retriever.load_index(index_dir)

        self.retrievers[model_name] = retriever
        logger.info(f"Retriever '{model_name}' setup complete")

    def load_queries_and_qrels(self,
                               queries_file: Path = None,
                               qrels_file: Path = None,
                               dataset_name: str = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Load queries and qrels from files or ir_datasets.

        Args:
            queries_file: Path to queries file (JSONL format)
            qrels_file: Path to qrels file (TREC format)
            dataset_name: ir_datasets dataset name

        Returns:
            Tuple of (queries, qrels)
        """
        queries = {}
        qrels = {}

        if dataset_name:
            logger.info(f"Loading queries and qrels from dataset: {dataset_name}")
            dataset = ir_datasets.load(dataset_name)

            # Load queries
            for query in dataset.queries_iter():
                if hasattr(query, 'text'):
                    queries[query.query_id] = query.text
                elif hasattr(query, 'title'):
                    queries[query.query_id] = query.title

            # Load qrels
            qrels_dict = defaultdict(dict)
            for qrel in dataset.qrels_iter():
                qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance
            qrels = dict(qrels_dict)

        else:
            # Load from files
            if queries_file:
                logger.info(f"Loading queries from: {queries_file}")
                with open(queries_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            query_data = json.loads(line)
                            queries[query_data['query_id']] = query_data['query_text']

            if qrels_file:
                logger.info(f"Loading qrels from: {qrels_file}")
                qrels_dict = defaultdict(dict)
                with open(qrels_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                            qrels_dict[qid][doc_id] = rel
                qrels = dict(qrels_dict)

        logger.info(f"Loaded {len(queries)} queries and qrels for {len(qrels)} queries")
        return queries, qrels

    def load_expansion_features(self, features_file: Path) -> Dict[str, Dict]:
        """
        Load expansion features for queries.

        Args:
            features_file: Path to features file

        Returns:
            Dictionary of {query_id: expansion_features}
        """
        logger.info(f"Loading expansion features from: {features_file}")

        if str(features_file).endswith('.jsonl'):
            # JSONL format
            features = {}
            with open(features_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        features[data['query_id']] = data.get('term_features', {})
        else:
            # Regular JSON format
            features_data = load_json(features_file)
            features = {}
            for qid, data in features_data.items():
                features[qid] = data.get('term_features', {})

        logger.info(f"Loaded expansion features for {len(features)} queries")
        return features

    def evaluate_model(self,
                       model_name: str,
                       queries: Dict[str, str],
                       qrels: Dict[str, Dict[str, int]],
                       expansion_features: Dict[str, Dict] = None) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model_name: Name of the model to evaluate
            queries: Query texts
            qrels: Relevance judgments
            expansion_features: Optional expansion features

        Returns:
            Evaluation results
        """
        if model_name not in self.retrievers:
            raise ValueError(f"Retriever '{model_name}' not setup")

        retriever = self.retrievers[model_name]

        logger.info(f"Evaluating model '{model_name}' on {len(queries)} queries")

        # Run retrieval
        retrieval_results = {}

        for query_id, query_text in tqdm(queries.items(), desc=f"Retrieving {model_name}"):
            # Get expansion features for this query
            query_expansion = expansion_features.get(query_id, {}) if expansion_features else {}

            # Retrieve documents
            results = retriever.search(
                query=query_text,
                expansion_features=query_expansion,
                top_k=self.top_k
            )

            retrieval_results[query_id] = results

        # Evaluate results
        evaluation_results = self.evaluator.evaluate_retrieval(retrieval_results, qrels)

        # Store results
        self.results[model_name] = {
            'retrieval_results': retrieval_results,
            'evaluation_metrics': evaluation_results,
            'num_queries': len(queries)
        }

        # Log results
        logger.info(f"Results for '{model_name}':")
        for metric, score in evaluation_results.items():
            logger.info(f"  {metric}: {score:.4f}")

        return evaluation_results

    def compare_models(self,
                       queries: Dict[str, str],
                       qrels: Dict[str, Dict[str, int]],
                       expansion_features: Dict[str, Dict] = None,
                       baseline_model: str = None) -> Dict[str, Any]:
        """
        Compare multiple models.

        Args:
            queries: Query texts
            qrels: Relevance judgments
            expansion_features: Optional expansion features
            baseline_model: Name of baseline model for comparison

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(self.retrievers)} models")

        all_results = {}

        # Evaluate each model
        for model_name in self.retrievers.keys():
            evaluation_results = self.evaluate_model(
                model_name, queries, qrels, expansion_features
            )
            all_results[model_name] = evaluation_results

        # Create comparison
        comparison = {
            'models_evaluated': list(self.retrievers.keys()),
            'evaluation_results': all_results,
            'num_queries': len(queries),
            'metrics': self.metrics,
            'baseline_model': baseline_model
        }

        # Compute improvements if baseline specified
        if baseline_model and baseline_model in all_results:
            baseline_scores = all_results[baseline_model]
            improvements = {}

            for model_name, results in all_results.items():
                if model_name == baseline_model:
                    continue

                model_improvements = {}
                for metric in self.metrics:
                    if metric in results and metric in baseline_scores:
                        baseline_score = baseline_scores[metric]
                        model_score = results[metric]

                        if baseline_score > 0:
                            improvement_pct = (model_score - baseline_score) / baseline_score * 100
                            model_improvements[f"{metric}_improvement_pct"] = improvement_pct
                            model_improvements[f"{metric}_improvement_abs"] = model_score - baseline_score

                improvements[model_name] = model_improvements

            comparison['improvements'] = improvements

        return comparison

    def save_results(self,
                     comparison_results: Dict[str, Any],
                     save_run_files: bool = True) -> None:
        """
        Save evaluation results.

        Args:
            comparison_results: Results from compare_models()
            save_run_files: Whether to save TREC run files
        """
        # Save main results
        results_file = self.output_dir / 'evaluation_results.json'
        save_json(comparison_results, results_file)
        logger.info(f"Saved evaluation results to: {results_file}")

        # Save run files if requested
        if save_run_files:
            runs_dir = ensure_dir(self.output_dir / 'runs')

            for model_name in self.results:
                if 'retrieval_results' in self.results[model_name]:
                    run_file = runs_dir / f'{model_name}.txt'
                    self._save_trec_run(
                        self.results[model_name]['retrieval_results'],
                        run_file,
                        model_name
                    )

        # Create summary report
        self._create_summary_report(comparison_results)

    def _save_trec_run(self, results: Dict[str, List[Tuple[str, float]]],
                       filepath: Path, run_name: str) -> None:
        """Save results in TREC run format."""
        with open(filepath, 'w') as f:
            for query_id, docs in results.items():
                for rank, (doc_id, score) in enumerate(docs, 1):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
        logger.info(f"Saved run file: {filepath}")

    def _create_summary_report(self, comparison_results: Dict[str, Any]) -> None:
        """Create a human-readable summary report."""
        report_file = self.output_dir / 'summary_report.txt'

        with open(report_file, 'w') as f:
            f.write("DENSE RETRIEVAL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Basic info
            f.write(f"Number of models evaluated: {len(comparison_results['models_evaluated'])}\n")
            f.write(f"Number of queries: {comparison_results['num_queries']}\n")
            f.write(f"Metrics: {', '.join(comparison_results['metrics'])}\n\n")

            # Results table
            f.write("RESULTS TABLE\n")
            f.write("-" * 30 + "\n")

            # Header
            f.write(f"{'Model':<20}")
            for metric in comparison_results['metrics']:
                f.write(f"{metric:>12}")
            f.write("\n")

            # Results
            for model_name in comparison_results['models_evaluated']:
                results = comparison_results['evaluation_results'][model_name]
                f.write(f"{model_name:<20}")
                for metric in comparison_results['metrics']:
                    score = results.get(metric, 0.0)
                    f.write(f"{score:>12.4f}")
                f.write("\n")

            # Improvements
            if 'improvements' in comparison_results:
                f.write(f"\nIMPROVEMENTS OVER BASELINE ({comparison_results['baseline_model']})\n")
                f.write("-" * 40 + "\n")

                for model_name, improvements in comparison_results['improvements'].items():
                    f.write(f"\n{model_name}:\n")
                    for metric in comparison_results['metrics']:
                        improvement_key = f"{metric}_improvement_pct"
                        if improvement_key in improvements:
                            improvement = improvements[improvement_key]
                            f.write(f"  {metric}: {improvement:+.2f}%\n")

        logger.info(f"Saved summary report to: {report_file}")


def load_models_config(config_file: Path) -> Dict[str, Path]:
    """
    Load models configuration file.

    Expected format:
    {
        "model_name_1": "/path/to/model1",
        "model_name_2": "/path/to/model2"
    }
    """
    config = load_json(config_file)
    return {name: Path(path) for name, path in config.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dense retrieval systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-path', type=str,
                       help='Path to single model directory')
    group.add_argument('--models-config', type=str,
                       help='Path to models configuration JSON file')

    # Index arguments
    parser.add_argument('--index-dir', type=str, required=True,
                        help='Directory containing dense index')

    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--dataset', type=str,
                            help='ir_datasets dataset name')
    data_group.add_argument('--queries-file', type=str,
                            help='Path to queries JSONL file')

    parser.add_argument('--qrels-file', type=str,
                        help='Path to qrels file (required if using --queries-file)')
    parser.add_argument('--features-file', type=str,
                        help='Path to expansion features file')

    # Evaluation arguments
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--metrics', nargs='+',
                        default=['recall_10', 'recall_100', 'ndcg_cut_10', 'ndcg_cut_100', 'map'],
                        help='Evaluation metrics to compute')
    parser.add_argument('--top-k', type=int, default=1000,
                        help='Number of documents to retrieve per query')
    parser.add_argument('--baseline-model', type=str,
                        help='Name of baseline model for comparison')

    # Other arguments
    parser.add_argument('--save-runs', action='store_true',
                        help='Save TREC run files')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Validate arguments
    if args.queries_file and not args.qrels_file:
        parser.error("--qrels-file is required when using --queries-file")

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("evaluate_retrieval", args.log_level,
                                      str(output_dir / 'evaluation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Initialize evaluator
        evaluator = RetrievalEvaluator(
            output_dir=output_dir,
            metrics=args.metrics,
            top_k=args.top_k
        )

        # Load models
        with TimedOperation(logger, "Loading models"):
            if args.model_path:
                # Single model
                evaluator.load_model('main_model', Path(args.model_path))
            else:
                # Multiple models
                models_config = load_models_config(Path(args.models_config))
                for model_name, model_path in models_config.items():
                    evaluator.load_model(model_name, model_path)

        # Setup retrievers
        with TimedOperation(logger, "Setting up retrievers"):
            for model_name in evaluator.models:
                evaluator.setup_retriever(model_name, Path(args.index_dir))

        # Load queries and qrels
        with TimedOperation(logger, "Loading queries and qrels"):
            queries, qrels = evaluator.load_queries_and_qrels(
                queries_file=Path(args.queries_file) if args.queries_file else None,
                qrels_file=Path(args.qrels_file) if args.qrels_file else None,
                dataset_name=args.dataset
            )

        # Load expansion features if provided
        expansion_features = None
        if args.features_file:
            with TimedOperation(logger, "Loading expansion features"):
                expansion_features = evaluator.load_expansion_features(Path(args.features_file))

        # Run evaluation
        with TimedOperation(logger, "Running evaluation"):
            comparison_results = evaluator.compare_models(
                queries=queries,
                qrels=qrels,
                expansion_features=expansion_features,
                baseline_model=args.baseline_model
            )

        # Save results
        with TimedOperation(logger, "Saving results"):
            evaluator.save_results(comparison_results, save_run_files=args.save_runs)

        # Log final summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Models evaluated: {len(evaluator.models)}")
        logger.info(f"Queries processed: {len(queries)}")
        logger.info(f"Results saved to: {output_dir}")

        # Show key results
        logger.info("\nKey Results:")
        for model_name in comparison_results['models_evaluated']:
            results = comparison_results['evaluation_results'][model_name]
            logger.info(f"  {model_name}:")
            for metric in ['ndcg_cut_10', 'map', 'recall_100']:
                if metric in results:
                    logger.info(f"    {metric}: {results[metric]:.4f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()