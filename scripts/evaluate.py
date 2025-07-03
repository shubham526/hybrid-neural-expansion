#!/usr/bin/env python3
"""
Evaluate Trained Neural Reranker

This script evaluates the trained neural reranker and compares it against baselines.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from src.models.neural_reranker import create_neural_reranker
from src.models.baseline_models import create_baseline_models
from src.evaluation.evaluator import TRECEvaluator
from src.utils.file_utils import load_json, save_json, save_trec_run, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        return query_obj.title
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained neural reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model-info-file', type=str, required=True,
                        help='Path to model_info.json from training')
    parser.add_argument('--feature-file', type=str, required=True,
                        help='Path to extracted features')

    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='IR dataset for evaluation')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--query-ids-file', type=str,
                        help='File with query IDs to evaluate (optional)')

    # Evaluation arguments
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run baseline comparisons')
    parser.add_argument('--save-runs', action='store_true',
                        help='Save TREC run files')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of documents to return per query')

    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("evaluate_model", args.log_level,
                                      str(output_dir / 'evaluation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Load model info and features
        with TimedOperation(logger, "Loading model and features"):
            model_info = load_json(args.model_info_file)
            features = load_json(args.feature_file)

            logger.info(f"Loaded model info: {model_info['model_name']}")
            logger.info(f"Learned weights: α={model_info['learned_weights']['alpha']:.4f}, "
                        f"β={model_info['learned_weights']['beta']:.4f}")
            logger.info(f"Loaded features for {len(features)} queries")

        # Load dataset
        with TimedOperation(logger, f"Loading dataset: {args.dataset}"):
            dataset = ir_datasets.load(args.dataset)

            queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}

            qrels = {}
            for qrel in dataset.qrels_iter():
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = {}
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

            first_stage_runs = {}
            if dataset.has_scoreddocs():
                for sdoc in dataset.scoreddocs_iter():
                    if sdoc.query_id not in first_stage_runs:
                        first_stage_runs[sdoc.query_id] = []
                    first_stage_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))

            logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")

            # Filter if needed
            if args.query_ids_file:
                with open(args.query_ids_file) as f:
                    subset_qids = {line.strip() for line in f if line.strip()}
                queries = {qid: text for qid, text in queries.items() if qid in subset_qids}
                features = {qid: data for qid, data in features.items() if qid in subset_qids}
                qrels = {qid: data for qid, data in qrels.items() if qid in subset_qids}
                first_stage_runs = {qid: data for qid, data in first_stage_runs.items() if qid in subset_qids}
                logger.info(f"Filtered to {len(queries)} queries")

        # Load trained model
        with TimedOperation(logger, "Loading trained neural reranker"):
            reranker = create_neural_reranker(
                model_name=model_info['model_name'],
                use_document_content=model_info['use_document_content'],
                max_expansion_terms=model_info['