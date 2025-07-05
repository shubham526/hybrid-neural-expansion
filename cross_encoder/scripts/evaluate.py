#!/usr/bin/env python3
"""
Evaluate Trained Neural Reranker

This script evaluates the trained neural reranker and compares it against baselines.
Updated for the new document-aware reranker architecture.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from cross_encoder.src.models.reranker import create_neural_reranker
from cross_encoder.src.evaluation.evaluator import TRECEvaluator
from cross_encoder.src.utils.file_utils import load_json, save_json, save_trec_run, ensure_dir
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        return query_obj.title
    return ""


class DocumentLoader:
    """Simple document loader for evaluation."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.documents = self._load_all_documents()
        logger.info(f"Loaded {len(self.documents)} documents into memory")

    def _load_all_documents(self) -> Dict[str, str]:
        """Load all documents into a dictionary."""
        logger.info("Loading all documents into memory...")
        documents = {}

        doc_count = 0
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents"):
            # Handle different document field formats
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                title = getattr(doc, 'title', '')
                body = doc.body
                doc_text = f"{title} {body}".strip() if title else body
            else:
                doc_text = str(doc)  # Fallback

            documents[doc.doc_id] = doc_text
            doc_count += 1

            # Progress logging for large collections
            if doc_count % 100000 == 0:
                logger.info(f"Loaded {doc_count} documents...")

        logger.info(f"Finished loading {len(documents)} documents")
        return documents

    def get_document(self, doc_id: str) -> str:
        """Get a single document by ID."""
        return self.documents.get(doc_id, "")


def load_test_data_from_jsonl(jsonl_file: Path) -> Dict[str, Any]:
    """
    Load test data from JSONL file (alternative to loading from ir_datasets).

    Args:
        jsonl_file: Path to test.jsonl file

    Returns:
        Dictionary with queries, features, candidates, and qrels
    """
    import json

    logger.info(f"Loading test data from JSONL: {jsonl_file}")

    queries = {}
    features = {}
    candidates = {}
    qrels = {}

    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)

                query_id = example['query_id']
                queries[query_id] = example['query_text']
                features[query_id] = {'term_features': example['expansion_features']}

                # Extract candidates and qrels
                candidate_list = []
                query_qrels = {}

                for candidate in example['candidates']:
                    doc_id = candidate['doc_id']
                    score = candidate['score']
                    relevance = candidate['relevance']

                    candidate_list.append((doc_id, score))
                    if relevance > 0:
                        query_qrels[doc_id] = relevance

                candidates[query_id] = candidate_list
                if query_qrels:
                    qrels[query_id] = query_qrels

    logger.info(f"Loaded {len(queries)} queries from JSONL")

    return {
        'queries': queries,
        'features': features,
        'candidates': candidates,
        'qrels': qrels
    }


def create_baseline_run(first_stage_runs: Dict[str, List[Tuple[str, float]]],
                        top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
    """Create baseline run from first-stage results."""
    baseline_run = {}
    for query_id, candidates in first_stage_runs.items():
        # Just use first-stage scores as-is
        baseline_run[query_id] = candidates[:top_k]
    return baseline_run


def evaluate_neural_reranker(reranker,
                             queries: Dict[str, str],
                             features: Dict[str, Any],
                             first_stage_runs: Dict[str, List[Tuple[str, float]]],
                             document_loader: DocumentLoader,
                             top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
    """
    Evaluate neural reranker on test data.

    Args:
        reranker: Trained neural reranker
        queries: Query texts
        features: Expansion features
        first_stage_runs: First-stage candidate documents
        document_loader: Document loader for getting document text
        top_k: Number of results to return

    Returns:
        Reranked results
    """
    logger.info("Running neural reranker evaluation...")

    reranked_results = {}

    for query_id in tqdm(queries.keys(), desc="Reranking queries"):
        if query_id not in features or query_id not in first_stage_runs:
            logger.warning(f"Missing data for query {query_id}, skipping")
            continue

        query_text = queries[query_id]
        expansion_features = features[query_id]['term_features']
        candidates = first_stage_runs[query_id]

        try:
            # Create document texts dict for this query's candidates
            document_texts = {}
            for doc_id, _ in candidates:
                doc_text = document_loader.get_document(doc_id)
                if doc_text:
                    document_texts[doc_id] = doc_text

            # Rerank using neural model
            reranked = reranker.rerank_candidates(
                query=query_text,
                expansion_features=expansion_features,
                candidates=candidates,
                document_texts=document_texts,
                top_k=top_k
            )

            reranked_results[query_id] = reranked

        except Exception as e:
            logger.error(f"Error reranking query {query_id}: {e}")
            # Fallback to first-stage results
            reranked_results[query_id] = candidates[:top_k]

    logger.info(f"Completed reranking for {len(reranked_results)} queries")
    return reranked_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained neural reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model-info-file', type=str, required=True,
                        help='Path to model_info.json from training')

    # Data arguments - support both JSONL and dataset modes
    parser.add_argument('--test-file', type=str,
                        help='Path to test.jsonl file (JSONL mode)')
    parser.add_argument('--dataset', type=str,
                        help='IR dataset for evaluation (dataset mode)')
    parser.add_argument('--feature-file', type=str,
                        help='Path to extracted features (dataset mode)')

    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--query-ids-file', type=str,
                        help='File with query IDs to evaluate (optional)')
    # Add these with other model arguments (around line 155):
    parser.add_argument('--force-hf', action='store_true',
                        help='Force using HuggingFace transformers (overrides model_info.json)')
    parser.add_argument('--pooling-strategy', choices=['cls', 'mean', 'max'],
                        help='Pooling strategy for HuggingFace models (overrides model_info.json)')

    # Evaluation arguments
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run baseline comparisons')
    parser.add_argument('--save-runs', action='store_true',
                        help='Save TREC run files')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of documents to return per query')

    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    # Validate arguments
    if not args.test_file and not (args.dataset and args.feature_file):
        parser.error("Either --test-file or (--dataset and --feature-file) required")

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("evaluate_model", args.log_level,
                                      str(output_dir / 'evaluation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Load model info
        with TimedOperation(logger, "Loading model info"):
            model_info = load_json(args.model_info_file)

            logger.info(f"Loaded model info: {model_info['model_name']}")
            logger.info(f"Learned weights: α={model_info['learned_weights']['alpha']:.4f}, "
                        f"β={model_info['learned_weights']['beta']:.4f}")

        # Load test data
        if args.test_file:
            # JSONL mode - load from test file
            with TimedOperation(logger, "Loading test data from JSONL"):
                test_data = load_test_data_from_jsonl(Path(args.test_file))
                queries = test_data['queries']
                features = test_data['features']
                first_stage_runs = test_data['candidates']
                qrels = test_data['qrels']

                # For JSONL mode, we need to determine the dataset to load documents
                # This is a limitation - we need the dataset for document loading
                if args.dataset:
                    dataset = ir_datasets.load(args.dataset)
                else:
                    # Try to infer dataset from model info or ask user to specify
                    raise ValueError("--dataset required for document loading in JSONL mode")

        else:
            # Dataset mode - load from ir_datasets
            with TimedOperation(logger, f"Loading dataset: {args.dataset}"):
                dataset = ir_datasets.load(args.dataset)
                features = load_json(args.feature_file)

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

        # Filter queries if specified
        if args.query_ids_file:
            with open(args.query_ids_file) as f:
                subset_qids = {line.strip() for line in f if line.strip()}
            queries = {qid: text for qid, text in queries.items() if qid in subset_qids}
            features = {qid: data for qid, data in features.items() if qid in subset_qids}
            qrels = {qid: data for qid, data in qrels.items() if qid in subset_qids}
            first_stage_runs = {qid: data for qid, data in first_stage_runs.items() if qid in subset_qids}
            logger.info(f"Filtered to {len(queries)} queries")

        # Load documents
        with TimedOperation(logger, "Loading document collection"):
            document_loader = DocumentLoader(dataset)

        # Load trained model
        # Load trained model
        with TimedOperation(logger, "Loading trained neural reranker"):
            # Use CLI args if provided, otherwise fall back to model_info
            force_hf = args.force_hf if hasattr(args, 'force_hf') and args.force_hf else model_info.get('force_hf',
                                                                                                        False)
            pooling_strategy = args.pooling_strategy if hasattr(args,
                                                                'pooling_strategy') and args.pooling_strategy else model_info.get(
                'pooling_strategy', 'cls')
            ablation_mode = model_info.get('ablation_mode', 'both')

            reranker = create_neural_reranker(
                model_name=model_info['model_name'],
                max_expansion_terms=model_info['max_expansion_terms'],
                hidden_dim=model_info['hidden_dim'],
                dropout=model_info.get('dropout', 0.1),
                scoring_method=model_info.get('scoring_method', 'neural'),
                force_hf=force_hf,
                pooling_strategy=pooling_strategy,
                ablation_mode=ablation_mode
            )

            logger.info(
                f"Model configuration: force_hf={force_hf}, pooling={pooling_strategy}, ablation={ablation_mode}")  # UPDATE THIS LINE
            # In evaluate.py, after loading model
            logger.info(f"Ablation mode: {ablation_mode}")
            if ablation_mode != "both":
                logger.info(f"Running ablation - using {ablation_mode.replace('_', ' ')} component only")

            # Load trained weights - try multiple possible paths
            model_paths = [
                Path(args.model_info_file).parent / 'best_model.pt',
                Path(args.model_info_file).parent / 'final_model.pt',
                Path(args.model_info_file).parent / 'neural_reranker.pt'
            ]

            model_loaded = False
            for model_path in model_paths:
                if model_path.exists():
                    reranker.load_state_dict(torch.load(model_path, map_location=reranker.device))
                    logger.info(f"Loaded trained model from: {model_path}")
                    model_loaded = True
                    break

            if not model_loaded:
                logger.warning("No trained model file found!")
                logger.warning("Using randomly initialized weights!")

            # Verify learned weights
            alpha, beta = reranker.get_learned_weights()
            logger.info(f"Current model weights: α={alpha:.4f}, β={beta:.4f}")

        # Create runs
        runs = {}

        # Baseline run (first-stage results)
        baseline_run = create_baseline_run(first_stage_runs, args.top_k)
        runs['baseline'] = baseline_run

        # Neural reranker run
        with TimedOperation(logger, "Running neural reranker"):
            neural_run = evaluate_neural_reranker(
                reranker=reranker,
                queries=queries,
                features=features,
                first_stage_runs=first_stage_runs,
                document_loader=document_loader,
                top_k=args.top_k
            )
            runs['neural_reranker'] = neural_run

        # Evaluate all runs
        with TimedOperation(logger, "Evaluating runs"):
            evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'ndcg_cut_20', 'recip_rank', 'recall_100'])

            evaluation_results = evaluator.evaluate_multiple_runs(runs, qrels)

            # Compute improvements
            comparison = evaluator.compare_runs(runs, qrels, baseline_run='baseline')

        # Save results
        with TimedOperation(logger, "Saving results"):
            # Save evaluation results
            save_json(evaluation_results, output_dir / 'evaluation_results.json')
            save_json(comparison, output_dir / 'comparison_results.json')

            # Save learned weights
            weights_info = {
                'learned_weights': model_info['learned_weights'],
                'current_weights': {'alpha': alpha, 'beta': beta},
                'model_info': model_info
            }
            save_json(weights_info, output_dir / 'learned_weights.json')

            # Save TREC run files if requested
            if args.save_runs:
                for run_name, run_results in runs.items():
                    run_file = output_dir / f'{run_name}.txt'
                    save_trec_run(run_results, run_file, run_name)
                    logger.info(f"Saved run file: {run_file}")

        # Log results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)

        for run_name, metrics in evaluation_results.items():
            logger.info(f"\n{run_name.upper()}:")
            for metric, score in metrics.items():
                logger.info(f"  {metric}: {score:.4f}")

        # Log improvements
        if 'improvements' in comparison:
            logger.info(f"\nIMPROVEMENTS OVER BASELINE:")
            for run_name, improvements in comparison['improvements'].items():
                logger.info(f"\n{run_name.upper()}:")
                for metric, improvement in improvements.items():
                    if 'improvement_pct' in metric:
                        base_metric = metric.replace('_improvement_pct', '')
                        logger.info(f"  {base_metric}: {improvement:+.1f}%")

        # Log learned weights
        logger.info(f"\nLEARNED WEIGHTS:")
        logger.info(f"  α (RM3 weight): {alpha:.4f}")
        logger.info(f"  β (Semantic weight): {beta:.4f}")

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()