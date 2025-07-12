#!/usr/bin/env python3
"""
Enhanced Train Neural Reranker with Dev Set Evaluation

Updated to:
1. Load evaluation data (queries, documents, qrels)
2. Evaluate after each epoch
3. Save best model based on dev set performance
4. Create dev run files for each epoch
"""

import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add the project's root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
import ir_datasets

from cross_encoder.src.models.reranker import create_neural_reranker
from cross_encoder.src.models.trainer import EvaluationAwareTrainer
from cross_encoder.src.utils.file_utils import save_json, ensure_dir
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation
from cross_encoder.src.utils.data_utils import DocumentAwareExpansionDataset, PairwiseExpansionDataset

logger = logging.getLogger(__name__)


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        return query_obj.title
    return ""


class DocumentLoader:
    """Document loader for evaluation."""

    def __init__(self, dataset=None, documents_dict=None):
        """
        Initialize document loader.

        Args:
            dataset: ir_datasets dataset object
            documents_dict: Pre-loaded documents dictionary {doc_id: doc_text}
        """
        self.documents = {}

        if documents_dict:
            self.documents = documents_dict
            logger.info(f"Initialized DocumentLoader with {len(self.documents)} pre-loaded documents")
        elif dataset:
            self._load_documents_from_dataset(dataset)
        else:
            logger.warning("No documents provided to DocumentLoader")

    def _load_documents_from_dataset(self, dataset):
        """Load documents from ir_datasets dataset."""
        logger.info("Loading documents from dataset...")

        for doc in dataset.docs_iter():
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                title = getattr(doc, 'title', '')
                body = doc.body
                doc_text = f"{title} {body}".strip() if title else body
            else:
                doc_text = str(doc)

            self.documents[doc.doc_id] = doc_text

        logger.info(f"Loaded {len(self.documents)} documents from dataset")

    def get_document(self, doc_id: str) -> str:
        """Get document text by ID."""
        return self.documents.get(doc_id, "")


def load_qrels_file(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from TREC format file.

    Args:
        qrels_file: Path to qrels file

    Returns:
        Dictionary {query_id: {doc_id: relevance}}
    """
    logger.info(f"Loading qrels from: {qrels_file}")

    qrels = defaultdict(dict)

    with open(qrels_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 4:
                logger.warning(f"Invalid qrels line {line_num}: {line}")
                continue

            query_id = parts[0]
            doc_id = parts[2]
            try:
                relevance = int(parts[3])
                qrels[query_id][doc_id] = relevance
            except ValueError:
                logger.warning(f"Invalid relevance score on line {line_num}: {parts[3]}")
                continue

    logger.info(f"Loaded qrels for {len(qrels)} queries")
    return dict(qrels)


def load_qrels_smart(qrels_source: str, queries: Dict[str, str] = None) -> Dict[str, Dict[str, int]]:
    """
    Load qrels from either ir_datasets or TREC format file.

    Args:
        qrels_source: Either an ir_datasets dataset name or path to TREC qrels file
        queries: Optional queries dict to filter qrels

    Returns:
        Dictionary {query_id: {doc_id: relevance}}
    """
    logger.info(f"Loading qrels from: {qrels_source}")

    # Try to load as ir_datasets first
    try:
        import ir_datasets
        dataset = ir_datasets.load(qrels_source)

        # Check if dataset has qrels
        if not hasattr(dataset, 'qrels_iter'):
            raise ValueError(f"Dataset {qrels_source} does not have qrels")

        logger.info(f"Loading qrels from ir_datasets: {qrels_source}")
        qrels = defaultdict(dict)

        for qrel in dataset.qrels_iter():
            # Filter to specific queries if provided
            if queries is None or qrel.query_id in queries:
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        logger.info(f"Loaded qrels for {len(qrels)} queries from ir_datasets")
        return dict(qrels)

    except Exception as e:
        logger.info(f"Could not load as ir_datasets ({e}), trying as TREC file...")

        # Try to load as TREC format file
        try:
            qrels_path = Path(qrels_source)
            if not qrels_path.exists():
                raise FileNotFoundError(f"Qrels file not found: {qrels_path}")

            all_qrels = load_qrels_file(qrels_path)

            # Filter to specific queries if provided
            if queries is not None:
                qrels = {qid: qrel_dict for qid, qrel_dict in all_qrels.items() if qid in queries}
            else:
                qrels = all_qrels

            logger.info(f"Loaded qrels for {len(qrels)} queries from TREC file")
            return qrels

        except Exception as file_error:
            logger.error(f"Failed to load qrels as both ir_datasets and TREC file:")
            logger.error(f"  ir_datasets error: {e}")
            logger.error(f"  TREC file error: {file_error}")
            raise ValueError(f"Could not load qrels from {qrels_source}")


def load_evaluation_data_from_jsonl(val_file: Path, dev_qrels_file: str = None, dataset_name: str = None):
    """
    Load evaluation data from validation JSONL file and optional qrels file.

    Args:
        val_file: Path to validation JSONL file
        dev_qrels_file: Dev qrels source (ir_datasets name or TREC file path)
        dataset_name: Dataset name for loading documents via ir_datasets

    Returns:
        Tuple of (queries, qrels, document_loader)
    """
    logger.info(f"Loading evaluation data from JSONL: {val_file}")

    val_data = load_jsonl(val_file)

    # Extract queries
    queries = {}
    all_doc_ids = set()

    for example in val_data:
        query_id = example['query_id']
        queries[query_id] = example['query_text']

        # Collect all document IDs
        for candidate in example['candidates']:
            doc_id = candidate['doc_id']
            all_doc_ids.add(doc_id)

    logger.info(f"Extracted {len(queries)} queries from JSONL")

    # Load qrels from external source if provided
    qrels = {}
    if dev_qrels_file:
        # Use the smart loader that handles both ir_datasets and TREC files
        qrels = load_qrels_smart(dev_qrels_file, queries)
    else:
        # Fallback: extract qrels from JSONL file
        logger.info("No external qrels source provided, extracting from JSONL...")
        qrels_from_jsonl = defaultdict(dict)

        for example in val_data:
            query_id = example['query_id']
            for candidate in example['candidates']:
                doc_id = candidate['doc_id']
                relevance = candidate['relevance']
                if relevance > 0:  # Only store positive relevance judgments
                    qrels_from_jsonl[query_id][doc_id] = relevance

        qrels = dict(qrels_from_jsonl)
        logger.info(f"Extracted qrels for {len(qrels)} queries from JSONL")

    # Load documents
    document_loader = None
    if dataset_name:
        try:
            dataset = ir_datasets.load(dataset_name)
            document_loader = DocumentLoader(dataset=dataset)
        except Exception as e:
            logger.warning(f"Could not load dataset {dataset_name}: {e}")

    # If no dataset or loading failed, try to extract documents from JSONL
    if not document_loader:
        logger.info("Extracting documents from JSONL file...")
        documents_dict = {}

        for example in val_data:
            for candidate in example['candidates']:
                if 'doc_text' in candidate:
                    documents_dict[candidate['doc_id']] = candidate['doc_text']

        document_loader = DocumentLoader(documents_dict=documents_dict)

    return dict(queries), qrels, document_loader


def load_evaluation_data_from_dataset(dataset_name: str, val_file: Path = None, dev_qrels_file: str = None):
    """
    Load evaluation data from ir_datasets with optional external qrels.

    Args:
        dataset_name: ir_datasets dataset name
        val_file: Optional validation file to filter queries
        dev_qrels_file: External qrels source (ir_datasets name or TREC file path)

    Returns:
        Tuple of (queries, qrels, document_loader)
    """
    logger.info(f"Loading evaluation data from dataset: {dataset_name}")

    dataset = ir_datasets.load(dataset_name)

    # Load all queries
    all_queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}

    # Filter to validation queries if file provided
    if val_file and val_file.exists():
        val_data = load_jsonl(val_file)
        val_query_ids = {example['query_id'] for example in val_data}
        queries = {qid: text for qid, text in all_queries.items() if qid in val_query_ids}
        logger.info(f"Filtered to {len(queries)} validation queries")
    else:
        queries = all_queries
        logger.info(f"Using all {len(queries)} queries for evaluation")

    # Load qrels - prioritize external source over dataset qrels
    qrels = {}
    if dev_qrels_file:
        # Use the smart loader that handles both ir_datasets and TREC files
        qrels = load_qrels_smart(dev_qrels_file, queries)
    else:
        # Fallback to dataset qrels
        logger.info("No external qrels source provided, using dataset qrels...")
        qrels_from_dataset = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            if qrel.query_id in queries:
                qrels_from_dataset[qrel.query_id][qrel.doc_id] = qrel.relevance
        qrels = dict(qrels_from_dataset)
        logger.info(f"Loaded qrels for {len(qrels)} queries from dataset")

    # Load documents
    document_loader = DocumentLoader(dataset=dataset)

    return queries, qrels, document_loader


def validate_training_data(data: List[Dict]) -> Dict[str, Any]:
    """Validate training data has required fields."""
    total_examples = len(data)
    queries_with_doc_text = 0
    total_candidates = 0
    candidates_with_doc_text = 0

    for example in data:
        has_doc_text = False
        for candidate in example.get('candidates', []):
            total_candidates += 1
            if 'doc_text' in candidate:
                candidates_with_doc_text += 1
                has_doc_text = True
        if has_doc_text:
            queries_with_doc_text += 1

    stats = {
        'total_queries': total_examples,
        'queries_with_doc_text': queries_with_doc_text,
        'total_candidates': total_candidates,
        'candidates_with_doc_text': candidates_with_doc_text,
        'query_coverage': queries_with_doc_text / total_examples if total_examples > 0 else 0,
        'candidate_coverage': candidates_with_doc_text / total_candidates if total_candidates > 0 else 0
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Train neural reranker with dev set evaluation")

    # Data arguments
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to train.jsonl file')
    parser.add_argument('--val-file', type=str,
                        help='Path to validation.jsonl file')
    parser.add_argument('--dev-qrels', type=str,
                        help='Dev qrels source: either ir_datasets dataset name or path to TREC format qrels file')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name for loading documents (optional)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained model and evaluation results')
    # Add this with other model arguments
    parser.add_argument('--ablation-mode', type=str, default='both',
                        choices=['both', 'rm3_only', 'cosine_only'],
                        help='Ablation mode: both components, RM3 only, or cosine similarity only')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    # Add these with the other model arguments (around line 230):
    parser.add_argument('--force-hf', action='store_true',
                        help='Force using HuggingFace transformers instead of SentenceTransformers')
    parser.add_argument('--pooling-strategy', choices=['cls', 'mean', 'max'], default='cls',
                        help='Pooling strategy for HuggingFace models')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms to use')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for neural layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--scoring-method', type=str, default='neural',
                        choices=['neural', 'bilinear', 'cosine'],
                        help='Scoring method: neural layers, bilinear, or cosine similarity')

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'bce', 'ranking'],
                        help='Loss function type')
    parser.add_argument('--training-mode', type=str, default='pointwise',
                        choices=['pointwise', 'pairwise'],
                        help='Training mode: pointwise or pairwise ranking')

    # Data processing arguments
    parser.add_argument('--max-candidates-per-query', type=int, default=100,
                        help='Maximum candidates per query')
    parser.add_argument('--max-negatives-per-query', type=int, default=50,
                        help='Maximum negative examples per query (for efficiency)')
    parser.add_argument('--max-pairs-per-query', type=int, default=50,
                        help='Maximum positive-negative pairs per query (pairwise mode)')

    # NEW: Evaluation arguments
    parser.add_argument('--eval-metric', type=str, default='ndcg_cut_20',
                        help='Metric for best model selection')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--disable-dev-eval', action='store_true',
                        help='Disable dev set evaluation (use original training)')
    parser.add_argument('--rerank-top-k', type=int, default=100,
                        help='Number of top documents to rerank during evaluation')

    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("train_neural_reranker", args.log_level,
                                      str(output_dir / 'training.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Load training data
        with TimedOperation(logger, "Loading training data"):
            train_data = load_jsonl(Path(args.train_file))
            logger.info(f"Loaded {len(train_data)} training examples")

            train_stats = validate_training_data(train_data)
            logger.info("Training data validation:")
            logger.info(f"  Query coverage: {train_stats['query_coverage']:.1%}")
            logger.info(f"  Candidate coverage: {train_stats['candidate_coverage']:.1%}")

            if train_stats['candidate_coverage'] < 0.5:
                logger.warning("Less than 50% of candidates have document text!")

        # Load validation data and evaluation components
        val_data = None
        queries = None
        qrels = None
        document_loader = None

        if args.val_file and not args.disable_dev_eval:
            with TimedOperation(logger, "Loading evaluation data"):
                val_data = load_jsonl(Path(args.val_file))
                val_stats = validate_training_data(val_data)
                logger.info(f"Loaded {len(val_data)} validation examples")
                logger.info(f"Validation coverage: {val_stats['candidate_coverage']:.1%}")

                # Require dev qrels for evaluation
                if not args.dev_qrels:
                    logger.error(
                        "--dev-qrels is required when --val-file is provided (unless --disable-dev-eval is used)")
                    logger.error("Please provide either an ir_datasets dataset name or path to TREC format qrels file")
                    sys.exit(1)

                # Load evaluation components - let load_qrels_smart handle validation
                if args.dataset:
                    # Load from dataset with external qrels
                    queries, qrels, document_loader = load_evaluation_data_from_dataset(
                        args.dataset, Path(args.val_file), args.dev_qrels
                    )
                else:
                    # Load from JSONL with external qrels
                    queries, qrels, document_loader = load_evaluation_data_from_jsonl(
                        Path(args.val_file), args.dev_qrels, args.dataset
                    )

                logger.info(f"Evaluation setup:")
                logger.info(f"  Queries: {len(queries)}")
                logger.info(f"  Qrels: {len(qrels)} queries with relevance judgments")
                logger.info(f"  Documents: {len(document_loader.documents)}")

                # Validate that we have qrels for validation queries
                val_query_ids = {example['query_id'] for example in val_data}
                qrels_query_ids = set(qrels.keys())
                common_queries = val_query_ids & qrels_query_ids

                if not common_queries:
                    logger.error("No overlap between validation queries and qrels!")
                    logger.error(f"Validation queries: {len(val_query_ids)}")
                    logger.error(f"Qrels queries: {len(qrels_query_ids)}")
                    sys.exit(1)
                elif len(common_queries) < len(val_query_ids):
                    logger.warning(f"Only {len(common_queries)}/{len(val_query_ids)} validation queries have qrels")

        elif args.disable_dev_eval:
            logger.info("Dev set evaluation disabled - using original training mode")
            if args.val_file:
                val_data = load_jsonl(Path(args.val_file))
                logger.info(f"Loaded {len(val_data)} validation examples (loss computation only)")
        else:
            logger.info("No validation file provided - training without validation")

        # Create datasets
        with TimedOperation(logger, "Creating datasets"):
            if args.training_mode == 'pointwise':
                train_dataset = DocumentAwareExpansionDataset(
                    train_data,
                    max_candidates_per_query=args.max_candidates_per_query,
                    max_negatives_per_query=args.max_negatives_per_query
                )

                val_dataset = None
                if val_data:
                    val_dataset = DocumentAwareExpansionDataset(
                        val_data,
                        max_candidates_per_query=args.max_candidates_per_query,
                        max_negatives_per_query=args.max_negatives_per_query
                    )

            elif args.training_mode == 'pairwise':
                train_dataset = PairwiseExpansionDataset(
                    train_data,
                    max_pairs_per_query=args.max_pairs_per_query
                )

                val_dataset = None
                if val_data:
                    val_dataset = PairwiseExpansionDataset(
                        val_data,
                        max_pairs_per_query=args.max_pairs_per_query
                    )

        # Create model
        # Update this section (around line 410):
        with TimedOperation(logger, "Creating neural reranker"):
            reranker = create_neural_reranker(
                model_name=args.model_name,
                max_expansion_terms=args.max_expansion_terms,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                scoring_method=args.scoring_method,
                device=args.device,
                force_hf=args.force_hf,  # NEW parameter
                pooling_strategy=args.pooling_strategy,  # NEW parameter
                ablation_mode=args.ablation_mode
            )
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{torch.cuda.current_device()}')
                reranker = reranker.to(device)
                print(f"Model moved to consistent device: {device}")

            initial_alpha, initial_beta, initial_lambda = reranker.get_learned_weights()
            logger.info(f"Initial weights: α={initial_alpha:.3f}, β={initial_beta:.3f}, λ={initial_lambda:.3f}")
            logger.info(f"Scoring method: {reranker.get_scoring_method()}")

        # Create trainer
        with TimedOperation(logger, "Initializing trainer"):
            trainer = EvaluationAwareTrainer(
                model=reranker,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                loss_type=args.loss_type,
                # NEW: Evaluation parameters
                queries=queries,
                document_loader=document_loader,
                qrels=qrels,
                output_dir=output_dir,
                eval_metric=args.eval_metric,
                patience=args.patience,
                rerank_top_k=args.rerank_top_k  # NEW parameter
            )

        # Train model
        with TimedOperation(logger, f"Training for {args.num_epochs} epochs"):
            history = trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=args.num_epochs
            )

        # Save results
        with TimedOperation(logger, "Saving trained model and results"):
            # Save final model state (in addition to best model)
            final_model_path = output_dir / 'final_model.pt'
            torch.save(reranker.state_dict(), final_model_path)

            # Get final weights
            final_alpha, final_beta, final_lambda = reranker.get_learned_weights()  # MODIFIED

            # Create comprehensive model info
            model_info = {
                'model_name': args.model_name,
                'ablation_mode': args.ablation_mode,
                'force_hf': args.force_hf,  # NEW
                'pooling_strategy': args.pooling_strategy,  # NEW
                'max_expansion_terms': args.max_expansion_terms,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'scoring_method': args.scoring_method,  # NEW field
                'learned_weights': {
                    'alpha': final_alpha,
                    'beta': final_beta,
                    'expansion_weight': final_lambda,
                },
                'training_config': {
                    'num_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'batch_size': args.batch_size,
                    'loss_type': args.loss_type,
                    'training_mode': args.training_mode,
                    'max_candidates_per_query': args.max_candidates_per_query,
                    'eval_metric': args.eval_metric,
                    'patience': args.patience
                },
                'final_model_path': str(final_model_path),
                'training_history': history,
                'data_validation': {
                    'train_stats': train_stats,
                    'val_stats': validate_training_data(val_data) if val_data else None
                }
            }

            # Add evaluation results if available
            if hasattr(trainer, 'best_score') and trainer.best_score > 0:
                model_info['best_dev_performance'] = {
                    'best_epoch': trainer.best_epoch,
                    'best_score': trainer.best_score,
                    'metric': args.eval_metric
                }

            if hasattr(trainer, 'eval_history'):
                model_info['eval_history'] = trainer.eval_history

            save_json(model_info, output_dir / 'model_info.json')
            save_json(history, output_dir / 'training_history.json')

            # Log training completion
            logger.info("TRAINING COMPLETED!")
            logger.info(f"Initial weights: α={initial_alpha:.4f}, β={initial_beta:.4f}, λ={initial_lambda:.4f}")
            logger.info(f"Final weights: α={final_alpha:.4f}, β={final_beta:.4f}, λ={final_lambda:.4f}")
            logger.info(
                f"Weight changes: Δα={final_alpha - initial_alpha:+.4f}, Δβ={final_beta - initial_beta:+.4f}, Δλ={final_lambda - initial_lambda:+.4f}")
            # MODIFIED BLOCK END

            # Log best model info
            if hasattr(trainer, 'best_score') and trainer.best_score > 0:
                logger.info(f"Best model: Epoch {trainer.best_epoch}, {args.eval_metric}={trainer.best_score:.4f}")
                best_model_path = output_dir / 'best_model.pt'
                if best_model_path.exists():
                    logger.info(f"Best model saved to: {best_model_path}")

            logger.info(f"Final model saved to: {final_model_path}")

            # Log performance summary
            if history['train_loss']:
                initial_loss = history['train_loss'][0]
                final_loss = history['train_loss'][-1]
                logger.info(
                    f"Training loss: {initial_loss:.4f} → {final_loss:.4f} ({((final_loss - initial_loss) / initial_loss * 100):+.1f}%)")

            if history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                logger.info(f"Final validation loss: {final_val_loss:.4f}")

            if history.get('dev_scores'):
                dev_scores = [s for s in history['dev_scores'] if s > 0]
                if dev_scores:
                    logger.info(f"Dev {args.eval_metric}: {dev_scores[0]:.4f} → {dev_scores[-1]:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()