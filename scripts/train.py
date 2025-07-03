#!/usr/bin/env python3
"""
Train Neural Reranker from JSONL Data

Updated for the new document-aware reranker that requires document content.
"""

import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.reranker import create_neural_reranker
from src.models.trainer import Trainer
from src.utils.file_utils import save_json, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation
from src.utils.data_utils import DocumentAwareExpansionDataset, PairwiseExpansionDataset


logger = logging.getLogger(__name__)


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def validate_training_data(data: List[Dict]) -> Dict[str, Any]:
    """
    Validate that training data has required fields for document-aware training.

    Returns:
        Validation statistics
    """
    total_examples = len(data)
    queries_with_doc_text = 0
    total_candidates = 0
    candidates_with_doc_text = 0

    for example in data:
        # Check if this query has any candidates with document text
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
    parser = argparse.ArgumentParser(description="Train neural reranker from JSONL data")

    # Data arguments
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to train.jsonl file')
    parser.add_argument('--val-file', type=str,
                        help='Path to validation.jsonl file (optional)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained model')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms to use')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for neural layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

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

            # Validate training data
            train_stats = validate_training_data(train_data)
            logger.info("Training data validation:")
            logger.info(
                f"  Queries with document text: {train_stats['queries_with_doc_text']}/{train_stats['total_queries']} ({train_stats['query_coverage']:.1%})")
            logger.info(
                f"  Candidates with document text: {train_stats['candidates_with_doc_text']}/{train_stats['total_candidates']} ({train_stats['candidate_coverage']:.1%})")

            if train_stats['candidate_coverage'] < 0.5:
                logger.warning(
                    "Less than 50% of candidates have document text. Consider regenerating training data with document content.")

            # Load validation data
            val_data = None
            if args.val_file:
                val_data = load_jsonl(Path(args.val_file))
                val_stats = validate_training_data(val_data)
                logger.info(f"Loaded {len(val_data)} validation examples")
                logger.info(f"  Validation document coverage: {val_stats['candidate_coverage']:.1%}")

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
        with TimedOperation(logger, "Creating neural reranker"):
            reranker = create_neural_reranker(
                model_name=args.model_name,
                max_expansion_terms=args.max_expansion_terms,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=args.device
            )

            initial_alpha, initial_beta = reranker.get_learned_weights()
            logger.info(f"Initial weights: α={initial_alpha:.3f}, β={initial_beta:.3f}")

        # Create trainer and train
        with TimedOperation(logger, f"Training for {args.num_epochs} epochs"):
            trainer = Trainer(
                model=reranker,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                loss_type=args.loss_type
            )

            history = trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epochs=args.num_epochs
            )

        # Save results
        with TimedOperation(logger, "Saving trained model"):
            # Save model state
            model_path = output_dir / 'neural_reranker.pt'
            torch.save(reranker.state_dict(), model_path)

            # Save model info
            final_alpha, final_beta = reranker.get_learned_weights()

            model_info = {
                'model_name': args.model_name,
                'max_expansion_terms': args.max_expansion_terms,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'learned_weights': {
                    'alpha': final_alpha,
                    'beta': final_beta
                },
                'training_config': {
                    'num_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'weight_decay': args.weight_decay,
                    'batch_size': args.batch_size,
                    'loss_type': args.loss_type,
                    'training_mode': args.training_mode,
                    'max_candidates_per_query': args.max_candidates_per_query
                },
                'model_path': str(model_path),
                'training_history': history,
                'data_validation': {
                    'train_stats': train_stats,
                    'val_stats': validate_training_data(val_data) if val_data else None
                }
            }

            save_json(model_info, output_dir / 'model_info.json')
            save_json(history, output_dir / 'training_history.json')

            logger.info("TRAINING COMPLETED!")
            logger.info(f"Initial weights: α={initial_alpha:.4f}, β={initial_beta:.4f}")
            logger.info(f"Final weights: α={final_alpha:.4f}, β={final_beta:.4f}")
            logger.info(f"Weight change: Δα={final_alpha - initial_alpha:+.4f}, Δβ={final_beta - initial_beta:+.4f}")
            logger.info(f"Model saved to: {model_path}")

            # Log final performance if available
            if history['train_loss']:
                initial_loss = history['train_loss'][0]
                final_loss = history['train_loss'][-1]
                logger.info(
                    f"Training loss: {initial_loss:.4f} → {final_loss:.4f} ({((final_loss - initial_loss) / initial_loss * 100):+.1f}%)")

            if history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                logger.info(f"Final validation loss: {final_val_loss:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()