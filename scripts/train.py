#!/usr/bin/env python3
"""
Train Neural Reranker from JSONL Data

This script trains the neural reranker using pre-created train.jsonl files.
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

from src.models.neural_reranker import create_neural_reranker
from src.models.neural_trainer import NeuralTrainer
from src.utils.file_utils import save_json, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class JSONLDataset(torch.utils.data.Dataset):
    """Dataset that loads from JSONL files."""

    def __init__(self, jsonl_file: Path):
        """Load data from JSONL file."""
        self.examples = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    self.examples.append(example)

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (keep at 1 for simplicity)')

    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("train_neural_jsonl", args.log_level,
                                      str(output_dir / 'training.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Load datasets
        with TimedOperation(logger, "Loading training data"):
            train_dataset = JSONLDataset(Path(args.train_file))

            val_dataset = None
            if args.val_file:
                val_dataset = JSONLDataset(Path(args.val_file))

        # Create model
        with TimedOperation(logger, "Creating neural reranker"):
            reranker = create_neural_reranker(
                model_name=args.model_name,
                use_document_content=False,  # Using query-level scoring
                max_expansion_terms=args.max_expansion_terms,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                device=args.device
            )

            initial_alpha, initial_beta = reranker.get_learned_weights()
            logger.info(f"Initial weights: α={initial_alpha:.3f}, β={initial_beta:.3f}")

        # Create trainer and train
        with TimedOperation(logger, f"Training for {args.num_epochs} epochs"):
            trainer = NeuralTrainer(
                model=reranker,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size
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
                'learned_weights': {
                    'alpha': final_alpha,
                    'beta': final_beta
                },
                'training_config': vars(args),
                'model_path': str(model_path),
                'training_history': history
            }

            save_json(model_info, output_dir / 'model_info.json')
            save_json(history, output_dir / 'training_history.json')

            logger.info("TRAINING COMPLETED!")
            logger.info(f"Final weights: α={final_alpha:.4f}, β={final_beta:.4f}")
            logger.info(f"Model saved to: {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()