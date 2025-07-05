#!/usr/bin/env python3
"""
Train Bi-Encoder with Contrastive Learning

This script trains bi-encoder models for dense retrieval using contrastive learning
on query-document pairs with expansion features. It supports multiple training strategies
and comprehensive evaluation during training.

Usage:
    # Train from bi-encoder data
    python train_bi_encoder.py \
        --train-file train.jsonl \
        --val-file val.jsonl \
        --output-dir ./models/bi_encoder \
        --model-name all-MiniLM-L6-v2 \
        --num-epochs 10 \
        --batch-size 16

    # Train with evaluation during training
    python train_bi_encoder.py \
        --train-file train.jsonl \
        --val-file val.jsonl \
        --eval-queries eval_queries.jsonl \
        --eval-qrels eval_qrels.txt \
        --eval-documents eval_docs.jsonl \
        --output-dir ./models/bi_encoder \
        --eval-during-training
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bi_encoder.src.models.bi_encoder import create_hybrid_bi_encoder
from bi_encoder.src.models.trainer import ContrastiveLoss, TripletLoss, BiEncoderTrainer
from bi_encoder.src.data.datasets import (
    create_bi_encoder_dataloader,
    BiEncoderDataset,
    InBatchNegativeDataset,
    PairwiseDataset
)
from bi_encoder.src.data.preprocessing import create_preprocessor
from bi_encoder.src.evaluation.dense_retrieval import create_end_to_end_evaluator
from cross_encoder.src.utils.file_utils import ensure_dir, load_json, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, TimedOperation, log_experiment_info

logger = logging.getLogger(__name__)


class BiEncoderTrainingPipeline:
    """
    Complete training pipeline for bi-encoder models.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 output_dir: Path):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration
            output_dir: Output directory for models and logs
        """
        self.config = config
        self.output_dir = ensure_dir(output_dir)

        # Initialize components
        self.model = None
        self.trainer = None
        self.train_dataloader = None
        self.val_dataloader = None

        # Evaluation components (if enabled)
        self.evaluator = None
        self.eval_queries = None
        self.eval_qrels = None
        self.eval_documents = None

        logger.info(f"BiEncoderTrainingPipeline initialized")
        logger.info(f"Output directory: {output_dir}")

    def create_model(self) -> None:
        """Create bi-encoder model."""
        logger.info("Creating bi-encoder model...")

        self.model = create_hybrid_bi_encoder(
            model_name=self.config['model_name'],
            max_expansion_terms=self.config['max_expansion_terms'],
            expansion_weight=self.config['expansion_weight'],
            similarity_function=self.config['similarity_function'],
            device=self.config.get('device'),
            force_hf=self.config.get('force_hf', False),
            pooling_strategy=self.config.get('pooling_strategy', 'cls')
        )

        # Log model info
        alpha, beta, exp_weight = self.model.get_learned_weights()
        logger.info(f"Model created: {self.config['model_name']}")
        logger.info(f"Initial weights: α={alpha:.4f}, β={beta:.4f}, expansion={exp_weight:.4f}")
        logger.info(f"Embedding dimension: {self.model.get_embedding_dimension()}")
        logger.info(f"Device: {self.model.device}")

    def load_data(self) -> None:
        """Load and preprocess training data."""
        logger.info("Loading training data...")

        # Load datasets
        train_data = self._load_jsonl(self.config['train_file'])
        val_data = None
        if self.config.get('val_file'):
            val_data = self._load_jsonl(self.config['val_file'])

        logger.info(f"Loaded {len(train_data)} training examples")
        if val_data:
            logger.info(f"Loaded {len(val_data)} validation examples")

        # Preprocess if enabled
        if self.config.get('preprocess_data', False):
            train_data, val_data = self._preprocess_data(train_data, val_data)

        # Create datasets
        dataset_type = self.config.get('dataset_type', 'contrastive')

        if dataset_type == 'contrastive':
            train_dataset = BiEncoderDataset(
                train_data,
                max_positives=self.config.get('max_positives', 3),
                max_negatives=self.config.get('max_negatives', 7),
                max_hard_negatives=self.config.get('max_hard_negatives', 5),
                include_hard_negatives=self.config.get('include_hard_negatives', True)
            )

            val_dataset = None
            if val_data:
                val_dataset = BiEncoderDataset(
                    val_data,
                    max_positives=self.config.get('max_positives', 3),
                    max_negatives=self.config.get('max_negatives', 7),
                    max_hard_negatives=self.config.get('max_hard_negatives', 5),
                    include_hard_negatives=self.config.get('include_hard_negatives', True)
                )

        elif dataset_type == 'in_batch':
            train_dataset = InBatchNegativeDataset(
                train_data,
                max_positives_per_query=self.config.get('max_positives_per_query', 1)
            )

            val_dataset = None
            if val_data:
                val_dataset = InBatchNegativeDataset(
                    val_data,
                    max_positives_per_query=self.config.get('max_positives_per_query', 1)
                )

        elif dataset_type == 'pairwise':
            train_dataset = PairwiseDataset(
                train_data,
                pairs_per_query=self.config.get('pairs_per_query', 5)
            )

            val_dataset = None
            if val_data:
                val_dataset = PairwiseDataset(
                    val_data,
                    pairs_per_query=self.config.get('pairs_per_query', 5)
                )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Create dataloaders
        self.train_dataloader = create_bi_encoder_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            dataset_type=dataset_type
        )

        if val_dataset:
            self.val_dataloader = create_bi_encoder_dataloader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 0),
                dataset_type=dataset_type
            )

        logger.info(f"Created {len(train_dataset)} training examples in {len(self.train_dataloader)} batches")
        if val_dataset:
            logger.info(f"Created {len(val_dataset)} validation examples in {len(self.val_dataloader)} batches")

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _preprocess_data(self, train_data: List[Dict], val_data: List[Dict] = None) -> Tuple[
        List[Dict], Optional[List[Dict]]]:
        """Preprocess training data."""
        logger.info("Preprocessing training data...")

        preprocessor = create_preprocessor(
            enable_augmentation=self.config.get('enable_augmentation', False),
            enable_filtering=self.config.get('enable_filtering', True),
            random_seed=self.config.get('random_seed', 42)
        )

        # Preprocess training data
        train_stats = preprocessor.analyze_dataset(train_data)
        logger.info(f"Training data analysis: {train_stats['total_examples']} examples")

        # Apply preprocessing
        train_preprocessed = preprocessor.preprocess_dataset(train_data)

        val_preprocessed = None
        if val_data:
            val_stats = preprocessor.analyze_dataset(val_data)
            logger.info(f"Validation data analysis: {val_stats['total_examples']} examples")
            val_preprocessed = preprocessor.preprocess_dataset(val_data)

        return train_preprocessed, val_preprocessed

    def setup_evaluation(self) -> None:
        """Setup evaluation components if enabled."""
        if not self.config.get('eval_during_training', False):
            return

        logger.info("Setting up evaluation components...")

        # Load evaluation data
        if self.config.get('eval_queries'):
            self.eval_queries = self._load_eval_queries(self.config['eval_queries'])

        if self.config.get('eval_qrels'):
            self.eval_qrels = self._load_eval_qrels(self.config['eval_qrels'])

        if self.config.get('eval_documents'):
            self.eval_documents = self._load_eval_documents(self.config['eval_documents'])

        # Create evaluator
        if self.eval_queries and self.eval_qrels and self.eval_documents:
            self.evaluator = create_end_to_end_evaluator(
                self.model,
                metrics=['ndcg_cut_10', 'ndcg_cut_20', 'map', 'recall_100']
            )

            # Setup index
            document_texts = list(self.eval_documents.values())
            document_ids = list(self.eval_documents.keys())

            self.evaluator.setup_retrieval_index(
                documents=document_texts,
                document_ids=document_ids,
                use_faiss=True,
                batch_size=self.config.get('eval_batch_size', 32)
            )

            logger.info(f"Evaluation setup complete:")
            logger.info(f"  Queries: {len(self.eval_queries)}")
            logger.info(f"  Documents: {len(self.eval_documents)}")
            logger.info(f"  Qrels: {len(self.eval_qrels)}")

    def _load_eval_queries(self, filepath: str) -> Dict[str, str]:
        """Load evaluation queries."""
        queries = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries[data['query_id']] = data['query_text']
        return queries

    def _load_eval_qrels(self, filepath: str) -> Dict[str, Dict[str, int]]:
        """Load evaluation qrels."""
        qrels = defaultdict(dict)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                    qrels[qid][doc_id] = rel
        return dict(qrels)

    def _load_eval_documents(self, filepath: str) -> Dict[str, str]:
        """Load evaluation documents."""
        documents = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'doc_id' in data and 'doc_text' in data:
                        documents[data['doc_id']] = data['doc_text']
                    elif 'id' in data and 'text' in data:
                        documents[data['id']] = data['text']
        return documents

    def create_trainer(self) -> None:
        """Create trainer."""
        logger.info("Creating trainer...")

        self.trainer = BiEncoderTrainer(
            model=self.model,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-6),
            batch_size=self.config['batch_size'],
            loss_type=self.config.get('loss_type', 'contrastive'),
            temperature=self.config.get('temperature', 0.05),
            margin=self.config.get('margin', 0.2),
            importance_weight_lr_multiplier=self.config.get('importance_weight_lr_multiplier', 10.0),
            warmup_steps=self.config.get('warmup_steps', 1000),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),
            device=self.config.get('device')
        )

        logger.info(f"Trainer created:")
        logger.info(f"  Loss type: {self.config.get('loss_type', 'contrastive')}")
        logger.info(f"  Learning rate: {self.config['learning_rate']}")
        logger.info(f"  Batch size: {self.config['batch_size']}")

    def train(self) -> Dict[str, Any]:
        """Run training."""
        logger.info("Starting training...")

        # Create custom training loop with evaluation
        history = {
            'train_loss': [],
            'val_loss': [],
            'alpha_values': [],
            'beta_values': [],
            'expansion_weight_values': [],
            'eval_scores': []
        }

        best_score = -1.0
        best_epoch = 0
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 5)

        for epoch in range(self.config['num_epochs']):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"EPOCH {epoch + 1}/{self.config['num_epochs']}")
            logger.info(f"{'=' * 50}")

            # Training epoch
            train_metrics = self.trainer.train_epoch(self.train_dataloader)
            history['train_loss'].append(train_metrics['train_loss'])
            history['alpha_values'].append(train_metrics['alpha'])
            history['beta_values'].append(train_metrics['beta'])

            if 'expansion_weight' in train_metrics:
                history['expansion_weight_values'].append(train_metrics['expansion_weight'])

            # Validation
            val_loss = 0.0
            if self.val_dataloader:
                val_metrics = self.trainer.evaluate(self.val_dataloader)
                val_loss = val_metrics['val_loss']
                history['val_loss'].append(val_loss)

            # Evaluation during training
            eval_score = 0.0
            if self.evaluator and (epoch + 1) % self.config.get('eval_frequency', 2) == 0:
                logger.info("Running evaluation...")

                eval_results = self.evaluator.run_retrieval_evaluation(
                    queries=self.eval_queries,
                    qrels=self.eval_qrels,
                    top_k=self.config.get('eval_top_k', 100)
                )

                eval_metric = self.config.get('eval_metric', 'ndcg_cut_10')
                eval_score = eval_results['evaluation_metrics'].get(eval_metric, 0.0)
                history['eval_scores'].append(eval_score)

                logger.info(f"Evaluation {eval_metric}: {eval_score:.4f}")

                # Save best model
                if eval_score > best_score:
                    best_score = eval_score
                    best_epoch = epoch + 1
                    patience_counter = 0

                    # Save best model
                    best_model_path = self.output_dir / 'best_model.pt'
                    torch.save(self.model.state_dict(), best_model_path)
                    logger.info(f"🏆 New best model saved! {eval_metric}={eval_score:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break

            # Log epoch summary
            logger.info(f"Epoch {epoch + 1} summary:")
            logger.info(f"  Train loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val loss: {val_loss:.4f}")
            logger.info(f"  Weights: α={train_metrics['alpha']:.4f}, β={train_metrics['beta']:.4f}")
            if eval_score > 0:
                logger.info(f"  Eval {self.config.get('eval_metric', 'ndcg_cut_10')}: {eval_score:.4f}")

        return history

    def save_model_and_results(self, history: Dict[str, Any]) -> None:
        """Save final model and training results."""
        logger.info("Saving model and results...")

        # Save final model
        final_model_path = self.output_dir / 'final_model.pt'
        torch.save(self.model.state_dict(), final_model_path)

        # Get final weights
        alpha, beta, exp_weight = self.model.get_learned_weights()

        # Create model info
        model_info = {
            'model_name': self.config['model_name'],
            'max_expansion_terms': self.config['max_expansion_terms'],
            'expansion_weight': self.config['expansion_weight'],
            'similarity_function': self.config['similarity_function'],
            'force_hf': self.config.get('force_hf', False),
            'pooling_strategy': self.config.get('pooling_strategy', 'cls'),
            'embedding_dimension': self.model.get_embedding_dimension(),
            'learned_weights': {
                'alpha': alpha,
                'beta': beta,
                'expansion_weight': exp_weight
            },
            'training_config': self.config,
            'final_model_path': str(final_model_path),
            'training_history': history
        }

        # Save model info
        model_info_path = self.output_dir / 'model_info.json'
        save_json(model_info, model_info_path)

        # Save training history
        history_path = self.output_dir / 'training_history.json'
        save_json(history, history_path)

        logger.info(f"Model saved to: {self.output_dir}")
        logger.info(f"Final weights: α={alpha:.4f}, β={beta:.4f}, expansion={exp_weight:.4f}")


def load_config_file(config_file: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return load_json(config_file)


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration dictionary from command line arguments."""
    config = {
        # Model configuration
        'model_name': args.model_name,
        'max_expansion_terms': args.max_expansion_terms,
        'expansion_weight': args.expansion_weight,
        'similarity_function': args.similarity_function,
        'force_hf': args.force_hf,
        'pooling_strategy': args.pooling_strategy,

        # Data configuration
        'train_file': args.train_file,
        'val_file': args.val_file,
        'dataset_type': args.dataset_type,
        'preprocess_data': args.preprocess_data,

        # Training configuration
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'temperature': args.temperature,
        'margin': args.margin,
        'importance_weight_lr_multiplier': args.importance_weight_lr_multiplier,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        'early_stopping_patience': args.early_stopping_patience,

        # Dataset configuration
        'max_positives': args.max_positives,
        'max_negatives': args.max_negatives,
        'max_hard_negatives': args.max_hard_negatives,
        'include_hard_negatives': args.include_hard_negatives,

        # Evaluation configuration
        'eval_during_training': args.eval_during_training,
        'eval_frequency': args.eval_frequency,
        'eval_metric': args.eval_metric,
        'eval_top_k': args.eval_top_k,
        'eval_batch_size': args.eval_batch_size,

        # Other configuration
        'device': args.device,
        'num_workers': args.num_workers,
        'random_seed': args.random_seed
    }

    # Add evaluation files if provided
    if args.eval_queries:
        config['eval_queries'] = args.eval_queries
    if args.eval_qrels:
        config['eval_qrels'] = args.eval_qrels
    if args.eval_documents:
        config['eval_documents'] = args.eval_documents

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train bi-encoder for dense retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file')

    # Data arguments
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training data JSONL file')
    parser.add_argument('--val-file', type=str,
                        help='Path to validation data JSONL file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained model')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Base model name')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms')
    parser.add_argument('--expansion-weight', type=float, default=0.3,
                        help='Weight for expansion terms')
    parser.add_argument('--similarity-function', type=str, default='cosine',
                        choices=['cosine', 'dot_product', 'learned', 'bilinear'],
                        help='Similarity function')
    parser.add_argument('--force-hf', action='store_true',
                        help='Force using HuggingFace transformers')
    parser.add_argument('--pooling-strategy', type=str, default='cls',
                        choices=['cls', 'mean', 'max'],
                        help='Pooling strategy for HF models')

    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--loss-type', type=str, default='contrastive',
                        choices=['contrastive', 'triplet'],
                        help='Loss function type')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for contrastive loss')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin for triplet loss')
    parser.add_argument('--importance-weight-lr-multiplier', type=float, default=10.0,
                        help='Learning rate multiplier for importance weights')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Early stopping patience')

    # Dataset arguments
    parser.add_argument('--dataset-type', type=str, default='contrastive',
                        choices=['contrastive', 'in_batch', 'pairwise'],
                        help='Dataset type for training')
    parser.add_argument('--max-positives', type=int, default=3,
                        help='Maximum positive documents per query')
    parser.add_argument('--max-negatives', type=int, default=7,
                        help='Maximum negative documents per query')
    parser.add_argument('--max-hard-negatives', type=int, default=5,
                        help='Maximum hard negatives per query')
    parser.add_argument('--include-hard-negatives', action='store_true',
                        help='Include hard negatives in training')
    parser.add_argument('--preprocess-data', action='store_true',
                        help='Enable data preprocessing')

    # Evaluation arguments
    parser.add_argument('--eval-during-training', action='store_true',
                        help='Enable evaluation during training')
    parser.add_argument('--eval-queries', type=str,
                        help='Path to evaluation queries JSONL file')
    parser.add_argument('--eval-qrels', type=str,
                        help='Path to evaluation qrels file')
    parser.add_argument('--eval-documents', type=str,
                        help='Path to evaluation documents JSONL file')
    parser.add_argument('--eval-frequency', type=int, default=2,
                        help='Evaluate every N epochs')
    parser.add_argument('--eval-metric', type=str, default='ndcg_cut_10',
                        help='Metric for best model selection')
    parser.add_argument('--eval-top-k', type=int, default=100,
                        help='Top-k for evaluation')
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='Batch size for evaluation')

    # Other arguments
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'],
                        help='Device for training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config_file(Path(args.config))
        print(f"Loaded configuration from: {args.config}")
    else:
        config = create_config_from_args(args)

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("train_bi_encoder", args.log_level,
                                      str(output_dir / 'training.log'))
    log_experiment_info(logger, **config)

    try:
        # Set random seed
        torch.manual_seed(config['random_seed'])

        # Initialize training pipeline
        pipeline = BiEncoderTrainingPipeline(config, output_dir)

        # Create model
        with TimedOperation(logger, "Creating model"):
            pipeline.create_model()

        # Load data
        with TimedOperation(logger, "Loading and preprocessing data"):
            pipeline.load_data()

        # Setup evaluation if enabled
        if config.get('eval_during_training', False):
            with TimedOperation(logger, "Setting up evaluation"):
                pipeline.setup_evaluation()

        # Create trainer
        with TimedOperation(logger, "Creating trainer"):
            pipeline.create_trainer()

        # Train model
        with TimedOperation(logger, f"Training for {config['num_epochs']} epochs"):
            history = pipeline.train()

        # Save results
        with TimedOperation(logger, "Saving model and results"):
            pipeline.save_model_and_results(history)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Training epochs: {len(history['train_loss'])}")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")

        if history['eval_scores']:
            best_eval = max(history['eval_scores'])
            logger.info(f"Best eval {config.get('eval_metric', 'ndcg_cut_10')}: {best_eval:.4f}")

        # Show learned weights
        alpha, beta, exp_weight = pipeline.model.get_learned_weights()
        logger.info(f"Final learned weights:")
        logger.info(f"  α (RM3): {alpha:.4f}")
        logger.info(f"  β (Semantic): {beta:.4f}")
        logger.info(f"  Expansion weight: {exp_weight:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()