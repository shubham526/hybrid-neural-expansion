#!/usr/bin/env python3
"""
Modular NPRF Trainer

Clean, modular training script using the refactored components.
"""

import argparse
import json
import logging
import time
import random
import numpy as np
import torch
from pathlib import Path

from nprf_core import (
    SimilarityComputer, NPRFFeatureExtractor, ModelFactory, hinge_loss
)
from nprf_data import NPRFDataLoader, load_and_preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NPRFTrainer:
    """Modular NPRF trainer."""
    
    def __init__(self, model, feature_extractor, device=None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader=None, num_epochs=30, learning_rate=0.001):
        """Train the NPRF model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in train_loader:
            optimizer.zero_grad()
            
            pos_features, neg_features = batch_data
            
            # Forward pass
            pos_scores = self._forward_batch(pos_features)
            neg_scores = self._forward_batch(neg_features)
            
            # Compute loss
            loss = hinge_loss(pos_scores, neg_scores)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                pos_features, neg_features = batch_data
                
                # Forward pass
                pos_scores = self._forward_batch(pos_features)
                neg_scores = self._forward_batch(neg_features)
                
                # Compute loss
                loss = hinge_loss(pos_scores, neg_scores)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _forward_batch(self, features):
        """Forward pass for a batch of features."""
        if len(features) == 3:  # DRMM
            dd_q, dd_d, doc_scores = features
            dd_q = dd_q.to(self.device)
            dd_d = dd_d.to(self.device)
            doc_scores = doc_scores.to(self.device)
            return self.model(dd_q, dd_d, doc_scores)
        else:  # K-NRM
            dd, doc_scores = features
            dd = dd.to(self.device)
            doc_scores = doc_scores.to(self.device)
            return self.model(dd, doc_scores)
    
    def save_model(self, output_path, args, feature_extractor):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': args.model_type,
            'args': vars(args),
            'feature_extractor_config': {
                'model_type': feature_extractor.model_type,
                'nb_supervised_doc': feature_extractor.nb_supervised_doc,
                'doc_topk_term': feature_extractor.doc_topk_term
            }
        }, output_path)
        logger.info(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train NPRF models (modular version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training data JSONL file')
    parser.add_argument('--val-file', type=str,
                        help='Path to validation data JSONL file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained models')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='drmm',
                        choices=['drmm', 'knrm'],
                        help='Type of neural IR model')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--sample-size', type=int, default=10,
                        help='Number of negative samples per positive')
    
    # Model hyperparameters
    parser.add_argument('--nb-supervised-doc', type=int, default=10,
                        help='Number of PRF documents')
    parser.add_argument('--doc-topk-term', type=int, default=20,
                        help='Number of top terms per document')
    parser.add_argument('--hist-size', type=int, default=30,
                        help='Histogram size for DRMM')
    parser.add_argument('--kernel-size', type=int, default=11,
                        help='Number of kernels for K-NRM')
    parser.add_argument('--hidden-size', type=int, default=5,
                        help='Hidden layer size')
    
    # Data preprocessing
    parser.add_argument('--min-candidates', type=int, default=5,
                        help='Minimum candidates per query')
    parser.add_argument('--balance-dataset', action='store_true',
                        help='Balance training dataset by relevance')
    parser.add_argument('--max-queries-per-class', type=int,
                        help='Max queries per relevance class (for balancing)')
    
    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    preprocess_config = {
        'min_candidates': args.min_candidates,
        'require_positive': True,
        'require_negative': True,
        'balance_dataset': args.balance_dataset,
        'max_queries_per_class': args.max_queries_per_class
    }
    
    datasets = load_and_preprocess_data(
        args.train_file, 
        args.val_file, 
        preprocess_config=preprocess_config
    )
    
    train_data = datasets['train']
    val_data = datasets.get('val')
    
    logger.info(f"Training data: {len(train_data)} queries")
    if val_data:
        logger.info(f"Validation data: {len(val_data)} queries")
    
    # Initialize components
    logger.info("Initializing NPRF components...")
    similarity_computer = SimilarityComputer(device=device)
    
    feature_extractor = NPRFFeatureExtractor(
        model_type=args.model_type,
        similarity_computer=similarity_computer,
        nb_supervised_doc=args.nb_supervised_doc,
        doc_topk_term=args.doc_topk_term,
        hist_size=args.hist_size,
        kernel_size=args.kernel_size
    )
    
    model = ModelFactory.create_model(
        args.model_type,
        hist_size=args.hist_size,
        kernel_size=args.kernel_size,
        hidden_size=args.hidden_size,
        nb_supervised_doc=args.nb_supervised_doc,
        doc_topk_term=args.doc_topk_term
    )
    
    logger.info(f"Model architecture:\n{model}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = NPRFDataLoader.create_train_loader(
        train_data, feature_extractor, 
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        num_workers=args.num_workers
    )
    
    val_loader = None
    if val_data:
        val_loader = NPRFDataLoader.create_train_loader(
            val_data, feature_extractor,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            num_workers=args.num_workers,
            shuffle=False
        )
    
    # Initialize trainer
    trainer = NPRFTrainer(model, feature_extractor, device)
    
    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    
    trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = output_dir / f"nprf_{args.model_type}_model.pt"
    trainer.save_model(model_path, args, feature_extractor)
    
    # Save training configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info(f"Training configuration saved to: {config_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()