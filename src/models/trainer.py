"""
Neural Training Pipeline for Importance-Weighted Reranking

Updated for the new reranker that requires document content.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
from src.utils.data_utils import DocumentAwareExpansionDataset, PairwiseExpansionDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the new document-aware neural reranking models."""

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 8,  # Smaller batch size due to document content
                 loss_type: str = "mse"):  # "mse", "ranking", or "bce"
        """
        Initialize trainer.

        Args:
            model: Neural reranker model
            learning_rate: Learning rate
            weight_decay: Weight decay
            batch_size: Batch size
            loss_type: Type of loss function
        """
        self.model = model
        self.batch_size = batch_size
        self.loss_type = loss_type

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "ranking":
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logger.info(f"NeuralTrainer initialized:")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Loss type: {loss_type}")

    def train_epoch_pointwise(self, dataset: DocumentAwareExpansionDataset) -> float:
        """Train for one epoch using pointwise loss."""
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 for simplicity
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

            try:
                # Forward pass
                predicted_score = self.model.forward(
                    query=example['query_text'],
                    expansion_features=example['expansion_features'],
                    document=example['doc_text']
                )

                # Target relevance score
                if self.loss_type == "mse":
                    # Use relevance as continuous target
                    target_score = torch.tensor(
                        float(example['relevance']),
                        device=self.model.device
                    )
                elif self.loss_type == "bce":
                    # Binary classification (relevant/not relevant)
                    target_score = torch.tensor(
                        1.0 if float(example['relevance']) > 0 else 0.0,
                        device=self.model.device
                    )

                # Compute loss
                loss = self.criterion(predicted_score, target_score)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error in training batch: {e}")
                continue

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train_epoch_pairwise(self, dataset: PairwiseExpansionDataset) -> float:
        """Train for one epoch using pairwise ranking loss."""
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Pairwise training"):
            example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

            try:
                # Score positive document
                pos_score = self.model.forward(
                    query=example['query_text'],
                    expansion_features=example['expansion_features'],
                    document=example['positive_doc']['doc_text']
                )

                # Score negative document
                neg_score = self.model.forward(
                    query=example['query_text'],
                    expansion_features=example['expansion_features'],
                    document=example['negative_doc']['doc_text']
                )

                # Ranking loss (positive should score higher than negative)
                target = torch.tensor(1.0, device=self.model.device)  # pos > neg
                loss = self.criterion(pos_score, neg_score, target)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error in pairwise training batch: {e}")
                continue

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset = None,
              num_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of epochs

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'alpha_values': [],
            'beta_values': []
        }

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            if isinstance(train_dataset, PairwiseExpansionDataset):
                train_loss = self.train_epoch_pairwise(train_dataset)
            else:
                train_loss = self.train_epoch_pointwise(train_dataset)

            history['train_loss'].append(train_loss)

            # Track learned weights
            if hasattr(self.model, 'get_learned_weights'):
                alpha, beta = self.model.get_learned_weights()
                history['alpha_values'].append(alpha)
                history['beta_values'].append(beta)

                logger.info(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, α={alpha:.3f}, β={beta:.3f}")
            else:
                logger.info(f"Epoch {epoch + 1}: Loss={train_loss:.4f}")

            # Validation
            if val_dataset:
                val_loss = self.evaluate(val_dataset)
                history['val_loss'].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")

        return history

    def evaluate(self, dataset: Dataset) -> float:
        """Evaluate model on dataset."""
        self.model.eval()

        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            for batch in dataloader:
                example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

                try:
                    if isinstance(dataset, PairwiseExpansionDataset):
                        # Pairwise evaluation
                        pos_score = self.model.forward(
                            query=example['query_text'],
                            expansion_features=example['expansion_features'],
                            document=example['positive_doc']['doc_text']
                        )

                        neg_score = self.model.forward(
                            query=example['query_text'],
                            expansion_features=example['expansion_features'],
                            document=example['negative_doc']['doc_text']
                        )

                        target = torch.tensor(1.0, device=self.model.device)
                        loss = self.criterion(pos_score, neg_score, target)

                    else:
                        # Pointwise evaluation
                        predicted_score = self.model.forward(
                            query=example['query_text'],
                            expansion_features=example['expansion_features'],
                            document=example['doc_text']
                        )

                        if self.loss_type == "mse":
                            target_score = torch.tensor(
                                float(example['relevance']),
                                device=self.model.device
                            )
                        elif self.loss_type == "bce":
                            target_score = torch.tensor(
                                1.0 if float(example['relevance']) > 0 else 0.0,
                                device=self.model.device
                            )

                        loss = self.criterion(predicted_score, target_score)

                    total_loss += loss.item()
                    num_examples += 1

                except Exception as e:
                    logger.warning(f"Error in evaluation: {e}")
                    continue

        return total_loss / num_examples if num_examples > 0 else 0.0