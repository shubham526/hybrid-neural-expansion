"""
Neural Training Pipeline for Importance-Weighted Reranking
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ExpansionDataset(Dataset):
    """Dataset for neural reranking training."""

    def __init__(self,
                 features: Dict[str, Dict[str, Any]],
                 queries: Dict[str, str],
                 qrels: Dict[str, Dict[str, int]],
                 first_stage_runs: Dict[str, List[Tuple[str, float]]],
                 documents: Dict[str, str] = None,
                 max_candidates_per_query: int = 100):
        """
        Initialize dataset.

        Args:
            features: Extracted expansion features
            queries: Query texts
            qrels: Relevance judgments
            first_stage_runs: First-stage retrieval results
            documents: Document collection (optional, for document-aware models)
            max_candidates_per_query: Limit candidates per query
        """
        self.features = features
        self.queries = queries
        self.qrels = qrels
        self.first_stage_runs = first_stage_runs
        self.documents = documents or {}
        self.max_candidates = max_candidates_per_query

        # Create training examples
        self.examples = self._create_training_examples()

        logger.info(f"Created dataset with {len(self.examples)} training examples")

    def _create_training_examples(self) -> List[Dict[str, Any]]:
        """Create training examples from data."""
        examples = []

        for query_id in self.features:
            if (query_id not in self.queries or
                    query_id not in self.qrels or
                    query_id not in self.first_stage_runs):
                continue

            query_text = self.queries[query_id]
            query_qrels = self.qrels[query_id]
            candidates = self.first_stage_runs[query_id][:self.max_candidates]
            expansion_features = self.features[query_id]['term_features']

            # Create one example per query (query-level scoring)
            # For document-aware models, you'd create one example per (query, doc) pair

            # Aggregate relevance for query-level target
            relevance_scores = [query_qrels.get(doc_id, 0) for doc_id, _ in candidates]
            avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0

            example = {
                'query_id': query_id,
                'query_text': query_text,
                'expansion_features': expansion_features,
                'target_relevance': float(avg_relevance),
                'num_candidates': len(candidates)
            }

            examples.append(example)

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DocumentAwareDataset(ExpansionDataset):
    """Dataset for document-aware neural reranking."""

    def _create_training_examples(self) -> List[Dict[str, Any]]:
        """Create (query, document) pair examples."""
        examples = []

        for query_id in self.features:
            if (query_id not in self.queries or
                    query_id not in self.qrels or
                    query_id not in self.first_stage_runs):
                continue

            query_text = self.queries[query_id]
            query_qrels = self.qrels[query_id]
            candidates = self.first_stage_runs[query_id][:self.max_candidates]
            expansion_features = self.features[query_id]['term_features']

            # Create one example per (query, document) pair
            for doc_id, first_stage_score in candidates:
                if doc_id not in self.documents:
                    continue

                doc_text = self.documents[doc_id]
                relevance = query_qrels.get(doc_id, 0)

                example = {
                    'query_id': query_id,
                    'query_text': query_text,
                    'expansion_features': expansion_features,
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'first_stage_score': first_stage_score,
                    'target_relevance': float(relevance)
                }

                examples.append(example)

        return examples


class NeuralTrainer:
    """Trainer for neural reranking models."""

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 use_ranking_loss: bool = False):
        """
        Initialize trainer.

        Args:
            model: Neural reranker model
            learning_rate: Learning rate
            weight_decay: Weight decay
            batch_size: Batch size
            use_ranking_loss: Whether to use ranking loss vs regression loss
        """
        self.model = model
        self.batch_size = batch_size
        self.use_ranking_loss = use_ranking_loss

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        if use_ranking_loss:
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        else:
            self.criterion = nn.MSELoss()

        logger.info(f"NeuralTrainer initialized:")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Loss type: {'ranking' if use_ranking_loss else 'regression'}")

    def train_epoch(self, dataset: Dataset) -> float:
        """Train for one epoch."""
        self.model.train()

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size 1 for simplicity
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

            try:
                # Forward pass
                if hasattr(self.model, 'encode_document') and 'doc_text' in example:
                    # Document-aware model
                    predicted_score = self.model.forward(
                        query=example['query_text'],
                        expansion_features=example['expansion_features'],
                        document=example['doc_text']
                    )
                else:
                    # Query-only model
                    predicted_score = self.model.forward(
                        query=example['query_text'],
                        expansion_features=example['expansion_features']
                    )

                # Target
                target_score = torch.tensor(
                    float(example['target_relevance']),
                    device=self.model.device
                )

                # Loss
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
            train_loss = self.train_epoch(train_dataset)
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
                    # Forward pass
                    if hasattr(self.model, 'encode_document') and 'doc_text' in example:
                        predicted_score = self.model.forward(
                            query=example['query_text'],
                            expansion_features=example['expansion_features'],
                            document=example['doc_text']
                        )
                    else:
                        predicted_score = self.model.forward(
                            query=example['query_text'],
                            expansion_features=example['expansion_features']
                        )

                    target_score = torch.tensor(
                        float(example['target_relevance']),
                        device=self.model.device
                    )

                    loss = self.criterion(predicted_score, target_score)
                    total_loss += loss.item()
                    num_examples += 1

                except Exception as e:
                    logger.warning(f"Error in evaluation: {e}")
                    continue

        return total_loss / num_examples if num_examples > 0 else 0.0