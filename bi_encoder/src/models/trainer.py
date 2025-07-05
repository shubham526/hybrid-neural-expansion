"""
Bi-Encoder Trainer with Contrastive Learning

This trainer implements contrastive learning for the hybrid bi-encoder model,
optimizing the separation between positive and negative query-document pairs
in the embedding space.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from pathlib import Path
import numpy as np
import time

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.

    This is the standard loss for bi-encoder training, encouraging positive pairs
    to have higher similarity than negative pairs.
    """

    def __init__(self, temperature: float = 0.05):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"ContrastiveLoss initialized with temperature={temperature}")

    def forward(self,
                query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor = None,
                similarity_computer=None) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            query_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, num_negatives, embedding_dim] or None
            similarity_computer: Similarity function to use

        Returns:
            Contrastive loss value
        """
        batch_size = query_embeddings.size(0)

        if negative_embeddings is None:
            # In-batch negatives: use other positives as negatives
            return self._in_batch_contrastive_loss(
                query_embeddings, positive_embeddings, similarity_computer
            )
        else:
            # Explicit negatives provided
            return self._explicit_negatives_loss(
                query_embeddings, positive_embeddings, negative_embeddings, similarity_computer
            )

    def _in_batch_contrastive_loss(self,
                                   query_embeddings: torch.Tensor,
                                   positive_embeddings: torch.Tensor,
                                   similarity_computer) -> torch.Tensor:
        """Compute loss using in-batch negatives."""
        # Compute similarities between all queries and all positives
        if similarity_computer is not None:
            # Use custom similarity function
            similarities = []
            for i in range(query_embeddings.size(0)):
                query_emb = query_embeddings[i]
                sim_scores = similarity_computer.compute_similarity(query_emb, positive_embeddings)
                similarities.append(sim_scores)
            similarities = torch.stack(similarities)  # [batch_size, batch_size]
        else:
            # Default to cosine similarity
            similarities = F.cosine_similarity(
                query_embeddings.unsqueeze(1),  # [batch_size, 1, embedding_dim]
                positive_embeddings.unsqueeze(0),  # [1, batch_size, embedding_dim]
                dim=2
            )  # [batch_size, batch_size]

        # Scale by temperature
        similarities = similarities / self.temperature

        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(similarities.size(0), device=similarities.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarities, labels)

        return loss

    def _explicit_negatives_loss(self,
                                 query_embeddings: torch.Tensor,
                                 positive_embeddings: torch.Tensor,
                                 negative_embeddings: torch.Tensor,
                                 similarity_computer) -> torch.Tensor:
        """Compute loss with explicit negative samples."""
        batch_size = query_embeddings.size(0)
        num_negatives = negative_embeddings.size(1)

        losses = []

        for i in range(batch_size):
            query_emb = query_embeddings[i]  # [embedding_dim]
            pos_emb = positive_embeddings[i]  # [embedding_dim]
            neg_embs = negative_embeddings[i]  # [num_negatives, embedding_dim]

            # Combine positive and negative embeddings
            all_docs = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)  # [1 + num_negatives, embedding_dim]

            # Compute similarities
            if similarity_computer is not None:
                similarities = similarity_computer.compute_similarity(query_emb, all_docs)
            else:
                similarities = F.cosine_similarity(
                    query_emb.unsqueeze(0), all_docs, dim=1
                )

            # Scale by temperature
            similarities = similarities / self.temperature

            # First document is positive (label = 0)
            label = torch.tensor(0, device=similarities.device)

            # Compute cross-entropy loss for this query
            query_loss = F.cross_entropy(similarities.unsqueeze(0), label.unsqueeze(0))
            losses.append(query_loss)

        return torch.stack(losses).mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for bi-encoder training.

    Alternative to InfoNCE that directly optimizes the margin between
    positive and negative similarities.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        logger.info(f"TripletLoss initialized with margin={margin}")

    def forward(self,
                query_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            query_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, embedding_dim] (single negative per query)

        Returns:
            Triplet loss value
        """
        if negative_embeddings.dim() == 3:
            # Multiple negatives: take the first one or average
            negative_embeddings = negative_embeddings[:, 0, :]  # Take first negative

        return self.triplet_loss(query_embeddings, positive_embeddings, negative_embeddings)


class BiEncoderTrainer:
    """
    Trainer for hybrid bi-encoder model with contrastive learning.
    """

    def __init__(self,
                 model,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 1e-6,
                 batch_size: int = 16,
                 loss_type: str = "contrastive",
                 temperature: float = 0.05,
                 margin: float = 0.2,
                 importance_weight_lr_multiplier: float = 10.0,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 device: str = None):
        """
        Initialize bi-encoder trainer.

        Args:
            model: BiEncoder model to train
            learning_rate: Learning rate for model parameters
            weight_decay: Weight decay for regularization
            batch_size: Training batch size
            loss_type: Loss function type ('contrastive', 'triplet')
            temperature: Temperature for contrastive loss
            margin: Margin for triplet loss
            importance_weight_lr_multiplier: LR multiplier for α, β parameters
            warmup_steps: Number of warmup steps for learning rate
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
        """
        self.model = model
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup loss function
        if loss_type == "contrastive":
            self.criterion = ContrastiveLoss(temperature=temperature)
        elif loss_type == "triplet":
            self.criterion = TripletLoss(margin=margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Setup optimizer with different learning rates
        self._setup_optimizer(learning_rate, weight_decay, importance_weight_lr_multiplier)

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_lr_lambda
        )

        # Training state
        self.step = 0
        self.epoch = 0

        logger.info(f"BiEncoderTrainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Temperature: {temperature}")

    def _setup_optimizer(self, learning_rate: float, weight_decay: float,
                         importance_weight_lr_multiplier: float):
        """Setup optimizer with different learning rates for different parameters."""
        # Separate parameters into groups
        importance_params = []
        similarity_params = []
        encoder_params = []

        for name, param in self.model.named_parameters():
            if name in ['alpha', 'beta', 'learnable_expansion_weight']:
                importance_params.append(param)
                logger.info(f"Adding {name} to importance weight group")
            elif 'similarity_computer' in name:
                similarity_params.append(param)
                logger.info(f"Adding {name} to similarity function group")
            else:
                encoder_params.append(param)

        # Create parameter groups
        param_groups = []

        # Importance weights with higher learning rate
        if importance_params:
            param_groups.append({
                'params': importance_params,
                'lr': learning_rate * importance_weight_lr_multiplier,
                'weight_decay': 0.0,  # No weight decay for importance weights
                'name': 'importance_weights'
            })

        # Similarity function parameters
        if similarity_params:
            param_groups.append({
                'params': similarity_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'name': 'similarity_function'
            })

        # Encoder parameters (usually frozen or very low LR)
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': learning_rate * 0.1,  # Lower LR for pre-trained encoder
                'weight_decay': weight_decay,
                'name': 'encoder'
            })

        self.optimizer = optim.AdamW(param_groups)

        logger.info(f"Optimizer setup complete with {len(param_groups)} parameter groups")

    def _get_lr_lambda(self, step):
        """Learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps
        else:
            # Cosine decay after warmup
            decay_steps = max(1, step - self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * decay_steps / (10 * self.warmup_steps)))

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        if hasattr(self.model, 'similarity_computer'):
            self.model.similarity_computer.train()

        total_loss = 0.0
        num_batches = 0

        # Track importance weights
        epoch_start_weights = self.model.get_learned_weights()

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                loss = self._forward_pass(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.step += 1

                # Update progress bar
                current_loss = total_loss / num_batches
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': self.step
                })

            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_end_weights = self.model.get_learned_weights()

        # Log weight changes
        alpha_change = epoch_end_weights[0] - epoch_start_weights[0]
        beta_change = epoch_end_weights[1] - epoch_start_weights[1]

        metrics = {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'alpha': epoch_end_weights[0],
            'beta': epoch_end_weights[1],
            'alpha_change': alpha_change,
            'beta_change': beta_change,
            'num_batches': num_batches
        }

        # Add expansion weight if available
        if len(epoch_end_weights) > 2:
            exp_weight_change = epoch_end_weights[2] - epoch_start_weights[2]
            metrics['expansion_weight'] = epoch_end_weights[2]
            metrics['expansion_weight_change'] = exp_weight_change

        self.epoch += 1

        logger.info(f"Epoch {self.epoch} completed:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Weight changes: Δα={alpha_change:+.4f}, Δβ={beta_change:+.4f}")

        return metrics

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform forward pass and compute loss.

        Expected batch format:
        {
            'query_texts': List[str],
            'expansion_features': List[Dict],
            'positive_docs': List[str] or torch.Tensor,
            'negative_docs': List[str] or torch.Tensor (optional)
        }
        """
        query_texts = batch['query_texts']
        expansion_features_list = batch['expansion_features']
        positive_docs = batch['positive_docs']
        negative_docs = batch.get('negative_docs', None)

        batch_size = len(query_texts)

        # Encode queries with expansion
        query_embeddings = []
        for i in range(batch_size):
            query_emb = self.model.encode_query(query_texts[i], expansion_features_list[i])
            query_embeddings.append(query_emb)
        query_embeddings = torch.stack(query_embeddings)  # [batch_size, embedding_dim]

        # Encode positive documents
        if isinstance(positive_docs, list):
            positive_embeddings = self.model.encode_documents(positive_docs)
        else:
            positive_embeddings = positive_docs  # Already encoded

        # Encode negative documents if provided
        negative_embeddings = None
        if negative_docs is not None:
            if isinstance(negative_docs, list):
                # Flatten if nested list (multiple negatives per query)
                if isinstance(negative_docs[0], list):
                    flat_negatives = [doc for doc_list in negative_docs for doc in doc_list]
                    neg_embs = self.model.encode_documents(flat_negatives)
                    # Reshape to [batch_size, num_negatives, embedding_dim]
                    num_negatives = len(negative_docs[0])
                    negative_embeddings = neg_embs.view(batch_size, num_negatives, -1)
                else:
                    negative_embeddings = self.model.encode_documents(negative_docs)
            else:
                negative_embeddings = negative_docs  # Already encoded

        # Compute loss
        similarity_computer = getattr(self.model, 'similarity_computer', None)
        loss = self.criterion(
            query_embeddings,
            positive_embeddings,
            negative_embeddings,
            similarity_computer
        )

        return loss

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        if hasattr(self.model, 'similarity_computer'):
            self.model.similarity_computer.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    batch = self._move_batch_to_device(batch)
                    loss = self._forward_pass(batch)

                    total_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader = None,
              num_epochs: int = 10,
              save_dir: Optional[Path] = None,
              save_every: int = 1) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'alpha_values': [],
            'beta_values': [],
            'expansion_weight_values': []
        }

        # Initial weights
        initial_weights = self.model.get_learned_weights()
        logger.info(f"Initial weights: α={initial_weights[0]:.4f}, β={initial_weights[1]:.4f}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'=' * 50}")

            # Training
            train_metrics = self.train_epoch(train_dataloader)
            history['train_loss'].append(train_metrics['train_loss'])
            history['alpha_values'].append(train_metrics['alpha'])
            history['beta_values'].append(train_metrics['beta'])

            if 'expansion_weight' in train_metrics:
                history['expansion_weight_values'].append(train_metrics['expansion_weight'])

            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history['val_loss'].append(val_metrics['val_loss'])

                logger.info(f"Validation loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    if save_dir:
                        self.save_checkpoint(save_dir / 'best_model.pt', epoch, train_metrics, val_metrics)

            # Save periodic checkpoint
            if save_dir and (epoch + 1) % save_every == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pt'
                self.save_checkpoint(checkpoint_path, epoch, train_metrics,
                                     val_metrics if val_dataloader else None)

        # Final summary
        final_weights = self.model.get_learned_weights()
        logger.info(f"\nTraining completed!")
        logger.info(f"Initial weights: α={initial_weights[0]:.4f}, β={initial_weights[1]:.4f}")
        logger.info(f"Final weights: α={final_weights[0]:.4f}, β={final_weights[1]:.4f}")
        logger.info(
            f"Total change: Δα={final_weights[0] - initial_weights[0]:+.4f}, Δβ={final_weights[1] - initial_weights[1]:+.4f}")

        return history

    def save_checkpoint(self, filepath: Path, epoch: int,
                        train_metrics: Dict, val_metrics: Dict = None):
        """Save model checkpoint."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'step': self.step,
            'learned_weights': self.model.get_learned_weights()
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']

        logger.info(f"Loaded checkpoint from epoch {self.epoch}: {filepath}")
        return checkpoint


# Factory function
def create_bi_encoder_trainer(model,
                              learning_rate: float = 2e-5,
                              batch_size: int = 16,
                              loss_type: str = "contrastive",
                              **kwargs) -> BiEncoderTrainer:
    """
    Factory function to create bi-encoder trainer.

    Args:
        model: BiEncoder model
        learning_rate: Learning rate
        batch_size: Batch size
        loss_type: Loss function type
        **kwargs: Additional trainer arguments

    Returns:
        Configured trainer
    """
    return BiEncoderTrainer(
        model=model,
        learning_rate=learning_rate,
        batch_size=batch_size,
        loss_type=loss_type,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing bi-encoder trainer...")


    # This would normally come from your bi_encoder module
    # from bi_encoder.src.models.bi_encoder import create_hybrid_bi_encoder

    # Create mock model for testing
    class MockBiEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(0.6))
            self.beta = nn.Parameter(torch.tensor(0.4))
            self.embedding_dim = 384

        def encode_query(self, query, features=None):
            return torch.randn(self.embedding_dim)

        def encode_documents(self, docs):
            return torch.randn(len(docs), self.embedding_dim)

        def get_learned_weights(self):
            return self.alpha.item(), self.beta.item()


    # Create trainer
    model = MockBiEncoder()
    trainer = create_bi_encoder_trainer(
        model=model,
        learning_rate=2e-5,
        batch_size=4,
        loss_type="contrastive"
    )

    print(f"Trainer created successfully!")
    print(f"Device: {trainer.device}")
    print(f"Loss type: {trainer.loss_type}")

    # Test loss computation
    query_embs = torch.randn(4, 384)
    pos_embs = torch.randn(4, 384)

    loss = trainer.criterion(query_embs, pos_embs)
    print(f"Test loss: {loss.item():.4f}")

    print("Bi-encoder trainer test completed!")