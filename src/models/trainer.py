"""
Broadcasting-Fixed Trainer

Fixes the "output with shape [] doesn't match the broadcast shape [1]" error
by ensuring proper tensor shapes throughout the training process.
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


class BroadcastFixedTrainer:
    """Trainer with explicit tensor shape handling to fix broadcasting errors."""

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 batch_size: int = 8,
                 loss_type: str = "mse",
                 importance_weight_lr: float = None,
                 debug_frequency: int = 10):
        """Initialize trainer with broadcasting fixes."""

        self.model = model
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.debug_frequency = debug_frequency

        # Setup optimizers
        if importance_weight_lr is None:
            importance_weight_lr = learning_rate * 10

        importance_params = []
        other_params = []

        for name, param in model.named_parameters():
            if name in ['alpha', 'beta']:
                importance_params.append(param)
                logger.info(f"Adding {name} to importance weight group (LR: {importance_weight_lr})")
            else:
                other_params.append(param)

        param_groups = []
        if importance_params:
            param_groups.append({
                'params': importance_params,
                'lr': importance_weight_lr,
                'weight_decay': 0.0,
                'name': 'importance_weights'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'name': 'neural_network'
            })

        self.optimizer = optim.Adam(param_groups)

        # Loss function with proper shape handling
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "ranking":
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logger.info(f"BroadcastFixedTrainer initialized:")
        logger.info(f"  Base learning rate: {learning_rate}")
        logger.info(f"  Importance weight LR: {importance_weight_lr}")
        logger.info(f"  Loss type: {loss_type}")

    def safe_forward_pass(self, query: str, expansion_features: Dict, document: str):
        """
        Safe forward pass with tensor shape validation and debugging.
        """
        try:
            # FIX 1: Get model prediction with shape validation
            predicted_score = self.model.forward(
                query=query,
                expansion_features=expansion_features,
                document=document
            )

            # FIX 2: Ensure predicted_score is a scalar tensor
            if predicted_score.dim() > 0:
                if predicted_score.numel() == 1:
                    predicted_score = predicted_score.squeeze()
                else:
                    logger.error(f"Predicted score has unexpected shape: {predicted_score.shape}")
                    # Take the first element if multiple values
                    predicted_score = predicted_score.flatten()[0]

            # Validate that it's now a scalar
            if predicted_score.dim() != 0:
                logger.error(f"After processing, predicted_score still not scalar: {predicted_score.shape}")
                predicted_score = predicted_score.mean()  # Force to scalar

            return predicted_score

        except Exception as e:
            logger.error(f"Error in safe_forward_pass: {e}")
            logger.error(f"Query: {query[:50]}...")
            logger.error(f"Document: {document[:50]}...")
            logger.error(f"Expansion features keys: {list(expansion_features.keys())[:3]}")
            raise

    def safe_target_creation(self, relevance_value, loss_type: str):
        """
        Create target tensor with proper shape for loss computation.
        """
        try:
            # FIX 3: Create target as scalar tensor on correct device
            if loss_type == "mse":
                # For MSE, use the actual relevance value
                target_score = torch.tensor(
                    float(relevance_value),
                    device=self.model.device,
                    dtype=torch.float32
                )
            elif loss_type == "bce":
                # For BCE, use binary classification
                target_score = torch.tensor(
                    1.0 if float(relevance_value) > 0 else 0.0,
                    device=self.model.device,
                    dtype=torch.float32
                )
            else:
                target_score = torch.tensor(
                    float(relevance_value),
                    device=self.model.device,
                    dtype=torch.float32
                )

            # Ensure target is scalar
            if target_score.dim() != 0:
                target_score = target_score.squeeze()
                if target_score.dim() != 0:
                    target_score = target_score.mean()

            return target_score

        except Exception as e:
            logger.error(f"Error creating target: {e}")
            logger.error(f"Relevance value: {relevance_value}, type: {type(relevance_value)}")
            raise

    def safe_loss_computation(self, predicted_score, target_score):
        """
        Compute loss with explicit shape checking and broadcasting fixes.
        """
        try:
            # FIX 4: Validate tensor shapes before loss computation
            logger.debug(f"Predicted score shape: {predicted_score.shape}, device: {predicted_score.device}")
            logger.debug(f"Target score shape: {target_score.shape}, device: {target_score.device}")

            # Ensure both tensors are scalars
            if predicted_score.dim() != 0:
                logger.warning(f"Predicted score not scalar: {predicted_score.shape}")
                predicted_score = predicted_score.squeeze()
                if predicted_score.dim() != 0:
                    predicted_score = predicted_score.mean()

            if target_score.dim() != 0:
                logger.warning(f"Target score not scalar: {target_score.shape}")
                target_score = target_score.squeeze()
                if target_score.dim() != 0:
                    target_score = target_score.mean()

            # Final validation
            assert predicted_score.dim() == 0, f"Predicted score still not scalar: {predicted_score.shape}"
            assert target_score.dim() == 0, f"Target score still not scalar: {target_score.shape}"
            assert predicted_score.device == target_score.device, f"Device mismatch: {predicted_score.device} vs {target_score.device}"

            # Compute loss
            loss = self.criterion(predicted_score, target_score)

            # Validate loss shape
            if loss.dim() != 0:
                logger.warning(f"Loss not scalar: {loss.shape}")
                loss = loss.mean()

            return loss

        except Exception as e:
            logger.error(f"Error in loss computation: {e}")
            logger.error(f"Predicted: {predicted_score}, shape: {predicted_score.shape}")
            logger.error(f"Target: {target_score}, shape: {target_score.shape}")
            logger.error(f"Loss type: {self.loss_type}")
            raise

    def train_epoch_pointwise(self, dataset, epoch: int) -> float:
        """Train epoch with broadcasting fixes."""
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        total_loss = 0.0
        num_batches = 0

        # Track weights
        if hasattr(self.model, 'alpha'):
            epoch_start_alpha = self.model.alpha.item()
            epoch_start_beta = self.model.beta.item()
            logger.info(f"Epoch {epoch} starting weights: α={epoch_start_alpha:.6f}, β={epoch_start_beta:.6f}")

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}
            global_step = epoch * len(dataloader) + step

            try:
                # Zero gradients
                self.optimizer.zero_grad()

                # FIX 5: Safe forward pass with shape validation
                predicted_score = self.safe_forward_pass(
                    query=example['query_text'],
                    expansion_features=example['expansion_features'],
                    document=example['doc_text']
                )

                # FIX 6: Safe target creation
                target_score = self.safe_target_creation(
                    example['relevance'],
                    self.loss_type
                )

                # FIX 7: Safe loss computation
                loss = self.safe_loss_computation(predicted_score, target_score)

                # Backward pass
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Log debug info
                if global_step % self.debug_frequency == 0:
                    if hasattr(self.model, 'alpha'):
                        alpha = self.model.alpha.item()
                        beta = self.model.beta.item()
                        alpha_grad = self.model.alpha.grad.item() if self.model.alpha.grad is not None else 0
                        beta_grad = self.model.beta.grad.item() if self.model.beta.grad is not None else 0

                        # logger.info(f"Step {global_step:4d} | α={alpha:.6f} β={beta:.6f} | Loss={loss.item():.4f}")
                        # logger.info(f"         | ∇α={alpha_grad:+.8f} ∇β={beta_grad:+.8f}")

                # Optimization step
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error in training step {global_step}: {e}")
                logger.warning(f"Skipping this training example")
                continue

        # Epoch summary
        if hasattr(self.model, 'alpha'):
            epoch_end_alpha = self.model.alpha.item()
            epoch_end_beta = self.model.beta.item()
            alpha_change = epoch_end_alpha - epoch_start_alpha
            beta_change = epoch_end_beta - epoch_start_beta

            logger.info(f"Epoch {epoch} weight changes: Δα={alpha_change:+.6f}, Δβ={beta_change:+.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset = None,
              num_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Train with broadcasting fixes.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'alpha_values': [],
            'beta_values': []
        }

        # Initial weights
        if hasattr(self.model, 'get_learned_weights'):
            initial_alpha, initial_beta = self.model.get_learned_weights()
            logger.info(f"Initial weights: α={initial_alpha:.6f}, β={initial_beta:.6f}")

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")

            # Training
            train_loss = self.train_epoch_pointwise(train_dataset, epoch + 1)
            history['train_loss'].append(train_loss)

            # Track weights
            if hasattr(self.model, 'get_learned_weights'):
                alpha, beta = self.model.get_learned_weights()
                history['alpha_values'].append(alpha)
                history['beta_values'].append(beta)
                logger.info(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, α={alpha:.6f}, β={beta:.6f}")

            # Validation
            if val_dataset:
                val_loss = self.evaluate(val_dataset)
                history['val_loss'].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")

        # Final summary
        if hasattr(self.model, 'get_learned_weights'):
            final_alpha, final_beta = self.model.get_learned_weights()
            logger.info(f"\nFinal weights: α={final_alpha:.6f}, β={final_beta:.6f}")

        return history

    def evaluate(self, dataset: Dataset) -> float:
        """Evaluate with broadcasting fixes."""
        self.model.eval()
        total_loss = 0.0
        num_examples = 0

        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            for batch in dataloader:
                example = {k: v[0] if isinstance(v, list) else v for k, v in batch.items()}

                try:
                    # Safe forward and loss computation
                    predicted_score = self.safe_forward_pass(
                        query=example['query_text'],
                        expansion_features=example['expansion_features'],
                        document=example['doc_text']
                    )

                    target_score = self.safe_target_creation(
                        example['relevance'],
                        self.loss_type
                    )

                    loss = self.safe_loss_computation(predicted_score, target_score)

                    total_loss += loss.item()
                    num_examples += 1

                except Exception as e:
                    logger.warning(f"Error in evaluation: {e}")
                    continue

        return total_loss / num_examples if num_examples > 0 else 0.0


# Debugging utilities
def debug_tensor_shapes(tensor, name="tensor"):
    """Debug utility to check tensor shapes."""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dim: {tensor.dim()}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Value: {tensor.item() if tensor.numel() == 1 else 'multiple values'}")


def test_loss_computation():
    """Test function to debug loss computation issues."""
    print("Testing loss computation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different tensor shapes
    test_cases = [
        (torch.tensor(0.5, device=device), torch.tensor(1.0, device=device)),  # Both scalars
        (torch.tensor([0.5], device=device), torch.tensor(1.0, device=device)),  # [1] vs scalar
        (torch.tensor(0.5, device=device), torch.tensor([1.0], device=device)),  # scalar vs [1]
        (torch.tensor([0.5], device=device), torch.tensor([1.0], device=device)),  # Both [1]
    ]

    criterion = nn.MSELoss()

    for i, (pred, target) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        debug_tensor_shapes(pred, "predicted")
        debug_tensor_shapes(target, "target")

        try:
            loss = criterion(pred, target)
            debug_tensor_shapes(loss, "loss")
            print(f"SUCCESS: Loss = {loss.item()}")
        except Exception as e:
            print(f"ERROR: {e}")


# Alias for compatibility
Trainer = BroadcastFixedTrainer

if __name__ == "__main__":
    test_loss_computation()