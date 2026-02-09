"""
Enhanced Trainer with Dev Set Evaluation After Each Epoch

Adds capability to:
1. Evaluate on dev set after each epoch
2. Create TREC run files for each epoch
3. Save best model based on NDCG@20
4. Track evaluation metrics across epochs
5. Proper PyTorch DataLoader batching with custom collate function
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationAwareTrainer:
    """Enhanced trainer with built-in dev set evaluation and best model tracking."""

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 batch_size: int = 8,
                 loss_type: str = "mse",
                 importance_weight_lr: float = None,
                 debug_frequency: int = 10,
                 # New evaluation parameters
                 queries: Optional[Dict[str, str]] = None,
                 document_loader = None,
                 qrels: Optional[Dict[str, Dict[str, int]]] = None,
                 output_dir: Optional[Path] = None,
                 eval_metric: str = "ndcg_cut_20",
                 patience: int = 5,
                 rerank_top_k: int = 100):  # NEW parameter
        """
        Initialize trainer with evaluation capabilities.

        Args:
            model: Neural reranker model
            learning_rate: Base learning rate
            weight_decay: Weight decay
            batch_size: Training batch size
            loss_type: Loss function type
            importance_weight_lr: Learning rate for importance weights
            debug_frequency: How often to log debug info
            queries: Dev set queries {query_id: query_text}
            document_loader: Document loader for getting document text
            qrels: Relevance judgments {query_id: {doc_id: relevance}}
            output_dir: Directory to save run files and best model
            eval_metric: Metric to use for best model selection
            patience: Early stopping patience (epochs without improvement)
            rerank_top_k: Number of top documents to rerank during evaluation
        """
        # Original trainer setup
        self.model = model
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.debug_frequency = debug_frequency

        # Setup optimizers (same as before)
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

        # Loss function setup (same as before)
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "ranking":
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # NEW: Evaluation setup
        self.queries = queries or {}
        self.document_loader = document_loader
        self.qrels = qrels or {}
        self.output_dir = Path(output_dir) if output_dir else None
        self.eval_metric = eval_metric
        self.patience = patience
        self.rerank_top_k = rerank_top_k  # NEW

        # Best model tracking
        self.best_score = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Evaluation history
        self.eval_history = []

        # Setup evaluator if evaluation data provided
        self.evaluator = None
        if self.queries and self.qrels:
            from cross_encoder.src.evaluation.evaluator import TRECEvaluator
            self.evaluator = TRECEvaluator(metrics=[eval_metric])
            logger.info(f"Evaluation setup complete - using {eval_metric} for best model selection")
        else:
            logger.warning("No evaluation data provided - skipping dev set evaluation")

        logger.info(f"EvaluationAwareTrainer initialized:")
        logger.info(f"  Base learning rate: {learning_rate}")
        logger.info(f"  Importance weight LR: {importance_weight_lr}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Evaluation metric: {eval_metric}")
        logger.info(f"  Early stopping patience: {patience}")
        logger.info(f"  Rerank top-k: {rerank_top_k}")  # NEW

        # FIX: Ensure consistent device handling
        self.device = self.model.device

        # Normalize device to consistent format
        if isinstance(self.device, torch.device):
            if self.device.type == 'cuda' and self.device.index is None:
                # Convert 'cuda' to 'cuda:0' for consistency
                self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
                self.model = self.model.to(self.device)

        logger.info(f"Trainer using device: {self.device}")

    def evaluate_on_dev_set(self, val_dataset: Dataset, epoch: int) -> Tuple[float, Dict[str, List[Tuple[str, float]]]]:
        """
        Evaluate model on dev set and create run file.

        Args:
            val_dataset: Validation dataset
            epoch: Current epoch number

        Returns:
            Tuple of (metric_score, reranked_results)
        """
        if not self.evaluator:
            logger.warning("No evaluator available - skipping dev evaluation")
            return 0.0, {}

        logger.info(f"Evaluating on dev set (epoch {epoch})...")

        # Get validation queries and create first-stage runs
        val_queries = {}
        val_features = {}
        first_stage_runs = {}

        # Import collate function
        from cross_encoder.src.utils.data_utils import expansion_collate_fn

        # Extract data from validation dataset
        dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=expansion_collate_fn)

        for batch in dataloader:
            # Since batch_size=1, each batch has one example
            query_id = batch['query_id'][0]
            if query_id not in val_queries:
                val_queries[query_id] = batch['query_text'][0]
                val_features[query_id] = batch['expansion_features'][0]
                first_stage_runs[query_id] = []

            # Add this document as a candidate
            first_stage_runs[query_id].append((
                batch['doc_id'][0],
                batch['first_stage_score'][0]
            ))

        # Rerank using neural model
        reranked_results = self._rerank_with_model(
            val_queries, val_features, first_stage_runs
        )

        # Create run file
        if self.output_dir:
            run_file = self.output_dir / f"dev_run_epoch_{epoch}.txt"
            self._save_run_file(reranked_results, run_file, f"neural_epoch_{epoch}")
            logger.info(f"Saved dev run file: {run_file}")

        # Evaluate using qrels
        eval_qrels = {qid: self.qrels[qid] for qid in val_queries.keys() if qid in self.qrels}

        if eval_qrels:
            evaluation_results = self.evaluator.evaluate_run(reranked_results, eval_qrels)
            metric_score = evaluation_results.get(self.eval_metric, 0.0)

            logger.info(f"Epoch {epoch} dev evaluation:")
            logger.info(f"  {self.eval_metric}: {metric_score:.4f}")

            # Store detailed results
            self.eval_history.append({
                'epoch': epoch,
                'metric_score': metric_score,
                'all_metrics': evaluation_results
            })

            return metric_score, reranked_results
        else:
            logger.warning("No matching qrels found for evaluation")
            return 0.0, reranked_results

    def _rerank_with_model(self, queries: Dict[str, str],
                          features: Dict[str, Dict],
                          first_stage_runs: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
        """Rerank documents using the neural model."""
        # FIX: Synchronize devices before reranking
        self._synchronize_model_devices()
        self.model.eval()
        reranked_results = {}

        with torch.no_grad():
            for query_id, query_text in queries.items():
                if query_id not in features or query_id not in first_stage_runs:
                    continue

                expansion_features = features[query_id]
                candidates = first_stage_runs[query_id]

                # Get document texts
                document_texts = {}
                for doc_id, _ in candidates:
                    if self.document_loader:
                        doc_text = self.document_loader.get_document(doc_id)
                        if doc_text:
                            document_texts[doc_id] = doc_text

                # Rerank using neural model
                try:
                    reranked = self.model.rerank_candidates(
                        query=query_text,
                        expansion_features=expansion_features,
                        candidates=candidates,
                        document_texts=document_texts,
                        top_k=self.rerank_top_k  # Use configurable top-k
                    )
                    reranked_results[query_id] = reranked
                except Exception as e:
                    logger.warning(f"Error reranking query {query_id}: {e}")
                    reranked_results[query_id] = candidates[:self.rerank_top_k]  # Use configurable top-k

        return reranked_results

    def _save_run_file(self, results: Dict[str, List[Tuple[str, float]]],
                       filepath: Path, run_name: str):
        """Save results in TREC run format."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            for query_id, docs in results.items():
                for rank, (doc_id, score) in enumerate(docs, 1):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

    def save_best_model(self, epoch: int, metric_score: float):
        """Save model if it's the best so far."""
        if metric_score > self.best_score:
            self.best_score = metric_score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0

            if self.output_dir:
                # Save model state
                best_model_path = self.output_dir / 'best_model.pt'
                torch.save(self.model.state_dict(), best_model_path)

                # Save model info with current performance
                # if hasattr(self.model, 'get_learned_weights'):
                #     alpha, beta = self.model.get_learned_weights()
                # else:
                #     alpha, beta = 0.5, 0.5
                #
                # best_model_info = {
                #     'epoch': epoch,
                #     'best_score': metric_score,
                #     'metric': self.eval_metric,
                #     'learned_weights': {'alpha': alpha, 'beta': beta},
                #     'model_path': str(best_model_path)
                # }
                if hasattr(self.model, 'get_learned_weights'):
                    alpha, beta, lambda_val = self.model.get_learned_weights()
                else:
                    alpha, beta, lambda_val = 0.5, 0.5, 0.5

                best_model_info = {
                    'epoch': epoch,
                    'best_score': metric_score,
                    'metric': self.eval_metric,
                    'learned_weights': {'alpha': alpha, 'beta': beta, 'expansion_weight': lambda_val},
                    'model_path': str(best_model_path)
                }

                import json
                info_path = self.output_dir / 'best_model_info.json'
                with open(info_path, 'w') as f:
                    json.dump(best_model_info, f, indent=2)

                logger.info(f"🏆 New best model saved! Epoch {epoch}, {self.eval_metric}={metric_score:.4f}")
        else:
            self.epochs_without_improvement += 1
            logger.info(f"No improvement. Best: {self.best_score:.4f} (epoch {self.best_epoch})")

    def should_early_stop(self) -> bool:
        """Check if training should stop early."""
        return self.epochs_without_improvement >= self.patience

    def train_epoch_pointwise(self, dataset, epoch: int) -> float:
        """Train epoch with proper batching and collate function."""
        # FIX: Synchronize devices before training
        self._synchronize_model_devices()
        self.model.train()

        # Import collate function
        from cross_encoder.src.utils.data_utils import expansion_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=expansion_collate_fn
        )
        total_loss = 0.0
        num_batches = 0

        # Debug: Log actual batching info
        logger.info(f"Dataset size: {len(dataset)}, Batch size: {self.batch_size}, Expected batches: {len(dataloader)}")

        # Track weights
        if hasattr(self.model, 'alpha'):
            # epoch_start_alpha = self.model.alpha.item()
            # epoch_start_beta = self.model.beta.item()
            # logger.info(f"Epoch {epoch} starting weights: α={epoch_start_alpha:.6f}, β={epoch_start_beta:.6f}")
            epoch_start_alpha, epoch_start_beta, epoch_start_lambda = self.model.get_learned_weights()
            logger.info(
                f"Epoch {epoch} starting weights: α={epoch_start_alpha:.6f}, β={epoch_start_beta:.6f}, λ={epoch_start_lambda:.6f}")

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            global_step = epoch * len(dataloader) + step

            try:
                # Zero gradients
                self.optimizer.zero_grad()

                # Debug: Log actual batch size received
                actual_batch_size = len(batch['query_text'])
                if step == 0:  # Log only for first batch
                    logger.info(f"First batch actual size: {actual_batch_size}")

                # Collect all losses in the batch
                batch_losses = []

                # Process each example in the batch individually
                for i in range(actual_batch_size):
                    # Extract single example from batch
                    query_text = batch['query_text'][i]
                    expansion_features = batch['expansion_features'][i]
                    doc_text = batch['doc_text'][i]
                    relevance = batch['relevance'][i]

                    # Forward pass for single example
                    predicted_score = self.safe_forward_pass(
                        query=query_text,
                        expansion_features=expansion_features,
                        document=doc_text
                    )

                    # Create target
                    target_score = self.safe_target_creation(relevance, self.loss_type)

                    # Compute loss for this example
                    loss = self.safe_loss_computation(predicted_score, target_score)
                    batch_losses.append(loss)

                # Average loss across batch and backpropagate
                if batch_losses:
                    batch_loss = torch.stack(batch_losses).mean()
                    batch_loss.backward()

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimization step
                    self.optimizer.step()

                    total_loss += batch_loss.item()
                    num_batches += 1

            except Exception as e:
                logger.warning(f"Error in training batch {global_step}: {e}")
                continue

        # Epoch summary
        if hasattr(self.model, 'alpha'):
            # epoch_end_alpha = self.model.alpha.item()
            # epoch_end_beta = self.model.beta.item()
            # alpha_change = epoch_end_alpha - epoch_start_alpha
            # beta_change = epoch_end_beta - epoch_start_beta
            #
            # logger.info(f"Epoch {epoch} weight changes: Δα={alpha_change:+.6f}, Δβ={beta_change:+.6f}")

            epoch_end_alpha, epoch_end_beta, epoch_end_lambda = self.model.get_learned_weights()
            alpha_change = epoch_end_alpha - epoch_start_alpha
            beta_change = epoch_end_beta - epoch_start_beta
            lambda_change = epoch_end_lambda - epoch_start_lambda

            logger.info(
                f"Epoch {epoch} weight changes: Δα={alpha_change:+.6f}, Δβ={beta_change:+.6f}, Δλ={lambda_change:+.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0


    def train_epoch_pairwise(self, dataset: Dataset, epoch: int) -> float:
        """Train for one epoch using pairwise ranking loss."""
        self._synchronize_model_devices()
        self.model.train()

        from cross_encoder.src.utils.data_utils import pairwise_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pairwise_collate_fn
        )
        total_loss = 0.0
        num_batches = 0

        logger.info(f"Dataset size: {len(dataset)}, Batch size: {self.batch_size}, Expected batches: {len(dataloader)}")

        # Track weights if they exist
        if hasattr(self.model, 'get_learned_weights'):
            epoch_start_alpha, epoch_start_beta, epoch_start_lambda = self.model.get_learned_weights()
            logger.info(
                f"Epoch {epoch} starting weights: α={epoch_start_alpha:.6f}, β={epoch_start_beta:.6f}, λ={epoch_start_lambda:.6f}")

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} (Pairwise)")):
            global_step = epoch * len(dataloader) + step

            try:
                self.optimizer.zero_grad()

                # Get batch data
                query_texts = batch['query_text']
                expansion_features_list = batch['expansion_features']
                positive_docs = batch['positive_doc']
                negative_docs = batch['negative_doc']

                positive_scores = []
                negative_scores = []

                for i in range(len(query_texts)):
                    pos_score = self.model(
                        query=query_texts[i],
                        expansion_features=expansion_features_list[i],
                        document=positive_docs[i]['doc_text']
                    )
                    neg_score = self.model(
                        query=query_texts[i],
                        expansion_features=expansion_features_list[i],
                        document=negative_docs[i]['doc_text']
                    )
                    positive_scores.append(pos_score)
                    negative_scores.append(neg_score)

                positive_scores = torch.stack(positive_scores)
                negative_scores = torch.stack(negative_scores)

                target = torch.ones(positive_scores.size(0), device=self.model.device)

                loss = self.criterion(positive_scores, negative_scores, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.error(f"Error in training batch {global_step}: {e}")
                continue

        # Epoch summary
        if hasattr(self.model, 'get_learned_weights'):
            epoch_end_alpha, epoch_end_beta, epoch_end_lambda = self.model.get_learned_weights()
            alpha_change = epoch_end_alpha - epoch_start_alpha
            beta_change = epoch_end_beta - epoch_start_beta
            lambda_change = epoch_end_lambda - epoch_start_lambda

            logger.info(
                f"Epoch {epoch} weight changes: Δα={alpha_change:+.6f}, Δβ={beta_change:+.6f}, Δλ={lambda_change:+.6f}")

        return total_loss / num_batches if num_batches > 0 else 0.0
    def _synchronize_model_devices(self):
        """Ensure all model components are on the same device."""
        if not hasattr(self.model, 'alpha') or not hasattr(self.model, 'beta'):
            return

        # Get the device from model parameters (most reliable)
        param_device = self.model.alpha.device

        # Update model's device attribute
        self.model.device = param_device

        # Move encoder if needed
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'device'):
            if str(self.model.encoder.device) != str(param_device):
                logger.info(f"Moving encoder from {self.model.encoder.device} to {param_device}")
                self.model.encoder.device = param_device

                # Move encoder's model components
                if hasattr(self.model.encoder, 'model'):
                    if self.model.encoder.model_type == "sentence_transformer":
                        self.model.encoder.model = self.model.encoder.model.to(param_device)
                    else:  # huggingface
                        self.model.encoder.model = self.model.encoder.model.to(param_device)

        logger.info(f"All model components synchronized to device: {param_device}")

    def safe_forward_pass(self, query: str, expansion_features: Dict, document: str):
        """Safe forward pass with tensor shape validation."""
        try:

            # FIX: Temporarily move model to consistent device before forward pass
            original_device = self.model.device

            # Ensure model is on the same device format as its parameters
            if hasattr(self.model, 'alpha') and hasattr(self.model, 'beta'):
                param_device = self.model.alpha.device
                if str(original_device) != str(param_device):
                    logger.debug(f"Device mismatch detected: model={original_device}, params={param_device}")
                    # Use the parameter device as the canonical device
                    self.model.device = param_device

            predicted_score = self.model.forward(
                query=query,
                expansion_features=expansion_features,
                document=document
            )

            if predicted_score.dim() > 0:
                if predicted_score.numel() == 1:
                    predicted_score = predicted_score.squeeze()
                else:
                    logger.error(f"Predicted score has unexpected shape: {predicted_score.shape}")
                    predicted_score = predicted_score.flatten()[0]

            if predicted_score.dim() != 0:
                logger.error(f"After processing, predicted_score still not scalar: {predicted_score.shape}")
                predicted_score = predicted_score.mean()

            return predicted_score

        except Exception as e:
            logger.error(f"Error in safe_forward_pass: {e}")
            # Log detailed device information for debugging
            logger.error(f"Model device attribute: {getattr(self.model, 'device', 'None')}")
            if hasattr(self.model, 'alpha'):
                logger.error(f"Model alpha device: {self.model.alpha.device}")
            if hasattr(self.model, 'beta'):
                logger.error(f"Model beta device: {self.model.beta.device}")
            if hasattr(self.model, 'encoder'):
                logger.error(f"Encoder device: {getattr(self.model.encoder, 'device', 'None')}")
            raise

    def safe_target_creation(self, relevance_value, loss_type: str):
        """Create target tensor with proper shape for loss computation."""
        try:
            # FIX: Use the actual parameter device instead of model.device
            target_device = self.model.alpha.device if hasattr(self.model, 'alpha') else self.device

            if loss_type == "mse":
                target_score = torch.tensor(
                    float(relevance_value),
                    device=target_device,  # Use parameter device
                    dtype=torch.float32
                )
            elif loss_type == "bce":
                target_score = torch.tensor(
                    1.0 if float(relevance_value) > 0 else 0.0,
                    device=target_device,  # Use parameter device
                    dtype=torch.float32
                )
            else:
                target_score = torch.tensor(
                    float(relevance_value),
                    device=target_device,  # Use parameter device
                    dtype=torch.float32
                )

            if target_score.dim() != 0:
                target_score = target_score.squeeze()
                if target_score.dim() != 0:
                    target_score = target_score.mean()

            return target_score

        except Exception as e:
            logger.error(f"Error creating target: {e}")
            raise

    def safe_loss_computation(self, predicted_score, target_score):
        """Compute loss with explicit shape checking and broadcasting fixes."""
        try:
            logger.debug(f"Predicted score shape: {predicted_score.shape}, device: {predicted_score.device}")
            logger.debug(f"Target score shape: {target_score.shape}, device: {target_score.device}")

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

            assert predicted_score.dim() == 0, f"Predicted score still not scalar: {predicted_score.shape}"
            assert target_score.dim() == 0, f"Target score still not scalar: {target_score.shape}"
            assert predicted_score.device == target_score.device, f"Device mismatch: {predicted_score.device} vs {target_score.device}"

            loss = self.criterion(predicted_score, target_score)

            if loss.dim() != 0:
                logger.warning(f"Loss not scalar: {loss.shape}")
                loss = loss.mean()

            return loss

        except Exception as e:
            logger.error(f"Error in loss computation: {e}")
            raise

    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset = None,
              num_epochs: int = 10) -> Dict[str, List[float]]:
        """
        Enhanced training with dev set evaluation after each epoch.
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'alpha_values': [],
            'beta_values': [],
            'lambda_values': [],  # NEW
            'dev_scores': [],  # NEW: Track dev set performance
            'best_epochs': []   # NEW: Track when best model was saved
        }

        # Initial weights
        if hasattr(self.model, 'get_learned_weights'):
            initial_alpha, initial_beta, initial_lambda = self.model.get_learned_weights()  # Modified
            logger.info(
                f"Initial weights: α={initial_alpha:.6f}, β={initial_beta:.6f}, λ={initial_lambda:.6f}")  # Modified

            # initial_alpha, initial_beta = self.model.get_learned_weights()
            # logger.info(f"Initial weights: α={initial_alpha:.6f}, β={initial_beta:.6f}")

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")

            # Training
            if self.loss_type == 'ranking':  # Or check a training_mode attribute if you pass it
                train_loss = self.train_epoch_pairwise(train_dataset, epoch + 1)
            else:
                train_loss = self.train_epoch_pointwise(train_dataset, epoch + 1)
            history['train_loss'].append(train_loss)

            # Track weights
            if hasattr(self.model, 'get_learned_weights'):
                alpha, beta, lambda_val = self.model.get_learned_weights()  # Modified
                history['alpha_values'].append(alpha)
                history['beta_values'].append(beta)
                history['lambda_values'].append(lambda_val)  # NEW
                logger.info(
                    f"Epoch {epoch + 1}: Loss={train_loss:.4f}, α={alpha:.6f}, β={beta:.6f}, λ={lambda_val:.6f}")  # Modified

            # Validation loss (original functionality)
            if val_dataset:
                val_loss = self.evaluate_training_loss(val_dataset)
                history['val_loss'].append(val_loss)
                logger.info(f"Validation Loss: {val_loss:.4f}")

            # NEW: Dev set evaluation and best model saving
            if val_dataset and self.evaluator:
                dev_score, _ = self.evaluate_on_dev_set(val_dataset, epoch + 1)
                history['dev_scores'].append(dev_score)

                # Save best model
                self.save_best_model(epoch + 1, dev_score)
                history['best_epochs'].append(self.best_epoch)

                # Early stopping check
                if self.should_early_stop():
                    logger.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                    break

        # Final summary
        if hasattr(self.model, 'get_learned_weights'):
            final_alpha, final_beta, final_lambda = self.model.get_learned_weights()  # Modified
            logger.info(f"\nFinal weights: α={final_alpha:.6f}, β={final_beta:.6f}, λ={final_lambda:.6f}")  # Modified

        if self.evaluator:
            logger.info(f"Best model: Epoch {self.best_epoch}, {self.eval_metric}={self.best_score:.4f}")

        return history

    def evaluate_training_loss(self, dataset: Dataset) -> float:
        """Evaluate training loss on validation set with proper collate function."""
        # FIX: Synchronize devices before evaluation
        self._synchronize_model_devices()
        self.model.eval()
        total_loss = 0.0
        num_examples = 0

        # Import collate function
        from cross_encoder.src.utils.data_utils import expansion_collate_fn

        with torch.no_grad():
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=expansion_collate_fn
            )

            for batch in dataloader:
                batch_size = len(batch['query_text'])

                for i in range(batch_size):
                    try:
                        # Extract single example from batch
                        query_text = batch['query_text'][i]
                        expansion_features = batch['expansion_features'][i]
                        doc_text = batch['doc_text'][i]
                        relevance = batch['relevance'][i]

                        predicted_score = self.safe_forward_pass(
                            query=query_text,
                            expansion_features=expansion_features,
                            document=doc_text
                        )

                        target_score = self.safe_target_creation(relevance, self.loss_type)
                        loss = self.safe_loss_computation(predicted_score, target_score)

                        total_loss += loss.item()
                        num_examples += 1

                    except Exception as e:
                        logger.warning(f"Error in evaluation: {e}")
                        continue

        return total_loss / num_examples if num_examples > 0 else 0.0


# Alias for compatibility
Trainer = EvaluationAwareTrainer