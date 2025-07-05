"""
Logging Utilities for Bi-Encoder Systems

This module provides specialized logging utilities for bi-encoder training,
evaluation, and deployment, including performance monitoring and experiment tracking.
"""

import logging
import sys
import os
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import functools
import torch
import psutil
import threading
from collections import defaultdict
import numpy as np

# Import from cross-encoder for compatibility
from cross_encoder.src.utils.logging_utils import (
    setup_logging,
    get_logger,
    TimedOperation,
    log_experiment_info
)


class BiEncoderLogger:
    """
    Specialized logger for bi-encoder systems with performance monitoring.
    """

    def __init__(self,
                 name: str,
                 log_level: str = "INFO",
                 log_file: Optional[Path] = None,
                 track_performance: bool = True):
        """
        Initialize bi-encoder logger.

        Args:
            name: Logger name
            log_level: Logging level
            log_file: Optional log file path
            track_performance: Whether to track performance metrics
        """
        self.logger = get_logger(name)
        self.track_performance = track_performance
        self.performance_data = []
        self.training_metrics = []
        self.eval_metrics = []

        # Setup logging if file provided
        if log_file:
            self._setup_file_logging(log_file, log_level)

        self.logger.info(f"BiEncoderLogger initialized: {name}")

    def _setup_file_logging(self, log_file: Path, log_level: str):
        """Setup file logging handler."""
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def log_model_info(self, model, model_name: str = "BiEncoder"):
        """Log detailed model information."""
        self.logger.info(f"=== {model_name} Model Information ===")

        # Basic model info
        if hasattr(model, 'model_name'):
            self.logger.info(f"  Base model: {model.model_name}")
        if hasattr(model, 'embedding_dim'):
            self.logger.info(f"  Embedding dimension: {model.embedding_dim}")
        if hasattr(model, 'max_expansion_terms'):
            self.logger.info(f"  Max expansion terms: {model.max_expansion_terms}")
        if hasattr(model, 'similarity_function_name'):
            self.logger.info(f"  Similarity function: {model.similarity_function_name}")

        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Learned weights
        if hasattr(model, 'get_learned_weights'):
            weights = model.get_learned_weights()
            if len(weights) >= 2:
                self.logger.info(f"  α (RM3 weight): {weights[0]:.4f}")
                self.logger.info(f"  β (Semantic weight): {weights[1]:.4f}")
            if len(weights) >= 3:
                self.logger.info(f"  Expansion weight: {weights[2]:.4f}")

        # Device info
        if hasattr(model, 'device'):
            self.logger.info(f"  Device: {model.device}")

        self.logger.info("=" * 40)

    def log_training_config(self,
                            trainer_config: Dict[str, Any],
                            dataset_info: Dict[str, Any] = None):
        """Log training configuration."""
        self.logger.info("=== Training Configuration ===")

        # Trainer config
        for key, value in trainer_config.items():
            self.logger.info(f"  {key}: {value}")

        # Dataset info
        if dataset_info:
            self.logger.info("--- Dataset Information ---")
            for key, value in dataset_info.items():
                if isinstance(value, (int, float, str, bool)):
                    self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 30)

    def log_epoch_metrics(self,
                          epoch: int,
                          train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float] = None,
                          performance_metrics: Dict[str, float] = None):
        """Log metrics for a training epoch."""
        self.logger.info(f"Epoch {epoch} Results:")

        # Training metrics
        self.logger.info("  Training:")
        for metric, value in train_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"    {metric}: {value:.6f}")
            else:
                self.logger.info(f"    {metric}: {value}")

        # Validation metrics
        if val_metrics:
            self.logger.info("  Validation:")
            for metric, value in val_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {metric}: {value:.6f}")
                else:
                    self.logger.info(f"    {metric}: {value}")

        # Performance metrics
        if performance_metrics:
            self.logger.info("  Performance:")
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {metric}: {value:.3f}")
                else:
                    self.logger.info(f"    {metric}: {value}")

        # Track performance data
        if self.track_performance:
            epoch_data = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics or {},
                'performance_metrics': performance_metrics or {}
            }
            self.performance_data.append(epoch_data)

    def log_retrieval_results(self,
                              results: Dict[str, Any],
                              comparison: Dict[str, Any] = None):
        """Log retrieval evaluation results."""
        self.logger.info("=== Retrieval Evaluation Results ===")

        # Main results
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']
            self.logger.info("Dense Retrieval Performance:")
            for metric, score in metrics.items():
                self.logger.info(f"  {metric.upper()}: {score:.4f}")

        # Comparison with baseline
        if comparison:
            self.logger.info("\nComparison with Baseline:")

            if 'improvements' in comparison:
                improvements = comparison['improvements']
                for metric, improvement in improvements.items():
                    if 'improvement_pct' in metric:
                        base_metric = metric.replace('_improvement_pct', '')
                        self.logger.info(f"  {base_metric.upper()}: {improvement:+.1f}%")

        # Query statistics
        if 'num_queries' in results:
            self.logger.info(f"\nQueries evaluated: {results['num_queries']}")
        if 'top_k' in results:
            self.logger.info(f"Top-k retrieved: {results['top_k']}")

        self.logger.info("=" * 35)

    def log_expansion_analysis(self,
                               expansion_results: Dict[str, Any]):
        """Log query expansion effectiveness analysis."""
        self.logger.info("=== Query Expansion Analysis ===")

        if 'without_expansion' in expansion_results:
            without = expansion_results['without_expansion']
            self.logger.info("Without Expansion:")
            for metric, score in without.items():
                self.logger.info(f"  {metric.upper()}: {score:.4f}")

        if 'with_expansion' in expansion_results:
            with_exp = expansion_results['with_expansion']
            self.logger.info("With Expansion:")
            for metric, score in with_exp.items():
                self.logger.info(f"  {metric.upper()}: {score:.4f}")

        if 'expansion_improvements' in expansion_results:
            improvements = expansion_results['expansion_improvements']
            self.logger.info("Expansion Improvements:")
            for metric, improvement in improvements.items():
                if 'improvement_pct' in metric:
                    base_metric = metric.replace('_improvement_pct', '')
                    self.logger.info(f"  {base_metric.upper()}: {improvement:+.1f}%")

        self.logger.info("=" * 30)

    def log_indexing_stats(self,
                           num_documents: int,
                           embedding_dim: int,
                           index_type: str,
                           build_time: float,
                           index_size_mb: float = None):
        """Log document indexing statistics."""
        self.logger.info("=== Document Indexing Statistics ===")
        self.logger.info(f"  Documents indexed: {num_documents:,}")
        self.logger.info(f"  Embedding dimension: {embedding_dim}")
        self.logger.info(f"  Index type: {index_type}")
        self.logger.info(f"  Build time: {build_time:.2f}s")

        if index_size_mb:
            self.logger.info(f"  Index size: {index_size_mb:.1f} MB")

        # Performance metrics
        docs_per_second = num_documents / build_time if build_time > 0 else 0
        self.logger.info(f"  Indexing rate: {docs_per_second:.0f} docs/sec")

        self.logger.info("=" * 35)

    def log_search_performance(self,
                               num_queries: int,
                               search_time: float,
                               top_k: int,
                               index_type: str = None):
        """Log search performance statistics."""
        self.logger.info("=== Search Performance ===")
        self.logger.info(f"  Queries processed: {num_queries:,}")
        self.logger.info(f"  Total search time: {search_time:.3f}s")
        self.logger.info(f"  Top-k retrieved: {top_k}")

        if index_type:
            self.logger.info(f"  Index type: {index_type}")

        # Performance metrics
        queries_per_second = num_queries / search_time if search_time > 0 else 0
        avg_latency_ms = (search_time / num_queries * 1000) if num_queries > 0 else 0

        self.logger.info(f"  Query rate: {queries_per_second:.1f} queries/sec")
        self.logger.info(f"  Avg latency: {avg_latency_ms:.1f} ms/query")

        self.logger.info("=" * 25)

    def log_weight_evolution(self,
                             initial_weights: tuple,
                             final_weights: tuple,
                             epoch_weights: List[tuple] = None):
        """Log the evolution of learned importance weights."""
        self.logger.info("=== Weight Evolution Analysis ===")

        # Initial vs final
        self.logger.info(f"Initial weights:")
        self.logger.info(f"  α (RM3): {initial_weights[0]:.6f}")
        self.logger.info(f"  β (Semantic): {initial_weights[1]:.6f}")
        if len(initial_weights) > 2:
            self.logger.info(f"  Expansion: {initial_weights[2]:.6f}")

        self.logger.info(f"Final weights:")
        self.logger.info(f"  α (RM3): {final_weights[0]:.6f}")
        self.logger.info(f"  β (Semantic): {final_weights[1]:.6f}")
        if len(final_weights) > 2:
            self.logger.info(f"  Expansion: {final_weights[2]:.6f}")

        # Changes
        alpha_change = final_weights[0] - initial_weights[0]
        beta_change = final_weights[1] - initial_weights[1]

        self.logger.info(f"Weight changes:")
        self.logger.info(f"  Δα: {alpha_change:+.6f}")
        self.logger.info(f"  Δβ: {beta_change:+.6f}")

        if len(final_weights) > 2:
            exp_change = final_weights[2] - initial_weights[2]
            self.logger.info(f"  ΔExpansion: {exp_change:+.6f}")

        # Weight ratio analysis
        if final_weights[0] + final_weights[1] > 0:
            alpha_ratio = final_weights[0] / (final_weights[0] + final_weights[1])
            self.logger.info(f"Final α/(α+β) ratio: {alpha_ratio:.3f}")

        self.logger.info("=" * 30)

    def log_contrastive_training(self,
                                 epoch: int,
                                 batch_idx: int,
                                 loss: float,
                                 in_batch_acc: float = None,
                                 hard_neg_ratio: float = None):
        """Log contrastive training statistics."""
        log_msg = f"Epoch {epoch}, Batch {batch_idx}: Loss={loss:.4f}"

        if in_batch_acc is not None:
            log_msg += f", In-batch Acc={in_batch_acc:.3f}"

        if hard_neg_ratio is not None:
            log_msg += f", Hard Neg Ratio={hard_neg_ratio:.3f}"

        self.logger.debug(log_msg)

    def log_embedding_quality(self,
                              query_embeddings: torch.Tensor,
                              doc_embeddings: torch.Tensor,
                              labels: torch.Tensor = None):
        """Log embedding quality metrics."""
        with torch.no_grad():
            # Embedding statistics
            query_norm = torch.norm(query_embeddings, dim=1).mean().item()
            doc_norm = torch.norm(doc_embeddings, dim=1).mean().item()

            # Similarity statistics
            similarities = torch.cosine_similarity(
                query_embeddings.unsqueeze(1),
                doc_embeddings.unsqueeze(0),
                dim=2
            )

            self.logger.debug(f"Embedding Quality:")
            self.logger.debug(f"  Query norm (avg): {query_norm:.4f}")
            self.logger.debug(f"  Doc norm (avg): {doc_norm:.4f}")
            self.logger.debug(f"  Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

            if labels is not None:
                # Compute accuracy for positive pairs
                pos_similarities = similarities[labels == 1]
                neg_similarities = similarities[labels == 0]

                if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                    pos_mean = pos_similarities.mean().item()
                    neg_mean = neg_similarities.mean().item()
                    separation = pos_mean - neg_mean

                    self.logger.debug(f"  Positive sim (avg): {pos_mean:.4f}")
                    self.logger.debug(f"  Negative sim (avg): {neg_mean:.4f}")
                    self.logger.debug(f"  Separation: {separation:.4f}")

    def log_batch_processing(self,
                             epoch: int,
                             batch_idx: int,
                             batch_size: int,
                             processing_time: float,
                             memory_usage: float = None):
        """Log batch processing information."""
        throughput = batch_size / processing_time if processing_time > 0 else 0

        log_msg = f"Epoch {epoch}, Batch {batch_idx}: {batch_size} samples, {processing_time:.3f}s, {throughput:.1f} samples/s"

        if memory_usage:
            log_msg += f", Memory: {memory_usage:.1f}MB"

        self.logger.debug(log_msg)

    def save_performance_log(self, filepath: Path):
        """Save performance data to JSON file."""
        if not self.performance_data:
            self.logger.warning("No performance data to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.performance_data, f, indent=2)

        self.logger.info(f"Performance log saved to: {filepath}")


class PerformanceMonitor:
    """
    Monitor system performance during training and inference.
    """

    def __init__(self, monitor_gpu: bool = True):
        """
        Initialize performance monitor.

        Args:
            monitor_gpu: Whether to monitor GPU usage
        """
        self.monitor_gpu = monitor_gpu and torch.cuda.is_available()
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()

        self.logger = get_logger("performance_monitor")

        if self.monitor_gpu:
            self.logger.info("GPU monitoring enabled")
        else:
            self.logger.info("GPU monitoring disabled")

    def capture_metrics(self) -> Dict[str, float]:
        """Capture current system metrics."""
        metrics = {}

        # CPU and memory
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)

        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024 ** 3)
        metrics['memory_available_gb'] = memory.available / (1024 ** 3)

        # GPU metrics if available
        if self.monitor_gpu:
            try:
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)
                metrics['gpu_memory_percent'] = (
                                                            torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            except Exception as e:
                self.logger.warning(f"Failed to capture GPU metrics: {e}")

        # Runtime
        metrics['runtime_minutes'] = (time.time() - self.start_time) / 60

        return metrics

    def log_metrics(self, prefix: str = ""):
        """Log current metrics."""
        metrics = self.capture_metrics()

        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        if prefix:
            prefix = f"{prefix} - "

        self.logger.info(f"{prefix}CPU: {metrics['cpu_percent']:.1f}%, "
                         f"Memory: {metrics['memory_percent']:.1f}% ({metrics['memory_used_gb']:.1f}GB), "
                         f"Runtime: {metrics['runtime_minutes']:.1f}min")

        if self.monitor_gpu and 'gpu_memory_allocated' in metrics:
            self.logger.info(f"{prefix}GPU Memory: {metrics['gpu_memory_allocated']:.1f}GB allocated, "
                             f"{metrics['gpu_memory_reserved']:.1f}GB reserved")

    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage."""
        peak_metrics = {}

        for metric, values in self.metrics_history.items():
            if values:
                if 'percent' in metric or 'allocated' in metric or 'used' in metric:
                    peak_metrics[f"peak_{metric}"] = max(values)
                else:
                    peak_metrics[f"avg_{metric}"] = sum(values) / len(values)

        return peak_metrics

    def reset(self):
        """Reset monitoring data."""
        self.metrics_history.clear()
        self.start_time = time.time()


class TrainingProgressLogger:
    """
    Specialized logger for tracking training progress.
    """

    def __init__(self, total_epochs: int, log_frequency: int = 10):
        """
        Initialize training progress logger.

        Args:
            total_epochs: Total number of training epochs
            log_frequency: How often to log detailed progress
        """
        self.total_epochs = total_epochs
        self.log_frequency = log_frequency
        self.epoch_start_time = None
        self.training_start_time = time.time()

        self.logger = get_logger("training_progress")

        # Metrics tracking
        self.epoch_times = []
        self.loss_history = []
        self.metric_history = defaultdict(list)

        self.logger.info(f"Training progress logger initialized for {total_epochs} epochs")

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch

        if epoch % self.log_frequency == 0 or epoch == 1:
            remaining_epochs = self.total_epochs - epoch + 1
            if self.epoch_times:
                avg_epoch_time = np.mean(self.epoch_times)
                eta_seconds = avg_epoch_time * remaining_epochs
                eta_minutes = eta_seconds / 60
                self.logger.info(f"Starting epoch {epoch}/{self.total_epochs} (ETA: {eta_minutes:.1f} min)")
            else:
                self.logger.info(f"Starting epoch {epoch}/{self.total_epochs}")

    def end_epoch(self,
                  epoch: int,
                  train_loss: float,
                  val_loss: float = None,
                  metrics: Dict[str, float] = None):
        """Mark the end of an epoch and log progress."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
        else:
            epoch_time = 0

        # Track metrics
        self.loss_history.append(train_loss)
        if metrics:
            for key, value in metrics.items():
                self.metric_history[key].append(value)

        # Log epoch summary
        log_msg = f"Epoch {epoch} completed in {epoch_time:.1f}s - Train Loss: {train_loss:.6f}"

        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.6f}"

        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_msg += f", {metric_str}"

        self.logger.info(log_msg)

        # Progress statistics
        if epoch % self.log_frequency == 0:
            self._log_progress_stats(epoch)

    def _log_progress_stats(self, epoch: int):
        """Log detailed progress statistics."""
        if len(self.epoch_times) > 1:
            avg_epoch_time = np.mean(self.epoch_times)
            remaining_epochs = self.total_epochs - epoch
            eta_total_minutes = (avg_epoch_time * remaining_epochs) / 60

            self.logger.info(f"Progress: {epoch}/{self.total_epochs} ({100 * epoch / self.total_epochs:.1f}%)")
            self.logger.info(f"Average epoch time: {avg_epoch_time:.1f}s")
            self.logger.info(f"Estimated time remaining: {eta_total_minutes:.1f} minutes")

        # Loss trend
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            if recent_losses[0] > recent_losses[-1]:
                trend = "decreasing"
                change = recent_losses[0] - recent_losses[-1]
            else:
                trend = "increasing"
                change = recent_losses[-1] - recent_losses[0]

            self.logger.info(f"Recent loss trend: {trend} (Δ{change:.6f} over last 5 epochs)")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        total_time = time.time() - self.training_start_time

        summary = {
            'total_training_time': total_time,
            'total_epochs_completed': len(self.epoch_times),
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'final_train_loss': self.loss_history[-1] if self.loss_history else None,
            'best_train_loss': min(self.loss_history) if self.loss_history else None,
            'loss_improvement': (self.loss_history[0] - self.loss_history[-1]) if len(self.loss_history) > 1 else 0
        }

        # Add metric summaries
        for metric_name, values in self.metric_history.items():
            if values:
                summary[f'best_{metric_name}'] = max(values)
                summary[f'final_{metric_name}'] = values[-1]

        return summary


# Factory functions for easy setup
def create_bi_encoder_logger(name: str,
                             output_dir: Path,
                             log_level: str = "INFO") -> BiEncoderLogger:
    """Create a bi-encoder logger with file output."""
    log_file = output_dir / "training.log"
    return BiEncoderLogger(name, log_level, log_file)


def setup_bi_encoder_experiment_logging(experiment_name: str,
                                        output_dir: Path,
                                        log_level: str = "INFO") -> Tuple[BiEncoderLogger, PerformanceMonitor]:
    """Setup comprehensive logging for bi-encoder experiments."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    bi_logger = create_bi_encoder_logger(experiment_name, output_dir, log_level)

    # Setup performance monitor
    perf_monitor = PerformanceMonitor(monitor_gpu=torch.cuda.is_available())

    # Log experiment start
    bi_logger.logger.info("=" * 60)
    bi_logger.logger.info(f"BI-ENCODER EXPERIMENT: {experiment_name}")
    bi_logger.logger.info("=" * 60)
    bi_logger.logger.info(f"Output directory: {output_dir}")
    bi_logger.logger.info(f"Log level: {log_level}")
    bi_logger.logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        bi_logger.logger.info(f"GPU device: {torch.cuda.get_device_name()}")
        bi_logger.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    bi_logger.logger.info("=" * 60)

    return bi_logger, perf_monitor


# Context managers for specific logging scenarios
class ContrastiveTrainingLogger:
    """Context manager for contrastive training logging."""

    def __init__(self, bi_logger: BiEncoderLogger, epoch: int, total_batches: int):
        self.bi_logger = bi_logger
        self.epoch = epoch
        self.total_batches = total_batches
        self.batch_losses = []
        self.start_time = time.time()

    def __enter__(self):
        self.bi_logger.logger.info(f"Starting contrastive training epoch {self.epoch}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        epoch_time = time.time() - self.start_time
        avg_loss = np.mean(self.batch_losses) if self.batch_losses else 0

        self.bi_logger.logger.info(f"Contrastive training epoch {self.epoch} completed:")
        self.bi_logger.logger.info(f"  Time: {epoch_time:.2f}s")
        self.bi_logger.logger.info(f"  Avg loss: {avg_loss:.6f}")
        self.bi_logger.logger.info(f"  Batches processed: {len(self.batch_losses)}/{self.total_batches}")

    def log_batch(self, batch_idx: int, loss: float, **kwargs):
        """Log a training batch."""
        self.batch_losses.append(loss)
        self.bi_logger.log_contrastive_training(self.epoch, batch_idx, loss, **kwargs)


class RetrievalEvaluationLogger:
    """Context manager for retrieval evaluation logging."""

    def __init__(self, bi_logger: BiEncoderLogger, num_queries: int):
        self.bi_logger = bi_logger
        self.num_queries = num_queries
        self.start_time = time.time()

    def __enter__(self):
        self.bi_logger.logger.info(f"Starting retrieval evaluation on {self.num_queries} queries")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        eval_time = time.time() - self.start_time
        queries_per_second = self.num_queries / eval_time if eval_time > 0 else 0

        self.bi_logger.logger.info(f"Retrieval evaluation completed:")
        self.bi_logger.logger.info(f"  Time: {eval_time:.2f}s")
        self.bi_logger.logger.info(f"  Rate: {queries_per_second:.1f} queries/sec")


# Utility decorators for automatic logging
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with timing."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={len(kwargs)}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def log_gpu_memory(logger: logging.Logger):
    """Decorator to log GPU memory usage before and after function calls."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.debug(f"{func.__name__} - GPU memory before: {mem_before:.2f}GB")

            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_diff = mem_after - mem_before
                logger.debug(f"{func.__name__} - GPU memory after: {mem_after:.2f}GB (Δ{mem_diff:+.2f}GB)")

            return result

        return wrapper

    return decorator


# Advanced logging utilities for specific bi-encoder operations
class IndexingLogger:
    """Specialized logger for document indexing operations."""

    def __init__(self, bi_logger: BiEncoderLogger):
        self.bi_logger = bi_logger
        self.logger = bi_logger.logger

    def log_indexing_start(self, num_documents: int, batch_size: int, embedding_dim: int):
        """Log the start of indexing process."""
        self.logger.info("=" * 50)
        self.logger.info("DOCUMENT INDEXING STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Documents to index: {num_documents:,}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Embedding dimension: {embedding_dim}")

        estimated_batches = (num_documents + batch_size - 1) // batch_size
        self.logger.info(f"Estimated batches: {estimated_batches}")

    def log_batch_indexed(self, batch_idx: int, batch_size: int,
                          processing_time: float, total_batches: int):
        """Log progress of batch indexing."""
        docs_per_second = batch_size / processing_time if processing_time > 0 else 0
        progress = (batch_idx + 1) / total_batches * 100

        if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
            self.logger.info(f"Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - "
                             f"{docs_per_second:.0f} docs/sec")

    def log_indexing_complete(self, total_time: float, num_documents: int,
                              index_size_mb: float = None):
        """Log completion of indexing."""
        docs_per_second = num_documents / total_time if total_time > 0 else 0

        self.logger.info("=" * 50)
        self.logger.info("DOCUMENT INDEXING COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Documents indexed: {num_documents:,}")
        self.logger.info(f"Average rate: {docs_per_second:.0f} docs/sec")

        if index_size_mb:
            self.logger.info(f"Index size: {index_size_mb:.1f} MB")
            mb_per_doc = index_size_mb / num_documents
            self.logger.info(f"Size per document: {mb_per_doc:.4f} MB")

        self.logger.info("=" * 50)


class EvaluationLogger:
    """Specialized logger for retrieval evaluation."""

    def __init__(self, bi_logger: BiEncoderLogger):
        self.bi_logger = bi_logger
        self.logger = bi_logger.logger

    def log_evaluation_start(self, num_queries: int, top_k: int, metrics: List[str]):
        """Log the start of evaluation."""
        self.logger.info("=" * 50)
        self.logger.info("RETRIEVAL EVALUATION STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Queries to evaluate: {num_queries:,}")
        self.logger.info(f"Top-k documents: {top_k}")
        self.logger.info(f"Metrics: {', '.join(metrics)}")

    def log_query_batch_evaluated(self, batch_idx: int, batch_size: int,
                                  processing_time: float, total_batches: int):
        """Log progress of query evaluation."""
        queries_per_second = batch_size / processing_time if processing_time > 0 else 0
        progress = (batch_idx + 1) / total_batches * 100

        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
            self.logger.info(f"Query batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - "
                             f"{queries_per_second:.1f} queries/sec")

    def log_metric_results(self, metrics: Dict[str, float],
                           comparison_baseline: str = None,
                           improvements: Dict[str, float] = None):
        """Log evaluation metric results."""
        self.logger.info("--- Evaluation Results ---")

        for metric, score in metrics.items():
            self.logger.info(f"{metric.upper()}: {score:.4f}")

        if comparison_baseline and improvements:
            self.logger.info(f"--- Improvements over {comparison_baseline} ---")
            for metric, improvement in improvements.items():
                if 'improvement_pct' in metric:
                    base_metric = metric.replace('_improvement_pct', '')
                    self.logger.info(f"{base_metric.upper()}: {improvement:+.1f}%")

    def log_evaluation_complete(self, total_time: float, num_queries: int,
                                final_metrics: Dict[str, float]):
        """Log completion of evaluation."""
        queries_per_second = num_queries / total_time if total_time > 0 else 0

        self.logger.info("=" * 50)
        self.logger.info("RETRIEVAL EVALUATION COMPLETED")
        self.logger.info("=" * 50)
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Queries evaluated: {num_queries:,}")
        self.logger.info(f"Average rate: {queries_per_second:.1f} queries/sec")

        self.logger.info("Final Results:")
        for metric, score in final_metrics.items():
            self.logger.info(f"  {metric.upper()}: {score:.4f}")

        self.logger.info("=" * 50)


class WeightLearningLogger:
    """Specialized logger for importance weight learning."""

    def __init__(self, bi_logger: BiEncoderLogger):
        self.bi_logger = bi_logger
        self.logger = bi_logger.logger
        self.weight_history = []

    def log_weight_initialization(self, initial_weights: Dict[str, float]):
        """Log initial weight values."""
        self.logger.info("=== Weight Learning Initialization ===")
        for name, value in initial_weights.items():
            self.logger.info(f"Initial {name}: {value:.6f}")
        self.weight_history.append({
            'step': 0,
            'weights': initial_weights.copy(),
            'timestamp': datetime.now().isoformat()
        })

    def log_weight_update(self, step: int, weights: Dict[str, float],
                          gradients: Dict[str, float] = None,
                          learning_rates: Dict[str, float] = None):
        """Log weight updates during training."""
        self.logger.debug(f"Step {step} - Weight Update:")
        for name, value in weights.items():
            log_msg = f"  {name}: {value:.6f}"

            if gradients and name in gradients:
                log_msg += f" (grad: {gradients[name]:.6f})"

            if learning_rates and name in learning_rates:
                log_msg += f" (lr: {learning_rates[name]:.2e})"

            self.logger.debug(log_msg)

        # Store in history
        self.weight_history.append({
            'step': step,
            'weights': weights.copy(),
            'gradients': gradients.copy() if gradients else None,
            'learning_rates': learning_rates.copy() if learning_rates else None,
            'timestamp': datetime.now().isoformat()
        })

    def log_weight_convergence(self, final_weights: Dict[str, float],
                               initial_weights: Dict[str, float],
                               convergence_step: int):
        """Log weight convergence analysis."""
        self.logger.info("=== Weight Learning Convergence ===")

        for name in final_weights.keys():
            initial = initial_weights[name]
            final = final_weights[name]
            change = final - initial
            change_pct = (change / initial * 100) if initial != 0 else 0

            self.logger.info(f"{name}:")
            self.logger.info(f"  Initial: {initial:.6f}")
            self.logger.info(f"  Final: {final:.6f}")
            self.logger.info(f"  Change: {change:+.6f} ({change_pct:+.2f}%)")

        self.logger.info(f"Convergence step: {convergence_step}")

        # Compute weight balance
        if 'alpha' in final_weights and 'beta' in final_weights:
            alpha, beta = final_weights['alpha'], final_weights['beta']
            total = alpha + beta
            if total > 0:
                alpha_ratio = alpha / total
                beta_ratio = beta / total
                self.logger.info(f"Final weight balance:")
                self.logger.info(f"  α/(α+β): {alpha_ratio:.3f}")
                self.logger.info(f"  β/(α+β): {beta_ratio:.3f}")

    def get_weight_history(self) -> List[Dict]:
        """Get complete weight learning history."""
        return self.weight_history.copy()

    def save_weight_history(self, filepath: Path):
        """Save weight learning history to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.weight_history, f, indent=2)

        self.logger.info(f"Weight history saved to: {filepath}")


# Integration utilities for logging with existing components
def log_dataset_statistics(bi_logger: BiEncoderLogger,
                           dataset_name: str,
                           num_queries: int,
                           num_documents: int,
                           num_positive_pairs: int,
                           num_negative_pairs: int,
                           avg_query_length: float = None,
                           avg_doc_length: float = None):
    """Log comprehensive dataset statistics."""
    bi_logger.logger.info("=== Dataset Statistics ===")
    bi_logger.logger.info(f"Dataset: {dataset_name}")
    bi_logger.logger.info(f"Queries: {num_queries:,}")
    bi_logger.logger.info(f"Documents: {num_documents:,}")
    bi_logger.logger.info(f"Positive pairs: {num_positive_pairs:,}")
    bi_logger.logger.info(f"Negative pairs: {num_negative_pairs:,}")

    if num_positive_pairs + num_negative_pairs > 0:
        pos_ratio = num_positive_pairs / (num_positive_pairs + num_negative_pairs)
        bi_logger.logger.info(f"Positive ratio: {pos_ratio:.3f}")

    if avg_query_length:
        bi_logger.logger.info(f"Avg query length: {avg_query_length:.1f} tokens")

    if avg_doc_length:
        bi_logger.logger.info(f"Avg document length: {avg_doc_length:.1f} tokens")

    bi_logger.logger.info("=" * 25)


def log_training_summary(bi_logger: BiEncoderLogger,
                         training_summary: Dict[str, Any]):
    """Log comprehensive training summary."""
    bi_logger.logger.info("=" * 60)
    bi_logger.logger.info("TRAINING SUMMARY")
    bi_logger.logger.info("=" * 60)

    # Time and epochs
    if 'total_training_time' in training_summary:
        total_time = training_summary['total_training_time']
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        bi_logger.logger.info(f"Total training time: {hours:.0f}h {minutes:.0f}m ({total_time:.1f}s)")

    if 'total_epochs_completed' in training_summary:
        bi_logger.logger.info(f"Epochs completed: {training_summary['total_epochs_completed']}")

    if 'average_epoch_time' in training_summary:
        bi_logger.logger.info(f"Average epoch time: {training_summary['average_epoch_time']:.1f}s")

    # Loss metrics
    if 'final_train_loss' in training_summary:
        bi_logger.logger.info(f"Final training loss: {training_summary['final_train_loss']:.6f}")

    if 'best_train_loss' in training_summary:
        bi_logger.logger.info(f"Best training loss: {training_summary['best_train_loss']:.6f}")

    if 'loss_improvement' in training_summary:
        improvement = training_summary['loss_improvement']
        bi_logger.logger.info(f"Loss improvement: {improvement:.6f}")

    # Performance metrics
    performance_metrics = {k: v for k, v in training_summary.items()
                           if k.startswith('best_') and not k.startswith('best_train_loss')}

    if performance_metrics:
        bi_logger.logger.info("Best performance metrics:")
        for metric, value in performance_metrics.items():
            clean_metric = metric.replace('best_', '')
            bi_logger.logger.info(f"  {clean_metric}: {value:.4f}")

    bi_logger.logger.info("=" * 60)


def log_experiment_completion(bi_logger: BiEncoderLogger,
                              perf_monitor: PerformanceMonitor,
                              final_results: Dict[str, Any]):
    """Log experiment completion with full summary."""
    bi_logger.logger.info("=" * 80)
    bi_logger.logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    bi_logger.logger.info("=" * 80)

    # Performance summary
    peak_metrics = perf_monitor.get_peak_metrics()
    if peak_metrics:
        bi_logger.logger.info("Resource Usage Summary:")
        for metric, value in peak_metrics.items():
            if 'memory' in metric and 'gb' in metric:
                bi_logger.logger.info(f"  {metric}: {value:.2f} GB")
            elif 'percent' in metric:
                bi_logger.logger.info(f"  {metric}: {value:.1f}%")
            elif 'minutes' in metric:
                bi_logger.logger.info(f"  {metric}: {value:.1f} min")

    # Final results
    if 'evaluation_metrics' in final_results:
        bi_logger.logger.info("Final Evaluation Results:")
        for metric, score in final_results['evaluation_metrics'].items():
            bi_logger.logger.info(f"  {metric.upper()}: {score:.4f}")

    if 'learned_weights' in final_results:
        bi_logger.logger.info("Final Learned Weights:")
        for weight_name, value in final_results['learned_weights'].items():
            bi_logger.logger.info(f"  {weight_name}: {value:.6f}")

    bi_logger.logger.info("=" * 80)


# Example usage and testing functions
if __name__ == "__main__":
    import tempfile

    print("Testing Bi-Encoder Logging Utilities...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_experiment"

        # Setup experiment logging
        bi_logger, perf_monitor = setup_bi_encoder_experiment_logging(
            "test_bi_encoder_experiment",
            output_dir,
            "DEBUG"
        )


        # Test model logging
        class MockBiEncoder:
            def __init__(self):
                self.model_name = "test-model"
                self.embedding_dim = 384
                self.max_expansion_terms = 15
                self.device = torch.device("cpu")

            def parameters(self):
                return [torch.randn(100, 384), torch.randn(384)]

            def get_learned_weights(self):
                return (0.6, 0.4, 0.3)


        mock_model = MockBiEncoder()
        bi_logger.log_model_info(mock_model)

        # Test training progress
        progress_logger = TrainingProgressLogger(total_epochs=5, log_frequency=2)

        for epoch in range(1, 6):
            progress_logger.start_epoch(epoch)

            # Simulate training
            time.sleep(0.1)
            train_loss = 1.0 - epoch * 0.1
            val_loss = 0.9 - epoch * 0.08
            metrics = {'accuracy': 0.5 + epoch * 0.1}

            progress_logger.end_epoch(epoch, train_loss, val_loss, metrics)

            # Log epoch metrics
            bi_logger.log_epoch_metrics(
                epoch,
                {'loss': train_loss},
                {'loss': val_loss},
                metrics
            )

            # Monitor performance
            perf_monitor.log_metrics(f"Epoch {epoch}")

        # Test indexing logger
        indexing_logger = IndexingLogger(bi_logger)
        indexing_logger.log_indexing_start(1000, 32, 384)

        for batch_idx in range(0, 32):
            indexing_logger.log_batch_indexed(batch_idx, 32, 0.1, 32)

        indexing_logger.log_indexing_complete(3.2, 1000, 15.5)

        # Test evaluation logger
        eval_logger = EvaluationLogger(bi_logger)
        eval_logger.log_evaluation_start(100, 10, ['ndcg@10', 'map'])

        final_metrics = {'ndcg@10': 0.6, 'map': 0.45}
        eval_logger.log_evaluation_complete(5.0, 100, final_metrics)

        # Test weight learning logger
        weight_logger = WeightLearningLogger(bi_logger)
        weight_logger.log_weight_initialization({'alpha': 0.5, 'beta': 0.5})

        for step in range(1, 6):
            weights = {'alpha': 0.5 + step * 0.02, 'beta': 0.5 - step * 0.01}
            gradients = {'alpha': 0.01, 'beta': -0.005}
            weight_logger.log_weight_update(step, weights, gradients)

        weight_logger.log_weight_convergence(
            {'alpha': 0.6, 'beta': 0.45},
            {'alpha': 0.5, 'beta': 0.5},
            5
        )

        # Test summary logging
        training_summary = progress_logger.get_training_summary()
        log_training_summary(bi_logger, training_summary)

        # Test completion logging
        final_results = {
            'evaluation_metrics': final_metrics,
            'learned_weights': {'alpha': 0.6, 'beta': 0.4}
        }
        log_experiment_completion(bi_logger, perf_monitor, final_results)

        # Save performance log
        bi_logger.save_performance_log(output_dir / "performance.json")

        print("✓ All logging utilities tested successfully!")
        print(f"✓ Logs saved to: {output_dir}")

        # Print some sample output
        print("\nSample log output:")
        print("-" * 50)
        print("See the generated log files for complete output.")