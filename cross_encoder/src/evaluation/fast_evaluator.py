"""
Fast Single-Metric Evaluator for Training

Optimized evaluator for computing single metrics quickly during training.
Avoids the overhead of creating temporary files for each evaluation.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
import pytrec_eval

logger = logging.getLogger(__name__)


class FastTrainingEvaluator:
    """
    Fast evaluator optimized for training-time evaluation.
    Computes single metrics without file I/O overhead.
    """

    def __init__(self, metric: str = "ndcg_cut_20"):
        """
        Initialize fast evaluator.

        Args:
            metric: Metric to compute (e.g., 'ndcg_cut_20', 'map', 'recip_rank')
        """
        self.metric = metric
        self.supported_metrics = {
            'ndcg_cut_20', 'ndcg_cut_10', 'ndcg_cut_5',
            'map', 'recip_rank', 'recall_10', 'recall_100',
            'P_10', 'P_20'
        }

        if metric not in self.supported_metrics:
            logger.warning(f"Metric {metric} may not be supported. Supported: {self.supported_metrics}")

        logger.info(f"FastTrainingEvaluator initialized with metric: {metric}")

    def evaluate_run_fast(self,
                          run_results: Dict[str, List[Tuple[str, float]]],
                          qrels: Dict[str, Dict[str, int]]) -> float:
        """
        Fast evaluation of a single run.

        Args:
            run_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            Single metric score (average across queries)
        """
        if not run_results or not qrels:
            logger.warning("Empty run_results or qrels provided")
            return 0.0

        # Convert to pytrec_eval format in memory
        run_dict = {}
        qrel_dict = {}

        # Only process queries that have both run results and qrels
        common_queries = set(run_results.keys()) & set(qrels.keys())

        if not common_queries:
            logger.warning("No common queries between run and qrels")
            return 0.0

        for query_id in common_queries:
            # Convert run format
            run_dict[query_id] = {}
            for doc_id, score in run_results[query_id]:
                run_dict[query_id][doc_id] = float(score)

            # Convert qrel format
            qrel_dict[query_id] = qrels[query_id]

        try:
            # Evaluate using pytrec_eval
            evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, {self.metric})
            results = evaluator.evaluate(run_dict)

            # Compute average score
            scores = [query_results[self.metric] for query_results in results.values()]
            avg_score = np.mean(scores) if scores else 0.0

            logger.debug(f"Evaluated {len(common_queries)} queries, {self.metric}={avg_score:.4f}")
            return float(avg_score)

        except Exception as e:
            logger.error(f"Error in fast evaluation: {e}")
            return 0.0

    def evaluate_with_details(self,
                              run_results: Dict[str, List[Tuple[str, float]]],
                              qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Evaluate with per-query details.

        Args:
            run_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            Dictionary with average score and per-query scores
        """
        if not run_results or not qrels:
            return {'average': 0.0, 'per_query': {}, 'num_queries': 0}

        # Convert formats
        run_dict = {}
        qrel_dict = {}

        common_queries = set(run_results.keys()) & set(qrels.keys())

        for query_id in common_queries:
            run_dict[query_id] = {doc_id: float(score)
                                  for doc_id, score in run_results[query_id]}
            qrel_dict[query_id] = qrels[query_id]

        try:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, {self.metric})
            results = evaluator.evaluate(run_dict)

            per_query_scores = {qid: res[self.metric] for qid, res in results.items()}
            average_score = np.mean(list(per_query_scores.values())) if per_query_scores else 0.0

            return {
                'average': float(average_score),
                'per_query': per_query_scores,
                'num_queries': len(per_query_scores)
            }

        except Exception as e:
            logger.error(f"Error in detailed evaluation: {e}")
            return {'average': 0.0, 'per_query': {}, 'num_queries': 0}


class MetricTracker:
    """
    Track metric values across training epochs.
    """

    def __init__(self, metric_name: str = "ndcg_cut_20"):
        """
        Initialize metric tracker.

        Args:
            metric_name: Name of metric being tracked
        """
        self.metric_name = metric_name
        self.epoch_scores = []
        self.best_score = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def update(self, epoch: int, score: float) -> bool:
        """
        Update tracker with new score.

        Args:
            epoch: Current epoch number
            score: Metric score for this epoch

        Returns:
            True if this is a new best score, False otherwise
        """
        self.epoch_scores.append(score)

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            logger.info(f"🏆 New best {self.metric_name}: {score:.4f} (epoch {epoch})")
            return True
        else:
            self.epochs_without_improvement += 1
            logger.info(f"No improvement. Best: {self.best_score:.4f} (epoch {self.best_epoch})")
            return False

    def should_early_stop(self, patience: int) -> bool:
        """Check if training should stop early."""
        return self.epochs_without_improvement >= patience

    def get_improvement_stats(self) -> Dict[str, float]:
        """Get improvement statistics."""
        if len(self.epoch_scores) < 2:
            return {}

        first_score = self.epoch_scores[0]
        last_score = self.epoch_scores[-1]

        return {
            'first_score': first_score,
            'last_score': last_score,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_improvement': last_score - first_score,
            'best_improvement': self.best_score - first_score,
            'num_epochs': len(self.epoch_scores)
        }


# Update the TRECEvaluator to include fast evaluation method
def add_fast_evaluation_to_trec_evaluator():
    """
    Monkey patch to add fast evaluation to existing TRECEvaluator.
    This maintains compatibility with existing code.
    """
    from cross_encoder.src.evaluation.evaluator import TRECEvaluator

    def evaluate_single_metric_fast(self, run_results: Dict[str, List[Tuple[str, float]]],
                                    qrels: Dict[str, Dict[str, int]],
                                    metric: str = "ndcg_cut_20") -> float:
        """
        Fast single metric evaluation for training.

        Args:
            run_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}
            metric: Metric to compute

        Returns:
            Single metric score
        """
        fast_eval = FastTrainingEvaluator(metric)
        return fast_eval.evaluate_run_fast(run_results, qrels)

    # Add method to TRECEvaluator
    TRECEvaluator.evaluate_single_metric_fast = evaluate_single_metric_fast


# Utility functions for integration with training
def create_training_evaluator(metric: str = "ndcg_cut_20") -> FastTrainingEvaluator:
    """
    Factory function for creating training evaluator.

    Args:
        metric: Metric to use for evaluation

    Returns:
        Configured FastTrainingEvaluator
    """
    return FastTrainingEvaluator(metric)


def evaluate_epoch_performance(run_results: Dict[str, List[Tuple[str, float]]],
                               qrels: Dict[str, Dict[str, int]],
                               metric: str = "ndcg_cut_20") -> float:
    """
    Convenience function for epoch evaluation.

    Args:
        run_results: Run results for the epoch
        qrels: Relevance judgments
        metric: Metric to compute

    Returns:
        Metric score
    """
    evaluator = FastTrainingEvaluator(metric)
    return evaluator.evaluate_run_fast(run_results, qrels)


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test data
    run_results = {
        'q1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
        'q2': [('doc4', 0.95), ('doc1', 0.85), ('doc5', 0.75)]
    }

    qrels = {
        'q1': {'doc1': 2, 'doc2': 1, 'doc3': 0},
        'q2': {'doc4': 2, 'doc1': 0, 'doc5': 1}
    }

    # Test fast evaluation
    evaluator = FastTrainingEvaluator("ndcg_cut_10")
    score = evaluator.evaluate_run_fast(run_results, qrels)
    print(f"NDCG@10: {score:.4f}")

    # Test detailed evaluation
    details = evaluator.evaluate_with_details(run_results, qrels)
    print(f"Detailed results: {details}")

    # Test metric tracker
    tracker = MetricTracker("ndcg_cut_10")

    # Simulate training epochs
    test_scores = [0.45, 0.47, 0.46, 0.49, 0.48, 0.50, 0.49]
    for epoch, score in enumerate(test_scores, 1):
        is_best = tracker.update(epoch, score)
        print(f"Epoch {epoch}: {score:.3f} {'(NEW BEST!)' if is_best else ''}")

    print(f"Final stats: {tracker.get_improvement_stats()}")