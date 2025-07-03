#!/usr/bin/env python3
"""
Run Complete Neural Reranking Experiment

This script orchestrates the entire experimental pipeline:
1. Extract RM3 + semantic similarity features
2. Create train/test data from folds
3. Train neural rerankers for each fold
4. Evaluate trained models
5. Aggregate results across folds

Usage:
    python scripts/run_experiment.py --config configs/robust04_experiment.yaml

Or with command line arguments:
    python scripts/run_experiment.py \
        --dataset disks45/nocr/trec-robust-2004 \
        --folds-file ./folds.json \
        --index-path ./indexes/robust04 \
        --lucene-path ./lucene-jars \
        --output-dir ./experiments/robust04_neural
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.file_utils import load_json, save_json, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the complete neural reranking experiment."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner with configuration.

        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.output_dir = ensure_dir(config['output_dir'])
        self.scripts_dir = project_root / 'scripts'

        # Create experiment subdirectories
        self.features_dir = ensure_dir(self.output_dir / 'features')
        self.data_dir = ensure_dir(self.output_dir / 'data')
        self.models_dir = ensure_dir(self.output_dir / 'models')
        self.results_dir = ensure_dir(self.output_dir / 'results')

        logger.info(f"ExperimentRunner initialized:")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Dataset: {config['dataset']}")
        logger.info(f"  Folds file: {config['folds_file']}")

    def run_command(self, cmd: List[str], step_name: str,
                    capture_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run a command with proper logging and error handling.

        Args:
            cmd: Command to run as list of strings
            step_name: Name of the step (for logging)
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        logger.info(f"Running {step_name}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=True,
                cwd=project_root
            )

            duration = time.time() - start_time
            logger.info(f"✓ {step_name} completed in {duration:.2f}s")

            return result

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            logger.error(f"✗ {step_name} failed after {duration:.2f}s")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Return code: {e.returncode}")
            if e.stdout:
                logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr:
                logger.error(f"STDERR:\n{e.stderr}")
            raise

    def step1_extract_features(self) -> Path:
        """
        Step 1: Extract RM3 + semantic similarity features.

        Returns:
            Path to extracted features file
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: EXTRACTING FEATURES")
        logger.info("=" * 60)

        # Build command
        cmd = [
            'python', str(self.scripts_dir / '1_create_features.py'),
            '--dataset', self.config['dataset'],
            '--output-dir', str(self.features_dir),
            '--index-path', self.config['index_path'],
            '--lucene-path', self.config['lucene_path'],
            '--semantic-model', self.config.get('semantic_model', 'all-MiniLM-L6-v2'),
            '--max-expansion-terms', str(self.config.get('max_expansion_terms', 15)),
            '--top-k-pseudo-docs', str(self.config.get('top_k_pseudo_docs', 10)),
            '--log-level', self.config.get('log_level', 'INFO')
        ]

        # Add optional arguments
        if 'run_file_path' in self.config:
            cmd.extend(['--run-file-path', self.config['run_file_path']])

        if 'query_ids_file' in self.config:
            cmd.extend(['--query-ids-file', self.config['query_ids_file']])

        # Run feature extraction
        self.run_command(cmd, "Feature extraction")

        # Find the generated features file
        dataset_name = self.config['dataset'].replace('/', '_')
        subset_name = "subset" if 'query_ids_file' in self.config else "full"
        features_file = self.features_dir / f"{dataset_name}_{subset_name}_features.json.gz"

        if not features_file.exists():
            # Try without compression
            features_file = self.features_dir / f"{dataset_name}_{subset_name}_features.json"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        logger.info(f"Features extracted to: {features_file}")
        return features_file

    def step2_create_train_test_data(self, features_file: Path) -> Dict[str, Path]:
        """
        Step 2: Create train/test data for each fold.

        Args:
            features_file: Path to extracted features

        Returns:
            Dictionary mapping fold_id -> fold_directory
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: CREATING TRAIN/TEST DATA")
        logger.info("=" * 60)

        # Build command
        cmd = [
            'python', str(self.scripts_dir / '0_create_train_test_data.py'),
            '--dataset', self.config['dataset'],
            '--features-file', str(features_file),
            '--folds-file', self.config['folds_file'],
            '--output-dir', str(self.data_dir),
            '--max-candidates-per-query', str(self.config.get('max_candidates_per_query', 100)),
            '--save-statistics',
            '--log-level', self.config.get('log_level', 'INFO')
        ]

        # Add optional arguments
        if 'run_file_path' in self.config:
            cmd.extend(['--run-file-path', self.config['run_file_path']])

        if self.config.get('ensure_positive_training', False):
            cmd.append('--ensure-positive-training')

        # Run data creation
        self.run_command(cmd, "Train/test data creation")

        # Get fold directories
        folds = load_json(self.config['folds_file'])
        fold_dirs = {}
        for fold_id in folds.keys():
            fold_dir = self.data_dir / f"fold_{fold_id}"
            if fold_dir.exists():
                fold_dirs[fold_id] = fold_dir

        logger.info(f"Created data for {len(fold_dirs)} folds")
        return fold_dirs

    def step3_train_models(self, fold_dirs: Dict[str, Path]) -> Dict[str, Path]:
        """
        Step 3: Train neural rerankers for each fold.

        Args:
            fold_dirs: Dictionary mapping fold_id -> fold_directory

        Returns:
            Dictionary mapping fold_id -> model_directory
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: TRAINING NEURAL MODELS")
        logger.info("=" * 60)

        model_dirs = {}

        for fold_id, fold_dir in fold_dirs.items():
            logger.info(f"\n--- Training model for fold {fold_id} ---")

            # Create model directory for this fold
            model_dir = ensure_dir(self.models_dir / f"fold_{fold_id}")

            # Check if training files exist
            train_file = fold_dir / 'train.jsonl'
            if not train_file.exists():
                logger.warning(f"Training file not found for fold {fold_id}: {train_file}")
                continue

            # Build command
            cmd = [
                'python', str(self.scripts_dir / '2_train_neural_model.py'),
                '--train-file', str(train_file),
                '--output-dir', str(model_dir),
                '--model-name', self.config.get('model_name', 'all-MiniLM-L6-v2'),
                '--max-expansion-terms', str(self.config.get('max_expansion_terms', 15)),
                '--hidden-dim', str(self.config.get('hidden_dim', 128)),
                '--dropout', str(self.config.get('dropout', 0.1)),
                '--num-epochs', str(self.config.get('num_epochs', 20)),
                '--learning-rate', str(self.config.get('learning_rate', 1e-3)),
                '--weight-decay', str(self.config.get('weight_decay', 1e-4)),
                '--log-level', self.config.get('log_level', 'INFO')
            ]

            # Add validation file if it exists
            val_file = fold_dir / 'val.jsonl'
            if val_file.exists():
                cmd.extend(['--val-file', str(val_file)])

            # Add device if specified
            if 'device' in self.config:
                cmd.extend(['--device', self.config['device']])

            # Run training
            try:
                self.run_command(cmd, f"Training fold {fold_id}")
                model_dirs[fold_id] = model_dir

                # Log learned weights
                model_info_file = model_dir / 'model_info.json'
                if model_info_file.exists():
                    model_info = load_json(model_info_file)
                    weights = model_info['learned_weights']
                    logger.info(f"Fold {fold_id} learned weights: α={weights['alpha']:.4f}, β={weights['beta']:.4f}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed for fold {fold_id}")
                continue

        logger.info(f"Successfully trained models for {len(model_dirs)} folds")
        return model_dirs

    def step4_evaluate_models(self, fold_dirs: Dict[str, Path],
                              model_dirs: Dict[str, Path]) -> Dict[str, Path]:
        """
        Step 4: Evaluate trained models.

        Args:
            fold_dirs: Dictionary mapping fold_id -> fold_directory
            model_dirs: Dictionary mapping fold_id -> model_directory

        Returns:
            Dictionary mapping fold_id -> results_directory
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: EVALUATING MODELS")
        logger.info("=" * 60)

        results_dirs = {}

        for fold_id in fold_dirs.keys():
            if fold_id not in model_dirs:
                logger.warning(f"No trained model for fold {fold_id}, skipping evaluation")
                continue

            logger.info(f"\n--- Evaluating model for fold {fold_id} ---")

            fold_dir = fold_dirs[fold_id]
            model_dir = model_dirs[fold_id]
            results_dir = ensure_dir(self.results_dir / f"fold_{fold_id}")

            # Check required files
            test_file = fold_dir / 'test.jsonl'
            model_info_file = model_dir / 'model_info.json'

            if not test_file.exists():
                logger.warning(f"Test file not found for fold {fold_id}: {test_file}")
                continue

            if not model_info_file.exists():
                logger.warning(f"Model info file not found for fold {fold_id}: {model_info_file}")
                continue

            # Build command
            cmd = [
                'python', str(self.scripts_dir / '3_evaluate_model.py'),
                '--test-file', str(test_file),
                '--model-info-file', str(model_info_file),
                '--dataset', self.config['dataset'],
                '--output-dir', str(results_dir),
                '--save-runs',
                '--top-k', str(self.config.get('top_k', 100)),
                '--log-level', self.config.get('log_level', 'INFO')
            ]

            # Add baseline evaluation if requested
            if self.config.get('run_baselines', False):
                cmd.append('--run-baselines')

            # Run evaluation
            try:
                self.run_command(cmd, f"Evaluation fold {fold_id}")
                results_dirs[fold_id] = results_dir

            except subprocess.CalledProcessError as e:
                logger.error(f"Evaluation failed for fold {fold_id}")
                continue

        logger.info(f"Successfully evaluated {len(results_dirs)} folds")
        return results_dirs

    def step5_aggregate_results(self, results_dirs: Dict[str, Path]) -> Dict[str, Any]:
        """
        Step 5: Aggregate results across all folds.

        Args:
            results_dirs: Dictionary mapping fold_id -> results_directory

        Returns:
            Aggregated results dictionary
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: AGGREGATING RESULTS")
        logger.info("=" * 60)

        # Collect results from all folds
        fold_results = {}
        all_learned_weights = {}

        for fold_id, results_dir in results_dirs.items():
            # Load evaluation results
            eval_file = results_dir / 'evaluation_results.json'
            if eval_file.exists():
                fold_results[fold_id] = load_json(eval_file)

            # Load learned weights
            model_dir = self.models_dir / f"fold_{fold_id}"
            model_info_file = model_dir / 'model_info.json'
            if model_info_file.exists():
                model_info = load_json(model_info_file)
                all_learned_weights[fold_id] = model_info['learned_weights']

        if not fold_results:
            logger.warning("No fold results found for aggregation")
            return {}

        # Aggregate metrics across folds
        metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_20', 'recall_100', 'recip_rank']
        aggregated_results = {
            'num_folds': len(fold_results),
            'dataset': self.config['dataset'],
            'fold_results': fold_results,
            'learned_weights': all_learned_weights,
            'aggregated_metrics': {}
        }

        # Compute mean and std for each metric
        for metric in metrics:
            values = []
            for fold_id, results in fold_results.items():
                if 'our_method' in results and metric in results['our_method']:
                    values.append(results['our_method'][metric])

            if values:
                import numpy as np
                aggregated_results['aggregated_metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values,
                    'num_folds': len(values)
                }

        # Compute mean learned weights
        if all_learned_weights:
            alpha_values = [w['alpha'] for w in all_learned_weights.values()]
            beta_values = [w['beta'] for w in all_learned_weights.values()]

            import numpy as np
            aggregated_results['mean_learned_weights'] = {
                'alpha': {
                    'mean': float(np.mean(alpha_values)),
                    'std': float(np.std(alpha_values)),
                    'values': alpha_values
                },
                'beta': {
                    'mean': float(np.mean(beta_values)),
                    'std': float(np.std(beta_values)),
                    'values': beta_values
                }
            }

        # Save aggregated results
        aggregated_file = self.output_dir / 'aggregated_results.json'
        save_json(aggregated_results, aggregated_file)
        logger.info(f"Saved aggregated results to: {aggregated_file}")

        # Log key results
        logger.info("\n" + "=" * 50)
        logger.info("FINAL AGGREGATED RESULTS")
        logger.info("=" * 50)

        for metric, stats in aggregated_results['aggregated_metrics'].items():
            logger.info(f"{metric.upper()}:")
            logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(f"  Folds: {stats['num_folds']}")

        if 'mean_learned_weights' in aggregated_results:
            weights = aggregated_results['mean_learned_weights']
            logger.info(f"\nLEARNED WEIGHTS:")
            logger.info(f"  α (RM3): {weights['alpha']['mean']:.4f} ± {weights['alpha']['std']:.4f}")
            logger.info(f"  β (Semantic): {weights['beta']['mean']:.4f} ± {weights['beta']['std']:.4f}")

        logger.info("=" * 50)

        return aggregated_results

    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline.

        Returns:
            Final aggregated results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE NEURAL RERANKING EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.config['dataset']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)

        try:
            # Step 1: Extract features
            features_file = self.step1_extract_features()

            # Step 2: Create train/test data
            fold_dirs = self.step2_create_train_test_data(features_file)

            # Step 3: Train models
            model_dirs = self.step3_train_models(fold_dirs)

            # Step 4: Evaluate models
            results_dirs = self.step4_evaluate_models(fold_dirs, model_dirs)

            # Step 5: Aggregate results
            final_results = self.step5_aggregate_results(results_dirs)

            # Save experiment summary
            experiment_summary = {
                'config': self.config,
                'features_file': str(features_file),
                'num_folds_trained': len(model_dirs),
                'num_folds_evaluated': len(results_dirs),
                'final_results': final_results,
                'experiment_directory': str(self.output_dir)
            }

            summary_file = self.output_dir / 'experiment_summary.json'
            save_json(experiment_summary, summary_file)

            logger.info("\n" + "=" * 80)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Results saved in: {self.output_dir}")
            logger.info(f"Summary file: {summary_file}")
            logger.info("=" * 80)

            return final_results

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def load_config(config_file: Optional[Path], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load configuration from file and command line arguments.

    Args:
        config_file: Path to YAML config file (optional)
        args: Command line arguments

    Returns:
        Merged configuration dictionary
    """
    config = {}

    # Load from YAML file if provided
    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {config_file}")

    # Override with command line arguments
    cli_config = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_config)

    # Validate required fields
    required_fields = ['dataset', 'folds_file', 'index_path', 'lucene_path', 'output_dir']
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run complete neural reranking experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument('--config', type=str,
                        help='Path to YAML configuration file')

    # Required arguments
    parser.add_argument('--dataset', type=str,
                        help='IR dataset name')
    parser.add_argument('--folds-file', type=str,
                        help='Path to folds.json file')
    parser.add_argument('--index-path', type=str,
                        help='Path to Lucene index for RM3')
    parser.add_argument('--lucene-path', type=str,
                        help='Path to Lucene JAR files')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for experiment')

    # Optional data arguments
    parser.add_argument('--run-file-path', type=str,
                        help='Path to first-stage run file')
    parser.add_argument('--query-ids-file', type=str,
                        help='File with query IDs to process')

    # Feature extraction arguments
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms')
    parser.add_argument('--top-k-pseudo-docs', type=int, default=10,
                        help='Number of pseudo-relevant documents')

    # Training arguments
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Model name for neural reranker')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')

    # Data creation arguments
    parser.add_argument('--max-candidates-per-query', type=int, default=100,
                        help='Maximum candidates per query')
    parser.add_argument('--ensure-positive-training', action='store_true',
                        help='Ensure training queries have positive examples')

    # Evaluation arguments
    parser.add_argument('--run-baselines', action='store_true',
                        help='Run baseline comparisons')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of results to return')

    # Other arguments
    parser.add_argument('--device', type=str,
                        help='Device (cuda/cpu)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    try:
        # Load configuration
        config_file = Path(args.config) if args.config else None
        config = load_config(config_file, args)

        # Setup logging
        output_dir = ensure_dir(config['output_dir'])
        logger = setup_experiment_logging("run_experiment", config['log_level'],
                                          str(output_dir / 'experiment.log'))
        log_experiment_info(logger, **config)

        # Run experiment
        runner = ExperimentRunner(config)
        final_results = runner.run_complete_experiment()

        # Print final summary
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results directory: {output_dir}")

        if 'aggregated_metrics' in final_results:
            print("\nKey Results:")
            for metric, stats in final_results['aggregated_metrics'].items():
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        if 'mean_learned_weights' in final_results:
            weights = final_results['mean_learned_weights']
            print(f"\nLearned Weights:")
            print(f"  α (RM3): {weights['alpha']['mean']:.4f} ± {weights['alpha']['std']:.4f}")
            print(f"  β (Semantic): {weights['beta']['mean']:.4f} ± {weights['beta']['std']:.4f}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Experiment runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()