#!/usr/bin/env python3
"""
NPRF Main Entry Point

Unified interface for NPRF training and inference with configuration management.
"""

import argparse
import logging
import sys
from pathlib import Path

# Import modular components
from nprf_config import get_config, list_configs, create_config_from_args
from nprf_trainer_modular import NPRFTrainer
from nprf_reranker_modular import NPRFReranker
from nprf_core import SimilarityComputer, NPRFFeatureExtractor, ModelFactory
from nprf_data import load_and_preprocess_data, NPRFDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_nprf(args):
    """Train NPRF model."""
    logger.info("Starting NPRF training...")
    
    # Get configuration
    if args.config:
        config = get_config(args.config)
        logger.info(f"Using predefined config: {args.config}")
    else:
        config = create_config_from_args(args)
        logger.info("Using configuration from command line arguments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    config.save_json(output_dir / "nprf_config.json")
    
    # Load and preprocess data
    datasets = load_and_preprocess_data(
        args.train_file,
        args.val_file,
        preprocess_config=config.data.__dict__
    )
    
    train_data = datasets['train']
    val_data = datasets.get('val')
    
    logger.info(f"Training data: {len(train_data)} queries")
    if val_data:
        logger.info(f"Validation data: {len(val_data)} queries")
    
    # Initialize components
    device = args.device if hasattr(args, 'device') else None
    similarity_computer = SimilarityComputer(device=device)
    
    feature_extractor = NPRFFeatureExtractor(
        model_type=config.model.model_type,
        similarity_computer=similarity_computer,
        nb_supervised_doc=config.model.nb_supervised_doc,
        doc_topk_term=config.model.doc_topk_term,
        hist_size=config.model.hist_size,
        kernel_size=config.model.kernel_size
    )
    
    model = ModelFactory.create_model(
        config.model.model_type,
        **config.model.__dict__
    )
    
    # Create data loaders
    train_loader = NPRFDataLoader.create_train_loader(
        train_data, feature_extractor,
        batch_size=config.training.batch_size,
        sample_size=config.training.sample_size,
        num_workers=config.training.num_workers
    )
    
    val_loader = None
    if val_data:
        val_loader = NPRFDataLoader.create_train_loader(
            val_data, feature_extractor,
            batch_size=config.training.batch_size,
            sample_size=config.training.sample_size,
            num_workers=config.training.num_workers,
            shuffle=False
        )
    
    # Train model
    trainer = NPRFTrainer(model, feature_extractor, device=device)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate
    )
    
    # Save model
    model_path = output_dir / f"nprf_{config.model.model_type}_model.pt"
    trainer.save_model(model_path, args, feature_extractor)
    
    logger.info(f"Training completed! Model saved to: {model_path}")


def infer_nprf(args):
    """Run NPRF inference."""
    logger.info("Starting NPRF inference...")
    
    # Get configuration
    if args.config:
        config = get_config(args.config)
        logger.info(f"Using predefined config: {args.config}")
    else:
        config = create_config_from_args(args)
        logger.info("Using configuration from command line arguments")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load test data
    datasets = load_and_preprocess_data(
        train_file=None,
        test_file=args.test_file,
        preprocess_config={
            'min_candidates': config.data.min_candidates,
            'require_positive': False,
            'require_negative': False,
            'balance_dataset': False
        }
    )
    
    test_data = datasets['test']
    logger.info(f"Loaded {len(test_data)} test queries")
    
    # Initialize reranker
    device = getattr(args, 'device', None)
    reranker = NPRFReranker(
        model_path=getattr(args, 'model_path', None),
        model_type=config.model.model_type,
        device=device,
        **config.model.__dict__
    )
    
    # Process queries
    from nprf_reranker_modular import process_single_query
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    all_results = []
    successful_queries = 0
    failed_queries = 0
    
    if config.inference.max_workers == 1:
        # Single-threaded
        for query_data in test_data:
            result = process_single_query(query_data, reranker)
            if result['success']:
                all_results.extend(result['results'])
                successful_queries += 1
            else:
                failed_queries += 1
    else:
        # Multi-threaded
        with ThreadPoolExecutor(max_workers=config.inference.max_workers) as executor:
            future_to_query = {
                executor.submit(process_single_query, query_data, reranker): query_data['query_id']
                for query_data in test_data
            }
            
            for future in as_completed(future_to_query):
                result = future.result()
                if result['success']:
                    all_results.extend(result['results'])
                    successful_queries += 1
                else:
                    failed_queries += 1
    
    # Write results
    if all_results:
        from nprf_core import write_trec_run
        
        all_results.sort(key=lambda x: (x['query_id'], x['rank']))
        trec_file = output_dir / f"{config.inference.run_name}.trec"
        write_trec_run(all_results, trec_file, config.inference.run_name)
        
        logger.info(f"Inference completed!")
        logger.info(f"  Successful queries: {successful_queries}")
        logger.info(f"  Failed queries: {failed_queries}")
        logger.info(f"  TREC run file: {trec_file}")
    else:
        logger.error("No results generated!")


def main():
    parser = argparse.ArgumentParser(
        description="NPRF: Neural Pseudo Relevance Feedback",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train NPRF model')
    train_parser.add_argument('--train-file', type=str, required=True,
                              help='Path to training data JSONL file')
    train_parser.add_argument('--val-file', type=str,
                              help='Path to validation data JSONL file')
    train_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for trained models')
    train_parser.add_argument('--config', type=str,
                              choices=list_configs(),
                              help='Use predefined configuration')
    
    # Add training-specific arguments (used when --config is not specified)
    train_parser.add_argument('--model-type', type=str, default='drmm',
                              choices=['drmm', 'knrm'])
    train_parser.add_argument('--num-epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=20)
    train_parser.add_argument('--learning-rate', type=float, default=0.001)
    train_parser.add_argument('--nb-supervised-doc', type=int, default=10)
    train_parser.add_argument('--doc-topk-term', type=int, default=20)
    train_parser.add_argument('--hist-size', type=int, default=30)
    train_parser.add_argument('--kernel-size', type=int, default=11)
    train_parser.add_argument('--hidden-size', type=int, default=5)
    train_parser.add_argument('--sample-size', type=int, default=10)
    train_parser.add_argument('--min-candidates', type=int, default=5)
    train_parser.add_argument('--balance-dataset', action='store_true')
    train_parser.add_argument('--max-queries-per-class', type=int)
    train_parser.add_argument('--num-workers', type=int, default=4)
    train_parser.add_argument('--seed', type=int, default=42)
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run NPRF inference')
    infer_parser.add_argument('--test-file', type=str, required=True,
                              help='Path to test data JSONL file')
    infer_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for results')
    infer_parser.add_argument('--model-path', type=str,
                              help='Path to trained model file')
    infer_parser.add_argument('--config', type=str,
                              choices=list_configs(),
                              help='Use predefined configuration')
    
    # Add inference-specific arguments
    infer_parser.add_argument('--model-type', type=str, default='drmm',
                              choices=['drmm', 'knrm'])
    infer_parser.add_argument('--nb-supervised-doc', type=int, default=10)
    infer_parser.add_argument('--doc-topk-term', type=int, default=20)
    infer_parser.add_argument('--hist-size', type=int, default=30)
    infer_parser.add_argument('--kernel-size', type=int, default=11)
    infer_parser.add_argument('--hidden-size', type=int, default=5)
    infer_parser.add_argument('--score-combination-weight', type=float, default=0.7)
    infer_parser.add_argument('--run-name', type=str, default='nprf')
    infer_parser.add_argument('--max-workers', type=int, default=2)
    infer_parser.add_argument('--min-candidates', type=int, default=5)
    
    # Config command
    config_parser = subparsers.add_parser('configs', help='List available configurations')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_nprf(args)
    elif args.command == 'infer':
        infer_nprf(args)
    elif args.command == 'configs':
        print("Available configurations:")
        for config_name in list_configs():
            print(f"  - {config_name}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()