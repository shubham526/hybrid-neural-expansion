#!/usr/bin/env python3
"""
Evaluate Neural Reranker

Standalone evaluation script that creates the model from scratch,
loads a checkpoint, and runs inference on test data.
No dependency on model_info.json files.
"""

import argparse
import logging
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from cross_encoder.src.models.reranker import create_neural_reranker

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data_from_jsonl(jsonl_file: Path) -> Dict[str, Any]:
    """Load test data from JSONL file with embedded document text."""
    logger.info(f"Loading test data from JSONL: {jsonl_file}")

    queries = {}
    features = {}
    candidates = {}
    documents = {}  # Store document texts from JSONL

    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)

                query_id = example['query_id']
                queries[query_id] = example['query_text']
                features[query_id] = {'term_features': example['expansion_features']}

                # Extract candidates and document texts
                candidate_list = []
                for candidate in example['candidates']:
                    doc_id = candidate['doc_id']
                    score = candidate['score']
                    candidate_list.append((doc_id, score))

                    # Store document text if available
                    if 'doc_text' in candidate:
                        documents[doc_id] = candidate['doc_text']

                candidates[query_id] = candidate_list

    logger.info(f"Loaded {len(queries)} queries from JSONL")
    logger.info(f"Extracted {len(documents)} document texts from JSONL")
    return {
        'queries': queries,
        'features': features,
        'candidates': candidates,
        'documents': documents
    }


def save_trec_run(results: Dict[str, List[Tuple[str, float]]], output_file: Path, run_tag: str = "neural"):
    """Save results in TREC run format."""
    with open(output_file, 'w') as f:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
    logger.info(f"Saved TREC run file: {output_file}")


def run_neural_reranker(reranker,
                        queries: Dict[str, str],
                        features: Dict[str, Any],
                        candidates: Dict[str, List[Tuple[str, float]]],
                        documents: Dict[str, str],
                        top_k: int = 100) -> Dict[str, List[Tuple[str, float]]]:
    """Run neural reranker on test data."""
    logger.info("Running neural reranker...")

    results = {}

    for query_id in tqdm(queries.keys(), desc="Reranking queries"):
        if query_id not in features or query_id not in candidates:
            logger.warning(f"Missing data for query {query_id}, skipping")
            continue

        query_text = queries[query_id]
        expansion_features = features[query_id]['term_features']
        query_candidates = candidates[query_id]

        try:
            # Create document texts dict for this query's candidates
            document_texts = {}
            missing_docs = 0
            for doc_id, _ in query_candidates:
                if doc_id in documents:
                    document_texts[doc_id] = documents[doc_id]
                else:
                    missing_docs += 1

            if missing_docs > 0:
                logger.warning(f"Query {query_id}: {missing_docs} candidates missing document text")

            # Rerank using neural model
            reranked = reranker.rerank_candidates(
                query=query_text,
                expansion_features=expansion_features,
                candidates=query_candidates,
                document_texts=document_texts,
                top_k=top_k
            )

            results[query_id] = reranked

        except Exception as e:
            logger.error(f"Error reranking query {query_id}: {e}")
            # Fallback to original candidates
            results[query_id] = query_candidates[:top_k]

    logger.info(f"Completed reranking for {len(results)} queries")
    return results


def synchronize_model_devices(model, target_device):
    """Ensure all model components are on the same device."""
    # Update model's device attribute to match actual parameter device
    model.device = target_device

    # Normalize device to consistent format
    if isinstance(target_device, torch.device):
        if target_device.type == 'cuda' and target_device.index is None:
            # Convert 'cuda' to 'cuda:0' for consistency
            target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            model.device = target_device

    # Move encoder if needed
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'device'):
        if str(model.encoder.device) != str(target_device):
            logger.info(f"Moving encoder from {model.encoder.device} to {target_device}")
            model.encoder.device = target_device

            # Move encoder's model components
            if hasattr(model.encoder, 'model'):
                if hasattr(model.encoder, 'model_type'):
                    if model.encoder.model_type == "sentence_transformer":
                        model.encoder.model = model.encoder.model.to(target_device)
                    else:  # huggingface
                        model.encoder.model = model.encoder.model.to(target_device)
                else:
                    # Fallback - try to move anyway
                    model.encoder.model = model.encoder.model.to(target_device)

    # Ensure all parameters are on the target device
    if hasattr(model, 'alpha'):
        if model.alpha.device != target_device:
            logger.info(f"Moving alpha from {model.alpha.device} to {target_device}")
            model.alpha = model.alpha.to(target_device)

    if hasattr(model, 'beta'):
        if model.beta.device != target_device:
            logger.info(f"Moving beta from {model.beta.device} to {target_device}")
            model.beta = model.beta.to(target_device)

    logger.info(f"All model components synchronized to device: {target_device}")


def ensure_model_device_consistency(reranker, device):
    """Final check and fix for device consistency before inference."""
    # Get the actual device from parameters
    if hasattr(reranker, 'alpha') and hasattr(reranker, 'beta'):
        param_device = reranker.alpha.device

        # Update model's device attribute to match parameters
        if str(reranker.device) != str(param_device):
            logger.info(f"Fixing device inconsistency: model.device={reranker.device}, param.device={param_device}")
            reranker.device = param_device

            # Re-synchronize encoder
            if hasattr(reranker, 'encoder'):
                reranker.encoder.device = param_device
                if hasattr(reranker.encoder, 'model'):
                    reranker.encoder.model = reranker.encoder.model.to(param_device)

            logger.info(f"Fixed model device to: {param_device}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate neural reranker on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test.jsonl file (with embedded doc_text)')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output TREC run file path')

    # Model architecture arguments (must match training)
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--max-expansion-terms', type=int, default=15,
                        help='Maximum expansion terms to use')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for neural layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--scoring-method', type=str, default='neural',
                        choices=['neural', 'bilinear', 'cosine'],
                        help='Scoring method')
    parser.add_argument('--ablation-mode', type=str, default='both',
                        choices=['both', 'rm3_only', 'cosine_only'],
                        help='Ablation mode')
    parser.add_argument('--force-hf', action='store_true',
                        help='Force using HuggingFace transformers')
    parser.add_argument('--pooling-strategy', choices=['cls', 'mean', 'max'], default='cls',
                        help='Pooling strategy for HuggingFace models')

    # Evaluation arguments
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of documents to return per query')
    parser.add_argument('--run-tag', type=str, default='neural',
                        help='Run tag for TREC file')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device to use for inference')

    args = parser.parse_args()

    try:
        # Check CUDA availability and set device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.device == 'cuda':
            if not torch.cuda.is_available():
                logger.error("CUDA requested but not available!")
                sys.exit(1)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Load test data
        logger.info("Loading test data...")
        test_data = load_test_data_from_jsonl(Path(args.test_file))
        queries = test_data['queries']
        features = test_data['features']
        candidates = test_data['candidates']
        documents = test_data['documents']

        # Create neural reranker with same architecture as training
        logger.info("Creating neural reranker...")
        logger.info(f"Model architecture:")
        logger.info(f"  model_name: {args.model_name}")
        logger.info(f"  max_expansion_terms: {args.max_expansion_terms}")
        logger.info(f"  hidden_dim: {args.hidden_dim}")
        logger.info(f"  scoring_method: {args.scoring_method}")
        logger.info(f"  ablation_mode: {args.ablation_mode}")
        logger.info(f"  force_hf: {args.force_hf}")
        logger.info(f"  pooling_strategy: {args.pooling_strategy}")

        reranker = create_neural_reranker(
            model_name=args.model_name,
            max_expansion_terms=args.max_expansion_terms,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            scoring_method=args.scoring_method,
            device=args.device,
            force_hf=args.force_hf,
            pooling_strategy=args.pooling_strategy,
            ablation_mode=args.ablation_mode
        )

        # Move model to device and synchronize all components
        reranker = reranker.to(device)

        # Ensure all model components are on the same device (fix from trainer)
        synchronize_model_devices(reranker, device)
        logger.info(f"Model moved to device: {device}")

        # Load trained weights
        checkpoint_path = Path(args.model_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Model checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        logger.info(f"Loading model checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        reranker.load_state_dict(checkpoint)
        logger.info("Model checkpoint loaded successfully")

        # Ensure device consistency after loading checkpoint
        ensure_model_device_consistency(reranker, device)

        # Log learned weights
        alpha, beta = reranker.get_learned_weights()
        logger.info(f"Loaded model weights: α={alpha:.4f}, β={beta:.4f}")

        # Set model to evaluation mode
        reranker.eval()

        # Final device consistency check before inference
        ensure_model_device_consistency(reranker, device)

        # Run reranker
        with torch.no_grad():
            results = run_neural_reranker(
                reranker=reranker,
                queries=queries,
                features=features,
                candidates=candidates,
                documents=documents,
                top_k=args.top_k
            )

        # Save results
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_trec_run(results, output_file, args.run_tag)

        logger.info(f"Successfully saved {len(results)} reranked queries to {output_file}")

        # Log summary statistics
        total_candidates = sum(len(candidates[qid]) for qid in results.keys())
        avg_candidates = total_candidates / len(results) if results else 0
        logger.info(f"Evaluation summary:")
        logger.info(f"  Queries processed: {len(results)}")
        logger.info(f"  Documents available: {len(documents)}")
        logger.info(f"  Total candidates: {total_candidates}")
        logger.info(f"  Average candidates per query: {avg_candidates:.1f}")
        logger.info(f"  Top-k returned: {args.top_k}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()