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
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

from cross_encoder.src.models.reranker import create_neural_reranker

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Simple document loader."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.documents = self._load_all_documents()
        logger.info(f"Loaded {len(self.documents)} documents into memory")

    def _load_all_documents(self) -> Dict[str, str]:
        """Load all documents into a dictionary."""
        logger.info("Loading all documents into memory...")
        documents = {}

        doc_count = 0
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents"):
            # Handle different document field formats
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                title = getattr(doc, 'title', '')
                body = doc.body
                doc_text = f"{title} {body}".strip() if title else body
            else:
                doc_text = str(doc)  # Fallback

            documents[doc.doc_id] = doc_text
            doc_count += 1

            # Progress logging for large collections
            if doc_count % 100000 == 0:
                logger.info(f"Loaded {doc_count} documents...")

        return documents

    def get_document(self, doc_id: str) -> str:
        """Get a single document by ID."""
        return self.documents.get(doc_id, "")


def load_test_data_from_jsonl(jsonl_file: Path) -> Dict[str, Any]:
    """Load test data from JSONL file."""
    logger.info(f"Loading test data from JSONL: {jsonl_file}")

    queries = {}
    features = {}
    candidates = {}

    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)

                query_id = example['query_id']
                queries[query_id] = example['query_text']
                features[query_id] = {'term_features': example['expansion_features']}

                # Extract candidates
                candidate_list = []
                for candidate in example['candidates']:
                    doc_id = candidate['doc_id']
                    score = candidate['score']
                    candidate_list.append((doc_id, score))

                candidates[query_id] = candidate_list

    logger.info(f"Loaded {len(queries)} queries from JSONL")
    return {'queries': queries, 'features': features, 'candidates': candidates}


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
                        document_loader: DocumentLoader,
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
            for doc_id, _ in query_candidates:
                doc_text = document_loader.get_document(doc_id)
                if doc_text:
                    document_texts[doc_id] = doc_text

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate neural reranker on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test.jsonl file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='IR dataset name for document loading')
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

        # Load dataset for documents
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = ir_datasets.load(args.dataset)
        document_loader = DocumentLoader(dataset)

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

        # Move model to device
        reranker = reranker.to(device)
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

        # Log learned weights
        alpha, beta = reranker.get_learned_weights()
        logger.info(f"Loaded model weights: α={alpha:.4f}, β={beta:.4f}")

        # Set model to evaluation mode
        reranker.eval()

        # Run reranker
        with torch.no_grad():
            results = run_neural_reranker(
                reranker=reranker,
                queries=queries,
                features=features,
                candidates=candidates,
                document_loader=document_loader,
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
        logger.info(f"  Total candidates: {total_candidates}")
        logger.info(f"  Average candidates per query: {avg_candidates:.1f}")
        logger.info(f"  Top-k returned: {args.top_k}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()