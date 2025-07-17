#!/usr/bin/env python3
"""
Experiment 1: Visualize Embedding Space Effectiveness (v4 - Diagnostic)

This script generates a 2D visualization to compare the semantic space of a
standard query embedding against an MEQE-enhanced query embedding.

This version adds diagnostics to check the similarity between the original and
enhanced embeddings and provides PCA as an alternative visualization method.
"""

import argparse
import logging
import sys
import torch
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set

# Add project root to path for module imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Third-party libraries
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing required libraries for visualization. Please install them:")
    print("pip install scikit-learn matplotlib seaborn pandas")
    sys.exit(1)

# Project-specific imports
import ir_datasets
from cross_encoder.src.models.reranker2 import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


# --- Helper Functions ---

def get_query_text(query_obj):
    """Extract query text from ir_datasets query object."""
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        if hasattr(query_obj, 'description') and query_obj.description:
            return f"{query_obj.title} {query_obj.description}"
        return query_obj.title
    return ""


def get_doc_text(doc_obj):
    """Extract main text from an ir_datasets document object robustly."""
    if hasattr(doc_obj, 'text'):
        return doc_obj.text
    if hasattr(doc_obj, 'body'):  # Common in TREC collections
        return doc_obj.body
    if hasattr(doc_obj, 'contents'):
        return doc_obj.contents
    # Fallback to converting the object to a string, which often works
    return str(doc_obj)


def load_features(features_file: Path) -> Dict[str, Any]:
    """Load the features file into a dictionary keyed by query_id."""
    features = {}
    with open(features_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            features[data['query_id']] = data['term_features']
    return features


def load_run_file(run_file_path: Path) -> Dict[str, List[str]]:
    """Loads a TREC-style run file into a dictionary mapping query_id to a list of doc_ids."""
    run = defaultdict(list)
    with open(run_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                query_id, _, doc_id, _, _, _ = parts
                run[query_id].append(doc_id)
    return run


def get_relevant_docs(dataset: ir_datasets.Dataset, query_id: str, run_file_docs: List[str]) -> Tuple[
    Set[str], List[str]]:
    """
    Get sets of relevant and non-relevant document IDs for a query using the efficient
    qrels_dict() method. Non-relevant docs are sourced from the provided run file.
    """
    all_qrels = dataset.qrels_dict()
    query_qrels = all_qrels.get(query_id, {})
    relevant_doc_ids = {doc_id for doc_id, rel in query_qrels.items() if rel > 0}
    non_relevant_doc_ids = [doc_id for doc_id in run_file_docs if doc_id not in relevant_doc_ids]
    return relevant_doc_ids, non_relevant_doc_ids


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the effectiveness of MEQE-enhanced query embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments ---
    parser.add_argument('--features-file', type=str, required=True, help='Path to the features JSONL file.')
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='Path to the trained model checkpoint (.pt).')
    parser.add_argument('--run-file', type=str, required=True, help='Path to the 1st stage TREC-style run file.')
    parser.add_argument('--dataset', type=str, required=True, help='ir_datasets name.')
    parser.add_argument('--query-id', type=str, required=True, help='The specific query ID to visualize.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output plots.')
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                        help='Name of the sentence transformer model used.')
    parser.add_argument('--num-relevant-docs', type=int, default=10, help='Number of relevant documents to plot.')
    parser.add_argument('--num-non-relevant-docs', type=int, default=10,
                        help='Number of non-relevant documents to plot.')
    parser.add_argument('--dim-reduction', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction technique.')
    args = parser.parse_args()

    # --- Setup ---
    output_dir = ensure_dir(args.output_dir)
    setup_experiment_logging("visualize_embedding_space", "INFO",
                             str(output_dir / f'visualization_{args.query_id}.log'))
    log_experiment_info(logger, **vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Load Data and Model ---
    with TimedOperation(logger, "Loading data and models"):
        features = load_features(Path(args.features_file))
        run_file_data = load_run_file(Path(args.run_file))
        dataset = ir_datasets.load(args.dataset)

        # NOTE: The parameters used here to create the model instance must match
        # the parameters of the model you are loading.
        reranker = create_neural_reranker(
            model_name='all-MiniLM-L6-v2',
            max_expansion_terms=15,
            hidden_dim=384,
            dropout=0.1,
            scoring_method='cosine',
            device=device
        )
        reranker.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        reranker.to(device)  # Ensure model is on the correct device
        reranker.eval()

        query_text = next((get_query_text(q) for q in dataset.queries_iter() if q.query_id == args.query_id), None)
        if not query_text:
            logger.error(f"Query ID {args.query_id} not found in dataset.")
            sys.exit(1)

        run_file_docs_for_query = run_file_data.get(args.query_id, [])
        if not run_file_docs_for_query:
            logger.error(f"Query ID {args.query_id} not found in the provided run file.")
            sys.exit(1)

        relevant_doc_ids, non_relevant_doc_ids = get_relevant_docs(dataset, args.query_id, run_file_docs_for_query)
        if not relevant_doc_ids:
            logger.warning(f"No relevant documents found for query {args.query_id}. Plot may not be meaningful.")

    # --- 2. Generate Embeddings & Run Diagnostics ---
    with TimedOperation(logger, "Generating embeddings and running diagnostics"):
        original_query_emb = reranker.encode_text(query_text).cpu().numpy()
        enhanced_query_emb = reranker.get_enhanced_query_embedding(query_text,
                                                                   features.get(args.query_id, {})).cpu().numpy()

        # <<< DIAGNOSTICS START >>>
        alpha, beta, _ = reranker.get_learned_weights()
        logger.info(f"DIAGNOSTIC: Learned Alpha = {alpha:.6f}, Learned Beta = {beta:.6f}")

        cos_sim = np.dot(original_query_emb, enhanced_query_emb) / (
                    np.linalg.norm(original_query_emb) * np.linalg.norm(enhanced_query_emb))
        logger.info(f"DIAGNOSTIC: Cosine Similarity between original and enhanced query = {cos_sim:.6f}")

        if cos_sim > 0.999:
            logger.warning(
                "WARNING: Embeddings are extremely similar. The visualization will likely show no significant change.")
        # <<< DIAGNOSTICS END >>>

        docs_store = dataset.docs_store()
        rel_docs_to_plot = list(relevant_doc_ids)[:args.num_relevant_docs]
        non_rel_docs_to_plot = list(non_relevant_doc_ids)[:args.num_non_relevant_docs]
        doc_ids_for_plot = rel_docs_to_plot + non_rel_docs_to_plot

        doc_texts_to_encode = [get_doc_text(docs_store.get(doc_id)) for doc_id in doc_ids_for_plot]
        doc_embeddings = reranker.encode_terms(doc_texts_to_encode).cpu().numpy()

    # --- 3. Dimensionality Reduction ---
    with TimedOperation(logger, f"Performing {args.dim_reduction.upper()} reduction"):
        reducer = TSNE if args.dim_reduction == 'tsne' else PCA

        baseline_embeddings = np.vstack([original_query_emb, doc_embeddings])
        reducer_baseline = reducer(n_components=2, random_state=42)
        baseline_2d = reducer_baseline.fit_transform(baseline_embeddings)

        meqe_embeddings = np.vstack([enhanced_query_emb, doc_embeddings])
        reducer_meqe = reducer(n_components=2, random_state=42)
        meqe_2d = reducer_meqe.fit_transform(meqe_embeddings)

    # --- 4. Create Plots ---
    with TimedOperation(logger, "Creating visualizations"):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), sharey=True, sharex=True)

        plot_labels = ['Relevant' if doc_id in relevant_doc_ids else 'Non-Relevant' for doc_id in doc_ids_for_plot]

        # Plot 1: Baseline Space
        orig_q_2d, baseline_docs_2d = baseline_2d[0], baseline_2d[1:]
        baseline_df = pd.DataFrame(baseline_docs_2d, columns=['x', 'y'])
        baseline_df['label'] = plot_labels

        ax1.set_title('Baseline: Original Query Embedding Space', fontsize=15, fontweight='bold')
        sns.scatterplot(data=baseline_df, x='x', y='y', hue='label',
                        palette={'Relevant': 'blue', 'Non-Relevant': 'red'}, ax=ax1, s=60, alpha=0.8, style='label',
                        markers={'Relevant': 'o', 'Non-Relevant': 'P'})
        ax1.scatter(orig_q_2d[0], orig_q_2d[1], c='green', marker='X', s=250, label='Original Query',
                    edgecolors='black', zorder=5)

        # Plot 2: MEQE Space
        enhanced_q_2d, meqe_docs_2d = meqe_2d[0], meqe_2d[1:]
        meqe_df = pd.DataFrame(meqe_docs_2d, columns=['x', 'y'])
        meqe_df['label'] = plot_labels

        ax2.set_title('MEQE: Enhanced Query Embedding Space', fontsize=15, fontweight='bold')
        sns.scatterplot(data=meqe_df, x='x', y='y', hue='label', palette={'Relevant': 'blue', 'Non-Relevant': 'red'},
                        ax=ax2, s=60, alpha=0.8, style='label', markers={'Relevant': 'o', 'Non-Relevant': 'P'})
        ax2.scatter(enhanced_q_2d[0], enhanced_q_2d[1], c='lime', marker='*', s=350, label='Enhanced Query',
                    edgecolors='black', zorder=5)

        for ax in [ax1, ax2]:
            ax.legend(title='Document Type', fontsize=12)
            ax.set_xlabel('Dimension 1', fontsize=12)
        ax1.set_ylabel('Dimension 2', fontsize=12)
        ax2.set_ylabel('')

        fig.suptitle(f'Embedding Space Comparison for Query ID: {args.query_id}', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = output_dir / f"embedding_space_query_{args.query_id}.png"
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Visualization saved to: {plot_path}")

    logger.info("Experiment finished successfully!")


if __name__ == '__main__':
    main()