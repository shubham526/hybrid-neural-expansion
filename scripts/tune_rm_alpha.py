#!/usr/bin/env python3
"""
Tune the RM3 alpha Hyperparameter (Self-Contained Version)

This script automates finding the optimal `alpha` for the RM3 model.
It now handles the final search execution directly using PyLucene,
relying on the rm_expansion module only for generating expansion terms.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm
import pytrec_eval

from src.core.rm_expansion import RMExpansion
from scripts.create_train_test_data import get_query_text
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation
from src.utils.lucene_utils import initialize_lucene, get_lucene_classes

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Tune the RM3 alpha hyperparameter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Core arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='IR dataset name (e.g., disks45/nocr/trec-robust-2004)')
    parser.add_argument('--index-path', type=str, required=True, help='Path to the Lucene index')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save runs and results')
    parser.add_argument('--lucene-path', type=str, required=True, help='Path to Lucene JAR files')

    # Tuning arguments
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='List of alpha values to test')
    parser.add_argument('--top-k-retrieval', type=int, default=1000,
                        help='Number of documents to retrieve for evaluation')
    parser.add_argument('--eval-metric', type=str, default='ndcg_cut_10',
                        help='pytrec_eval metric to optimize (e.g., ndcg_cut_10, map)')

    args = parser.parse_args()

    # Setup directories and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_experiment_logging("tune_alpha", "INFO", str(output_dir / 'tuning.log'))
    log_experiment_info(logger, **vars(args))

    # Initialize Lucene once
    if not initialize_lucene(args.lucene_path):
        raise RuntimeError("Failed to initialize Lucene")

    # Load Lucene classes needed for searching
    lucene_classes = get_lucene_classes()
    FSDirectory = lucene_classes['FSDirectory']
    Paths = lucene_classes['Path']
    DirectoryReader = lucene_classes['DirectoryReader']
    IndexSearcher = lucene_classes['IndexSearcher']
    BM25Similarity = lucene_classes['BM25Similarity']
    BooleanQueryBuilder = lucene_classes['BooleanQueryBuilder']
    TermQuery = lucene_classes['TermQuery']
    Term = lucene_classes['Term']
    Occur = lucene_classes['BooleanClauseOccur']
    BoostQuery = lucene_classes['BoostQuery']

    # Load queries and qrels directly from ir_datasets
    logger.info(f"Loading dataset {args.dataset}...")
    dataset = ir_datasets.load(args.dataset)
    # Load queries
    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = get_query_text(query)
    logger.info(f"Loaded {len(queries)} queries")
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    logger.info(f"Loaded {len(queries)} queries and qrels for {len(qrels)} queries.")

    results = {}

    # Create a single RMExpansion object to generate terms
    rm_expansion = RMExpansion(args.index_path)

    # Create a single IndexSearcher to execute searches
    directory = FSDirectory.open(Paths.get(args.index_path))
    reader = DirectoryReader.open(directory)
    reader_context = reader.getContext()
    searcher = IndexSearcher(reader_context)
    searcher.setSimilarity(BM25Similarity())

    logger.info(f"Index initialized with: {args.index_path}")

    try:
        for alpha in tqdm(args.alphas, desc="Tuning Alphas"):
            with TimedOperation(logger, f"Processing alpha = {alpha}"):
                run_file = output_dir / f"run_alpha_{alpha:.1f}.txt"

                logger.info(f"Running retrieval for {len(queries)} queries with alpha={alpha}...")
                with open(run_file, 'w') as f_run:
                    for qid, q_text in tqdm(queries.items(), desc="Retrieval", leave=False):
                        # Step 1: Get weighted expansion terms from our module
                        rm_terms, _ = rm_expansion.expand_query_with_originals(
                            query=q_text,
                            alpha=alpha
                        )

                        # Step 2: Build the weighted query directly in this script
                        builder = BooleanQueryBuilder()
                        for term, weight in rm_terms:
                            term_query = TermQuery(Term("contents", term))
                            boosted_query = BoostQuery(term_query, weight)
                            builder.add(boosted_query, Occur.SHOULD)

                        final_query = builder.build()

                        # Step 3: Execute the search
                        top_docs = searcher.search(final_query, args.top_k_retrieval)

                        for i, score_doc in enumerate(top_docs.scoreDocs):
                            # The 'id' field must exist in your Lucene index
                            doc_id = reader.storedFields().document(score_doc.doc).get("id")
                            if doc_id:
                                f_run.write(f'{qid} Q0 {doc_id} {i + 1} {score_doc.score:.6f} RM3-alpha{alpha}\n')

                # Evaluate the run file
                logger.info("Evaluating run file...")
                with open(run_file, 'r') as f_run:
                    run_data = pytrec_eval.parse_run(f_run)

                evaluator = pytrec_eval.RelevanceEvaluator(qrels, {args.eval_metric})
                eval_results = evaluator.evaluate(run_data)

                metric_sum = sum(res[args.eval_metric] for res in eval_results.values())
                avg_metric = metric_sum / len(eval_results) if eval_results else 0
                results[alpha] = avg_metric
                logger.info(f"Result for alpha={alpha}: {args.eval_metric} = {avg_metric:.4f}")
    finally:
        logger.info("Closing resources.")
        rm_expansion.close()
        reader.close()
        directory.close()

    # Final Step: Report the best alpha
    logger.info("\n" + "=" * 40)
    logger.info("Tuning Complete. Results:")
    logger.info(f"{'Alpha':<10} | {args.eval_metric}")
    logger.info("-" * 40)
    for alpha, score in sorted(results.items()):
        logger.info(f"{alpha:<10.1f} | {score:.4f}")
    logger.info("=" * 40)

    if results:
        best_alpha = max(results, key=results.get)
        logger.info(f"🏆 Best Alpha: {best_alpha} (Score: {results[best_alpha]:.4f})")
    else:
        logger.warning("No results were generated during tuning.")
    logger.info("=" * 40)


if __name__ == '__main__':
    main()