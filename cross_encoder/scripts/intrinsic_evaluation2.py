#!/usr/bin/env python3
"""
Intrinsic Evaluation: Final Expanded Query Performance

This script evaluates whether the final weighted query from MEQE outperforms
the final weighted query from RM3.
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import json
from scipy.stats import ttest_rel

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.models.reranker import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir, save_json
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

# Import ir_measures for metric calculation
import ir_measures
from ir_measures import *

logger = logging.getLogger(__name__)


class FinalQueryEvaluator:
    """
    Evaluates term rankings by building a final, fully-weighted query.
    """

    def __init__(self, features_file: str, output_dir: Path, qrels_file: str, index_path: str):
        self.features_file = Path(features_file)
        self.output_dir = ensure_dir(output_dir)
        self.qrels_file = qrels_file
        self.index_path = index_path
        self.searcher = None
        self.reader = None
        self.qrels = None

    def _initialize_searcher(self):
        """Initializes the Lucene searcher and loads qrels."""
        if self.searcher: return
        logger.info("Initializing Lucene searcher and loading qrels...")
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        lucene_classes = get_lucene_classes()
        self.FSDirectory = lucene_classes['FSDirectory']
        self.Paths = lucene_classes['Path']
        self.DirectoryReader = lucene_classes['DirectoryReader']
        self.IndexSearcher = lucene_classes['IndexSearcher']
        self.BM25Similarity = lucene_classes['BM25Similarity']

        directory = self.FSDirectory.open(self.Paths.get(self.index_path))
        self.reader = self.DirectoryReader.open(directory)
        reader_context = self.reader.getContext()
        self.searcher = self.IndexSearcher(reader_context)
        self.searcher.setSimilarity(self.BM25Similarity())
        logger.info(f"Lucene index loaded: {self.index_path}")
        self.qrels = ir_measures.read_trec_qrels(self.qrels_file)

    def _tokenize_query_lucene(self, query_text: str) -> List[str]:
        """Tokenize query using Lucene's analyzer."""
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        EnglishAnalyzer = get_lucene_classes()['EnglishAnalyzer']
        CharTermAttribute = get_lucene_classes()['CharTermAttribute']
        analyzer = EnglishAnalyzer()
        tokens = []
        try:
            stream = analyzer.tokenStream("contents", query_text)
            term_attr = stream.addAttribute(CharTermAttribute)
            stream.reset()
            while stream.incrementToken():
                tokens.append(term_attr.toString())
            stream.close()
        except Exception:
            tokens = query_text.lower().split()
        return tokens

    def load_features(self) -> Dict[str, Dict]:
        logger.info("Loading features...")
        features = {}
        with open(self.features_file, 'r') as f:
            for line in tqdm(f, desc="Loading features"):
                if line.strip():
                    data = json.loads(line)
                    features[data['query_id']] = data
        return features

    def evaluate_meqe_model(self, features: Dict[str, Dict], model_checkpoint_path: str,
                            model_config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        logger.info("Getting MEQE weights...")
        model = create_neural_reranker(**model_config)
        import torch
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        alpha, beta, _ = model.get_learned_weights()
        logger.info(f"Model weights: α={alpha:.4f}, β={beta:.4f}")
        meqe_weights = {}
        with torch.no_grad():
            for qid, qdata in features.items():
                meqe_weights[qid] = {
                    term: alpha * tdata['rm_weight'] + beta * tdata['semantic_score']
                    for term, tdata in qdata['term_features'].items()
                }
        return meqe_weights

    def _run_expanded_query(self, query_id: str, original_query_terms: List[str],
                            expansion_terms_with_weights: Dict[str, float], metric: "ir_measures.Measure") -> float:
        """Builds and executes a single, fully-weighted expanded query."""
        from cross_encoder.src.utils.lucene_utils import get_lucene_classes
        BooleanQueryBuilder = get_lucene_classes()['BooleanQueryBuilder']
        TermQuery = get_lucene_classes()['TermQuery']
        BoostQuery = get_lucene_classes()['BoostQuery']
        Term = get_lucene_classes()['Term']
        Occur = get_lucene_classes()['BooleanClauseOccur']

        builder = BooleanQueryBuilder()
        # Add original terms with a default boost of 1.0
        for term in set(original_query_terms):
            builder.add(TermQuery(Term("contents", term)), Occur.SHOULD)

        # Add expansion terms with their learned weights as boosts
        for term, weight in expansion_terms_with_weights.items():
            # Lucene boosts must be positive, so we handle negative weights
            # A simple way is to use a small positive value for negative weights
            boost_value = max(1e-6, weight)
            term_query = TermQuery(Term("contents", term))
            builder.add(BoostQuery(term_query, float(boost_value)), Occur.SHOULD)

        # Execute search
        top_docs = self.searcher.search(builder.build(), 20)

        # Evaluate the results
        if not top_docs.scoreDocs:
            return 0.0

        run_data = [
            ir_measures.ScoredDoc(query_id, self.reader.storedFields().document(sd.doc).get("id"), sd.score)
            for sd in top_docs.scoreDocs
        ]

        qrels_for_query = [q for q in self.qrels if q.query_id == query_id]
        if not qrels_for_query:
            return 0.0  # Return 0 if no relevance judgments exist for this query

        return ir_measures.calc_aggregate([metric], qrels_for_query, run_data).get(metric, 0.0)

    def run_final_query_evaluation(self, features: Dict[str, Dict], meqe_weights: Dict[str, Dict[str, float]],
                                   metric_str: str) -> Dict[str, Any]:
        self._initialize_searcher()
        logger.info("Starting Final Query Evaluation experiment...")

        metric = ir_measures.parse_measure(metric_str)
        all_results = {}

        qrels_query_ids = {q.query_id for q in self.qrels}
        runnable_queries = sorted(list(set(features.keys()) & set(meqe_weights.keys()) & qrels_query_ids))

        logger.info(f"Processing {len(runnable_queries)} queries with relevance judgments.")

        for query_id in tqdm(runnable_queries, desc="Evaluating Final Queries"):
            original_terms = self._tokenize_query_lucene(features[query_id]['query_text'])

            # Get RM3 weights
            rm3_terms_with_weights = {
                term: data['rm_weight']
                for term, data in features[query_id]['term_features'].items()
            }

            # Get MEQE weights
            meqe_terms_with_weights = meqe_weights[query_id]

            # Run and score both final queries
            rm3_score = self._run_expanded_query(query_id, original_terms, rm3_terms_with_weights, metric)
            meqe_score = self._run_expanded_query(query_id, original_terms, meqe_terms_with_weights, metric)

            all_results[query_id] = {'rm3_score': rm3_score, 'meqe_score': meqe_score}

        # --- Aggregate and analyze results ---
        if not all_results: return {"error": "No results were generated."}

        rm3_scores = [res['rm3_score'] for res in all_results.values()]
        meqe_scores = [res['meqe_score'] for res in all_results.values()]

        t_stat, p_value = ttest_rel(meqe_scores, rm3_scores)

        summary = {
            'rm3_mean_score': np.mean(rm3_scores),
            'meqe_mean_score': np.mean(meqe_scores),
            'mean_improvement': np.mean(meqe_scores) - np.mean(rm3_scores),
            'queries_improved': int(np.sum(np.array(meqe_scores) > np.array(rm3_scores))),
            'num_queries': len(all_results),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_improvement': p_value < 0.05 and np.mean(meqe_scores) > np.mean(rm3_scores)
        }

        logger.info(f"--- Final Query Performance Results ({metric_str}) ---")
        logger.info(f"  RM3 Mean Score: {summary['rm3_mean_score']:.4f}")
        logger.info(f"  MEQE Mean Score: {summary['meqe_mean_score']:.4f}")
        logger.info(f"  Mean Improvement: {summary['mean_improvement']:.4f}")
        logger.info(f"  Queries where MEQE > RM3: {summary['queries_improved']} / {summary['num_queries']}")
        logger.info(
            f"  Paired T-test p-value: {summary['p_value']:.4f} ({'Significant' if summary['significant_improvement'] else 'Not Significant'})")

        return {'summary': summary, 'per_query_results': all_results}


def main():
    parser = argparse.ArgumentParser(description="Run Final Query Evaluation of MEQE.")
    # Arguments from original script...
    parser.add_argument("--features-file", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--qrels-file", type=str, required=True)
    parser.add_argument("--lucene-path", type=str, required=True)
    parser.add_argument("--metric", type=str, default="nDCG@20")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max-expansion-terms", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scoring-method", type=str, default="neural")
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--pooling-strategy", type=str, default="cls")
    parser.add_argument('--log-level', type=str, default='INFO')

    args = parser.parse_args()

    setup_experiment_logging("final_query_eval", args.log_level, str(Path(args.output_dir) / 'final_query_eval.log'))

    from cross_encoder.src.utils.lucene_utils import initialize_lucene
    if not initialize_lucene(args.lucene_path): sys.exit(1)

    model_config = {
        'model_name': args.model_name, 'max_expansion_terms': args.max_expansion_terms,
        'hidden_dim': args.hidden_dim, 'dropout': args.dropout,
        'scoring_method': args.scoring_method, 'force_hf': args.force_hf,
        'pooling_strategy': args.pooling_strategy, 'device': 'cpu'
    }

    evaluator = FinalQueryEvaluator(
        features_file=args.features_file, output_dir=Path(args.output_dir),
        qrels_file=args.qrels_file, index_path=args.index_path
    )

    try:
        features = evaluator.load_features()
        meqe_weights = evaluator.evaluate_meqe_model(features, args.model_checkpoint, model_config)
        results = evaluator.run_final_query_evaluation(features, meqe_weights, args.metric)

        results_file = Path(args.output_dir) / 'final_query_evaluation_results.json'
        save_json(results, results_file)
        logger.info(f"Saved final results to {results_file}")

        if 'error' in results: return 1
    except Exception as e:
        logger.error(f"Evaluation failed with an unhandled exception: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())