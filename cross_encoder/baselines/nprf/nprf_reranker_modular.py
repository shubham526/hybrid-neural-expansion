#!/usr/bin/env python3
"""
Modular NPRF Reranker

Clean, modular inference script using the refactored components.
"""

import argparse
import os
import logging
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from nprf_core import (
    SimilarityComputer, NPRFFeatureExtractor, ModelFactory, 
    load_jsonl, write_trec_run
)
from nprf_data import load_and_preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NPRFReranker:
    """Modular NPRF reranker for inference."""
    
    def __init__(self, model_path=None, model_type="drmm", device=None, **model_kwargs):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path and os.path.exists(model_path):
            # Load trained model
            self.model, checkpoint = ModelFactory.load_model(model_path, self.device)
            self.model_type = checkpoint['model_type']
            self.config = checkpoint['args']
            
            # Initialize feature extractor from config
            similarity_computer = SimilarityComputer(device=self.device)
            self.feature_extractor = NPRFFeatureExtractor(
                model_type=self.model_type,
                similarity_computer=similarity_computer,
                **checkpoint.get('feature_extractor_config', {})
            )
            
            logger.info(f"Loaded trained model: {model_path}")
            
        else:
            # Create untrained model
            if model_path:
                logger.warning(f"Model path {model_path} not found. Using untrained model.")
            
            self.model_type = model_type
            self.config = model_kwargs
            
            # Initialize components
            similarity_computer = SimilarityComputer(device=self.device)
            self.feature_extractor = NPRFFeatureExtractor(
                model_type=model_type,
                similarity_computer=similarity_computer,
                **model_kwargs
            )
            
            self.model = ModelFactory.create_model(model_type, **model_kwargs)
            self.model.to(self.device)
            self.model.eval()
            
            logger.warning("Using untrained model - results may be poor")
    
    def rerank_query(self, query_data, score_combination_weight=0.7):
        """Rerank candidates for a single query."""
        query_id = query_data['query_id']
        query_text = query_data['query_text']
        candidates = query_data['candidates']
        
        # Check minimum requirements
        min_docs = self.feature_extractor.nb_supervised_doc
        if len(candidates) < min_docs:
            logger.warning(f"Query {query_id}: Not enough candidates ({len(candidates)} < {min_docs})")
            return self._return_original_ranking(candidates)
        
        try:
            # Select PRF documents
            prf_docs = candidates[:min_docs]
            
            # Extract features and score each candidate
            scores = []
            for candidate in candidates:
                features = self.feature_extractor.extract_features(query_text, candidate, prf_docs)
                score = self._score_candidate(features)
                scores.append(score)
            
            # Combine with original scores
            final_scores = []
            for i, (candidate, nprf_score) in enumerate(zip(candidates, scores)):
                combined_score = (score_combination_weight * candidate['score'] + 
                                (1 - score_combination_weight) * nprf_score)
                final_scores.append((candidate['doc_id'], combined_score))
            
            # Sort and assign ranks
            final_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_results = [
                {
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'rank': rank + 1,
                    'score': float(score)
                }
                for rank, (doc_id, score) in enumerate(final_scores)
            ]
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            return self._return_original_ranking(candidates)
    
    def _score_candidate(self, features):
        """Score a single candidate using the model."""
        self.model.eval()
        
        with torch.no_grad():
            if len(features) == 3:  # DRMM
                dd_q, dd_d, doc_scores = features
                dd_q = dd_q.unsqueeze(0).to(self.device)
                dd_d = dd_d.unsqueeze(0).to(self.device)
                doc_scores = doc_scores.unsqueeze(0).to(self.device)
                score = self.model(dd_q, dd_d, doc_scores)
            else:  # K-NRM
                dd, doc_scores = features
                dd = dd.unsqueeze(0).to(self.device)
                doc_scores = doc_scores.unsqueeze(0).to(self.device)
                score = self.model(dd, doc_scores)
            
            return score.item()
    
    def _return_original_ranking(self, candidates):
        """Return original ranking when reranking fails."""
        return [
            {
                'query_id': candidates[0].get('query_id', 'unknown'),
                'doc_id': c['doc_id'],
                'rank': i + 1,
                'score': float(c['score'])
            }
            for i, c in enumerate(candidates)
        ]


def process_single_query(query_data, reranker):
    """Process a single query with the reranker."""
    query_id = query_data['query_id']
    
    try:
        results = reranker.rerank_query(query_data)
        
        return {
            'query_id': query_id,
            'success': True,
            'results': results,
            'num_results': len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to process query {query_id}: {e}")
        return {
            'query_id': query_id,
            'success': False,
            'error': str(e),
            'results': []
        }


def main():
    parser = argparse.ArgumentParser(
        description="NPRF reranker (modular version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test data JSONL file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for TREC run file')
    
    # Model arguments
    parser.add_argument('--model-path', type=str,
                        help='Path to trained NPRF model (.pt file)')
    parser.add_argument('--model-type', type=str, default='drmm',
                        choices=['drmm', 'knrm'],
                        help='Type of neural IR model (for untrained models)')
    
    # Model hyperparameters (for untrained models)
    parser.add_argument('--nb-supervised-doc', type=int, default=10,
                        help='Number of PRF documents')
    parser.add_argument('--doc-topk-term', type=int, default=20,
                        help='Number of top terms per document')
    parser.add_argument('--hist-size', type=int, default=30,
                        help='Histogram size for DRMM')
    parser.add_argument('--kernel-size', type=int, default=11,
                        help='Number of kernels for K-NRM')
    parser.add_argument('--hidden-size', type=int, default=5,
                        help='Hidden layer size')
    
    # Inference arguments
    parser.add_argument('--score-combination-weight', type=float, default=0.7,
                        help='Weight for original scores in combination (0-1)')
    parser.add_argument('--run-name', type=str, default='nprf',
                        help='Run name for TREC output')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Number of parallel workers')
    
    # Data preprocessing
    parser.add_argument('--min-candidates', type=int, default=5,
                        help='Minimum candidates per query')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and preprocess test data
    logger.info(f"Loading test data from: {args.test_file}")
    preprocess_config = {
        'min_candidates': args.min_candidates,
        'require_positive': False,  # Test data may not have relevance labels
        'require_negative': False,
        'balance_dataset': False
    }
    
    datasets = load_and_preprocess_data(
        train_file=None,
        test_file=args.test_file,
        preprocess_config=preprocess_config
    )
    
    test_data = datasets['test']
    logger.info(f"Loaded {len(test_data)} test queries")
    
    # Initialize reranker
    logger.info("Initializing NPRF reranker...")
    model_kwargs = {
        'nb_supervised_doc': args.nb_supervised_doc,
        'doc_topk_term': args.doc_topk_term,
        'hist_size': args.hist_size,
        'kernel_size': args.kernel_size,
        'hidden_size': args.hidden_size
    }
    
    reranker = NPRFReranker(
        model_path=args.model_path,
        model_type=args.model_type,
        **model_kwargs
    )
    
    logger.info(f"Starting NPRF reranking...")
    logger.info(f"  Model type: {reranker.model_type}")
    logger.info(f"  Score combination weight: {args.score_combination_weight}")
    logger.info(f"  Max workers: {args.max_workers}")
    
    # Process queries
    all_query_results = []
    successful_queries = 0
    failed_queries = 0
    
    if args.max_workers == 1:
        # Single-threaded processing
        for query_data in test_data:
            result = process_single_query(query_data, reranker)
            all_query_results.append(result)
            
            if result['success']:
                successful_queries += 1
                logger.info(f"✓ {result['query_id']}: {result['num_results']} results")
            else:
                failed_queries += 1
                logger.error(f"✗ {result['query_id']}: {result['error']}")
    else:
        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(process_single_query, query_data, reranker): query_data['query_id']
                for query_data in test_data
            }
            
            # Collect results
            for future in as_completed(future_to_query):
                query_id = future_to_query[future]
                
                try:
                    result = future.result()
                    all_query_results.append(result)
                    
                    if result['success']:
                        successful_queries += 1
                        logger.info(f"✓ {query_id}: {result['num_results']} results")
                    else:
                        failed_queries += 1
                        logger.error(f"✗ {query_id}: {result['error']}")
                        
                except Exception as e:
                    failed_queries += 1
                    logger.error(f"✗ {query_id}: {e}")
    
    # Collect and write results
    all_results = []
    for query_result in all_query_results:
        if query_result['success']:
            all_results.extend(query_result['results'])
    
    if all_results:
        # Sort by query_id, then by rank
        all_results.sort(key=lambda x: (x['query_id'], x['rank']))
        
        # Write TREC run file
        trec_file = output_dir / f"{args.run_name}.trec"
        write_trec_run(all_results, trec_file, args.run_name)
        
        logger.info(f"\n✅ NPRF reranking completed!")
        logger.info(f"   Successful queries: {successful_queries}")
        logger.info(f"   Failed queries: {failed_queries}")
        logger.info(f"   Total results: {len(all_results)}")
        logger.info(f"   TREC run file: {trec_file}")
    else:
        logger.error("❌ No results generated!")


if __name__ == "__main__":
    main()