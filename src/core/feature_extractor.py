"""
Feature Extractor for RM3 + Semantic Similarity

Extracts the two core features needed for learning:
1. RM3 weights for expansion terms
2. Semantic similarity scores (cosine similarity with query)
"""

import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity

logger = logging.getLogger(__name__)


class ExpansionFeatureExtractor:
    """Extract RM3 and semantic similarity features for query expansion."""

    def __init__(self,
                 rm_expansion: RMExpansion,
                 semantic_similarity: SemanticSimilarity,
                 max_expansion_terms: int = 15,
                 top_k_pseudo_docs: int = 10):
        """
        Initialize feature extractor.

        Args:
            rm_expansion: RM3 expansion component
            semantic_similarity: Semantic similarity component
            max_expansion_terms: Maximum expansion terms per query
            top_k_pseudo_docs: Number of pseudo-relevant documents
        """
        self.rm_expansion = rm_expansion
        self.semantic_similarity = semantic_similarity
        self.max_expansion_terms = max_expansion_terms
        self.top_k_pseudo_docs = top_k_pseudo_docs

        logger.info("ExpansionFeatureExtractor initialized")

    def extract_features_for_query(self,
                                   query_id: str,
                                   query_text: str,
                                   pseudo_relevant_docs: List[str],
                                   pseudo_relevant_scores: List[float]) -> Optional[Dict[str, any]]:
        """
        Extract features for a single query.

        Args:
            query_id: Query identifier
            query_text: Query text
            pseudo_relevant_docs: List of pseudo-relevant document texts
            pseudo_relevant_scores: Relevance scores for pseudo-relevant docs

        Returns:
            Feature dictionary or None if extraction fails
        """
        try:
            if not pseudo_relevant_docs:
                logger.warning(f"No pseudo-relevant docs for query {query_id}")
                return None

            # Step 1: Get RM3 expansion terms and weights
            rm_terms = self.rm_expansion.expand_query(
                query=query_text,
                documents=pseudo_relevant_docs,
                scores=pseudo_relevant_scores,
                num_expansion_terms=self.max_expansion_terms,
                rm_type="rm3"
            )

            if not rm_terms:
                logger.warning(f"No RM3 terms for query {query_id}")
                return None

            # Step 2: Extract terms and compute semantic similarities
            expansion_terms = [term for term, _ in rm_terms]
            semantic_scores = self.semantic_similarity.compute_query_expansion_similarities(
                query_text, expansion_terms
            )

            # Step 3: Combine into feature dictionary
            term_features = {}
            for term, rm_weight in rm_terms:
                semantic_score = semantic_scores.get(term, 0.0)

                term_features[term] = {
                    'rm_weight': float(rm_weight),
                    'semantic_score': float(semantic_score)
                }

            query_features = {
                'query_id': query_id,
                'query_text': query_text,
                'term_features': term_features,
                'num_pseudo_docs': len(pseudo_relevant_docs),
                'num_expansion_terms': len(term_features)
            }

            logger.debug(f"Extracted features for query {query_id}: {len(term_features)} terms")
            return query_features

        except Exception as e:
            logger.error(f"Error extracting features for query {query_id}: {e}")
            return None

    def extract_features_for_dataset(self,
                                     queries: Dict[str, str],
                                     first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                     document_collection: Dict[str, str]) -> Dict[str, Dict[str, any]]:
        """
        Extract features for entire dataset.

        Args:
            queries: {query_id: query_text}
            first_stage_runs: {query_id: [(doc_id, score), ...]}
            document_collection: {doc_id: doc_text}

        Returns:
            {query_id: feature_dict}
        """
        all_features = {}

        for query_id, query_text in queries.items():
            if query_id not in first_stage_runs:
                logger.warning(f"No first-stage run for query {query_id}")
                continue

            # Get pseudo-relevant documents
            top_docs = first_stage_runs[query_id][:self.top_k_pseudo_docs]

            pseudo_docs = []
            pseudo_scores = []

            for doc_id, score in top_docs:
                if doc_id in document_collection:
                    pseudo_docs.append(document_collection[doc_id])
                    pseudo_scores.append(score)

            if not pseudo_docs:
                logger.warning(f"No valid pseudo-docs for query {query_id}")
                continue

            # Extract features
            features = self.extract_features_for_query(
                query_id, query_text, pseudo_docs, pseudo_scores
            )

            if features:
                all_features[query_id] = features

        logger.info(f"Extracted features for {len(all_features)} queries")
        return all_features

    def get_feature_statistics(self, all_features: Dict[str, Dict[str, any]]) -> Dict[str, any]:
        """Compute statistics about extracted features."""
        if not all_features:
            return {}

        num_queries = len(all_features)
        num_terms_per_query = [len(features['term_features']) for features in all_features.values()]

        all_rm_weights = []
        all_semantic_scores = []

        for features in all_features.values():
            for term_data in features['term_features'].values():
                all_rm_weights.append(term_data['rm_weight'])
                all_semantic_scores.append(term_data['semantic_score'])

        import numpy as np

        stats = {
            'num_queries': num_queries,
            'avg_terms_per_query': np.mean(num_terms_per_query),
            'total_unique_terms': len(set().union(*[
                set(f['term_features'].keys()) for f in all_features.values()
            ])),
            'rm_weight_stats': {
                'mean': np.mean(all_rm_weights),
                'std': np.std(all_rm_weights),
                'min': np.min(all_rm_weights),
                'max': np.max(all_rm_weights)
            },
            'semantic_score_stats': {
                'mean': np.mean(all_semantic_scores),
                'std': np.std(all_semantic_scores),
                'min': np.min(all_semantic_scores),
                'max': np.max(all_semantic_scores)
            }
        }

        return stats