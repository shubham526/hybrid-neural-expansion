"""
Feature Extractor for RM3 + Semantic Similarity

Extracts the two core features needed for learning:
1. RM3 weights for expansion terms
2. Semantic similarity scores (cosine similarity with query)
"""

import logging
from typing import Dict, List, Tuple, Optional

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity

logger = logging.getLogger(__name__)


class ExpansionFeatureExtractor:
    """Extract RM3 and semantic similarity features for query expansion."""

    def __init__(self, config: dict):
        """Initialize the feature extractor with configuration."""
        self.config = config
        self.index_path = config['index_path']
        self.max_expansion_terms = config.get('max_expansion_terms', 10)
        self.top_k_pseudo_docs = config.get('top_k_pseudo_docs', 10)
        self.rm_alpha = config.get('rm_alpha', 0.5)

        self.rm_expansion = RMExpansion(
            index_path=self.index_path,
            k1=self.config.get('k1', 1.2),
            b=self.config.get('b', 0.75)
        )

        self.semantic_sim = SemanticSimilarity(config.get('embedding_model'))
        logger.info("ExpansionFeatureExtractor initialized.")

    def extract_features_for_query(self,
                                   query_id: str,
                                   query_text: str) -> Optional[Dict[str, any]]:
        """
        Extract features for a single query.

        Args:
            query_id: Query identifier
            query_text: Query text

        Returns:
            A dictionary containing all features for the query, or None if extraction fails.
        """
        try:
            # Step 1: Get RM3 expansion terms and original mappings.
            # This is self-contained now; it performs its own search.
            rm_terms, original_mappings = self.rm_expansion.expand_query_with_originals(
                query=query_text,
                num_expansion_terms=self.max_expansion_terms,
                num_feedback_docs=self.top_k_pseudo_docs,
                alpha=self.rm_alpha
            )

            if not rm_terms:
                logger.warning(f"No RM3 terms could be generated for query {query_id}")
                return None

            # Step 2: Use the original (unstemmed) words to get semantic scores.
            expansion_terms = [term for term, _ in rm_terms]
            original_terms_for_similarity = [original_mappings.get(term, term) for term in expansion_terms]

            semantic_scores = self.semantic_sim.compute_query_expansion_similarities(
                query_text, original_terms_for_similarity
            )

            # Step 3: Combine all features into a final dictionary.
            term_features = {}
            for i, (stemmed_term, rm_weight) in enumerate(rm_terms):
                original_term = original_terms_for_similarity[i]
                semantic_score = semantic_scores.get(original_term, 0.0)

                term_features[stemmed_term] = {
                    'rm_weight': float(rm_weight),
                    'semantic_score': float(semantic_score),
                    'original_term': original_term
                }

            query_features = {
                'query_id': query_id,
                'query_text': query_text,
                'term_features': term_features,
                'num_expansion_terms': len(term_features)
            }

            logger.debug(f"Extracted features for query {query_id}: {len(term_features)} terms")
            return query_features

        except Exception as e:
            logger.error(f"Error extracting features for query {query_id}: {e}", exc_info=True)
            return None

    def close(self):
        """Closes the underlying RMExpansion module to release resources."""
        logger.info("Closing ExpansionFeatureExtractor resources...")
        if hasattr(self, 'rm_expansion'):
            self.rm_expansion.close()