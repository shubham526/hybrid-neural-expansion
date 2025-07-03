"""
RM (Relevance Model) Expansion Module - Lucene Backend

Clean interface to Lucene-based RM1 and RM3 query expansion.
Replaces the previous Python implementation with proven Java/Lucene code.

Author: Your Name
"""

import logging
import json
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from src.utils.lucene_utils import get_lucene_classes

logger = logging.getLogger(__name__)



class LuceneRM3Scorer:
    """
    Lucene-based RM3 implementation using PyJnius.
    Direct translation of the Java relevanceModel() method.
    """

    def __init__(self, index_path: str, k1: float = 1.2, b: float = 0.75):
        """
        Initialize Lucene RM3 scorer.

        Args:
            index_path: Path to Lucene index
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        try:
            self.index_path = index_path

            # Get Lucene classes lazily
            classes = get_lucene_classes()
            for name, cls in classes.items():
                setattr(self, name, cls)

            # Open index and setup searcher
            directory = self.FSDirectory.open(self.Path.get(index_path))
            self.reader = self.DirectoryReader.open(directory)
            self.searcher = self.IndexSearcher(self.reader)
            self.searcher.setSimilarity(self.BM25Similarity(k1, b))

            # Setup analyzer (same as used for indexing)
            self.analyzer = self.EnglishAnalyzer()

            logger.info(f"LuceneRM3Scorer initialized with index: {index_path}")

        except Exception as e:
            logger.error(f"Error initializing LuceneRM3Scorer: {e}")
            raise

    def compute_rm3_expansion(self,
                              query: str,
                              num_feedback_docs: int = 10,
                              num_expansion_terms: int = 20,
                              omit_query_terms: bool = False) -> List[Tuple[str, float]]:
        """
        Compute RM3 query expansion using Lucene (direct translation of Java relevanceModel).

        Args:
            query: Original query string
            num_feedback_docs: Number of pseudo-relevant documents to use
            num_expansion_terms: Number of expansion terms to return
            omit_query_terms: If True, return RM1 (no original query terms)

        Returns:
            List of (term, weight) tuples sorted by weight (descending)
        """
        try:
            logger.debug(f"Computing RM3 for query: '{query}'")

            # Step 1: Build initial query and search (Java: booleanQuery = queryBuilder.toQuery(queryStr))
            initial_query = self._build_boolean_query(query)
            top_docs = self.searcher.search(initial_query, num_feedback_docs)

            if top_docs.totalHits.value() == 0:
                logger.warning(f"No documents found for query: {query}")
                return []

            logger.debug(f"Found {top_docs.totalHits.value()} documents for feedback")

            # Step 2: Extract terms and compute RM3 weights (Java: relevanceModel logic)
            term_weights = self._compute_rm3_weights(
                query, top_docs.scoreDocs, omit_query_terms
            )

            # Step 3: Sort and return top terms (Java: allWordFreqs.sort + subList)
            sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
            result = sorted_terms[:num_expansion_terms]

            logger.debug(f"RM3 expansion completed: {len(result)} terms")
            return result

        except Exception as e:
            logger.error(f"Error in RM3 expansion: {e}")
            return []

    def _build_boolean_query(self, query_str: str):
        """Build initial boolean query from query string (Java: queryBuilder.toQuery)."""
        try:
            query_terms = self._tokenize_query(query_str)

            if not query_terms:
                raise ValueError(f"No valid terms in query: {query_str}")

            builder = self.BooleanQueryBuilder()

            for term in query_terms:
                term_query = self.TermQuery(self.Term("contents", term))
                builder.add(term_query, self.BooleanClauseOccur.SHOULD)

            return builder.build()

        except Exception as e:
            logger.error(f"Error building query: {e}")
            raise

    def _tokenize_query(self, query_str: str) -> List[str]:
        """Tokenize query string using Lucene analyzer (Java: tokenizeQuery)."""
        try:
            tokens = []
            token_stream = self.analyzer.tokenStream("contents", self.StringReader(query_str))
            char_term_attr = token_stream.addAttribute(self.CharTermAttribute)

            token_stream.reset()
            while token_stream.incrementToken():
                token = char_term_attr.toString()
                tokens.append(token)

            token_stream.end()
            token_stream.close()

            return tokens

        except Exception as e:
            logger.error(f"Error tokenizing query: {e}")
            return query_str.lower().split()  # Fallback

    def _compute_rm3_weights(self,
                            query_str: str,
                            score_docs,
                            omit_query_terms: bool) -> Dict[str, float]:
        """
        Compute RM3 weights following the Java implementation exactly.
        Direct translation of the Java relevanceModel() method.
        """
        try:
            term_weights = defaultdict(float)

            # Step 1: Add original query terms with weight 1.0 (Java: queryBuilder.addTokens(queryStr, 1.0f, wordFreq))
            if not omit_query_terms:
                query_tokens = self._tokenize_query(query_str)
                for token in query_tokens:
                    term_weights[token] = 1.0

                logger.debug(f"Added {len(query_tokens)} query terms with weight 1.0")

            # Step 2: Determine if we have log scores (Java: useLog detection)
            use_log = any(score_doc.score < 0.0 for score_doc in score_docs)

            # Step 3: Compute score normalizer (Java: normalizer computation)
            normalizer = 0.0
            for score_doc in score_docs:
                if use_log:
                    normalizer += np.exp(score_doc.score)
                else:
                    normalizer += score_doc.score

            if use_log:
                normalizer = np.log(normalizer) if normalizer > 0 else 1.0

            # Step 4: Process each pseudo-relevant document (Java: for loop over scoreDoc)
            for score_doc in score_docs:
                # Compute document weight (Java: weight computation)
                if use_log:
                    doc_weight = score_doc.score - normalizer if normalizer > 0 else 0.0
                else:
                    doc_weight = score_doc.score / normalizer if normalizer > 0 else 0.0

                # Get document content (Java: searcher.doc(score.doc).get(expansionField))
                doc = self.searcher.doc(score_doc.doc)
                doc_content = doc.get("contents")

                if doc_content:
                    # Tokenize document and add weighted terms (Java: queryBuilder.addTokens)
                    self._add_document_terms(doc_content, doc_weight, term_weights)

            logger.debug(f"Processed {len(score_docs)} documents, total terms: {len(term_weights)}")
            return dict(term_weights)

        except Exception as e:
            logger.error(f"Error computing RM3 weights: {e}")
            return {}

    def _add_document_terms(self, doc_content: str, doc_weight: float, term_weights: Dict[str, float]):
        """Add document terms to the relevance model (Java: addTokens method logic)."""
        try:
            # Tokenize document content (Java: analyzer.tokenStream)
            doc_tokens = self._tokenize_document(doc_content)

            if not doc_tokens:
                return

            # Count term frequencies
            term_counts = Counter(doc_tokens)
            doc_length = len(doc_tokens)

            # Add weighted term frequencies (Java: wordFreqs.compute logic)
            for term, count in term_counts.items():
                term_prob = count / doc_length if doc_length > 0 else 0.0
                # Java: wordFreqs.compute(token, (t, oldV) -> (oldV==null)? weight : oldV + weight)
                term_weights[term] += doc_weight * term_prob

        except Exception as e:
            logger.debug(f"Error processing document: {e}")

    def _tokenize_document(self, doc_content: str) -> List[str]:
        """Tokenize document content using Lucene analyzer (Java: analyzer.tokenStream)."""
        try:
            tokens = []
            token_stream = self.analyzer.tokenStream("contents", self.StringReader(doc_content))
            char_term_attr = token_stream.addAttribute(self.CharTermAttribute)

            token_stream.reset()
            while token_stream.incrementToken() and len(tokens) < 1000:  # Limit for performance
                token = char_term_attr.toString()
                tokens.append(token)

            token_stream.end()
            token_stream.close()

            return tokens

        except Exception as e:
            logger.debug(f"Error tokenizing document: {e}")
            return []

    def explain_expansion(self, query: str, num_feedback_docs: int = 10) -> Dict[str, any]:
        """Return detailed information about the expansion process for debugging."""
        try:
            initial_query = self._build_boolean_query(query)
            top_docs = self.searcher.search(initial_query, num_feedback_docs)

            explanation = {
                'query': query,
                'query_terms': self._tokenize_query(query),
                'num_feedback_docs': len(top_docs.scoreDocs),
                'feedback_doc_scores': [score_doc.score for score_doc in top_docs.scoreDocs],
                'use_log_scores': any(score_doc.score < 0.0 for score_doc in top_docs.scoreDocs),
            }

            return explanation

        except Exception as e:
            logger.error(f"Error creating expansion explanation: {e}")
            return {'error': str(e)}

    def __del__(self):
        """Clean up Lucene resources."""
        try:
            if hasattr(self, 'reader'):
                self.reader.close()
        except Exception as e:
            logger.error(f"Error closing Lucene reader: {e}")


class RMExpansion:
    """
    Relevance Model (RM) query expansion using Lucene backend.

    Provides RM1 and RM3 expansion using the proven Java implementation
    via PyJnius integration.
    """

    def __init__(self, index_path: str, k1: float = 1.2, b: float = 0.75):
        """
        Initialize RM expansion with Lucene backend.

        Args:
            index_path: Path to Lucene index
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise ValueError(f"Index path does not exist: {index_path}")

        self.lucene_rm3 = LuceneRM3Scorer(str(self.index_path), k1, b)

        logger.info(f"RMExpansion initialized with Lucene backend: {index_path}")

    def expand_query(self,
                     query: str,
                     documents: List[str],  # Ignored - Lucene uses index
                     scores: List[float],   # Ignored - Lucene computes scores
                     num_expansion_terms: int = 10,
                     rm_type: str = "rm3") -> List[Tuple[str, float]]:
        """
        Expand query using Lucene RM1 or RM3 algorithm.

        Args:
            query: Original query string
            documents: Ignored (Lucene retrieves from index)
            scores: Ignored (Lucene computes scores)
            num_expansion_terms: Number of expansion terms to return
            rm_type: "rm1" (expansion only) or "rm3" (query + expansion)

        Returns:
            List of (term, weight) tuples, sorted by weight (descending)
        """
        if rm_type not in ["rm1", "rm3"]:
            raise ValueError(f"rm_type must be 'rm1' or 'rm3', got '{rm_type}'")

        logger.debug(f"Expanding query '{query}' using Lucene {rm_type.upper()}")

        omit_query_terms = (rm_type == "rm1")

        try:
            result = self.lucene_rm3.compute_rm3_expansion(
                query=query,
                num_feedback_docs=10,  # Standard number for TREC
                num_expansion_terms=num_expansion_terms,
                omit_query_terms=omit_query_terms
            )

            logger.debug(f"Lucene {rm_type.upper()} expansion completed: {len(result)} terms")
            return result

        except Exception as e:
            logger.error(f"Lucene {rm_type} expansion failed: {e}")
            return []

    def get_expansion_statistics(self,
                                 expansion_terms: List[Tuple[str, float]]) -> dict:
        """
        Compute statistics about expansion terms.

        Args:
            expansion_terms: List of (term, weight) tuples

        Returns:
            Dictionary with statistics
        """
        if not expansion_terms:
            return {}

        weights = [weight for _, weight in expansion_terms]

        import statistics
        stats = {
            'num_terms': len(expansion_terms),
            'mean_weight': statistics.mean(weights),
            'median_weight': statistics.median(weights),
            'min_weight': min(weights),
            'max_weight': max(weights)
        }

        if len(weights) > 1:
            stats['std_weight'] = statistics.stdev(weights)
        else:
            stats['std_weight'] = 0.0

        return stats

    def explain_expansion(self, query: str) -> dict:
        """Get detailed information about expansion process."""
        try:
            return self.lucene_rm3.explain_expansion(query)
        except Exception as e:
            logger.error(f"Error explaining expansion: {e}")
            return {'error': str(e)}


# Factory function
def create_rm_expansion(index_path: str, **kwargs) -> RMExpansion:
    """
    Create RM expansion with Lucene backend.

    Args:
        index_path: Path to Lucene index
        **kwargs: Additional arguments for Lucene scorer

    Returns:
        RMExpansion instance with Lucene backend
    """
    return RMExpansion(index_path, **kwargs)


# Convenience functions
def rm1_expansion(query: str,
                  index_path: str,
                  num_terms: int = 10,
                  **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for RM1 expansion using Lucene.

    Args:
        query: Query string
        index_path: Path to Lucene index
        num_terms: Number of expansion terms
        **kwargs: Additional arguments for RMExpansion

    Returns:
        List of (term, weight) tuples
    """
    rm = RMExpansion(index_path, **kwargs)
    return rm.expand_query(query, [], [], num_terms, rm_type="rm1")


def rm3_expansion(query: str,
                  index_path: str,
                  num_terms: int = 10,
                  **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for RM3 expansion using Lucene.

    Args:
        query: Query string
        index_path: Path to Lucene index
        num_terms: Number of expansion terms
        **kwargs: Additional arguments for RMExpansion

    Returns:
        List of (term, weight) tuples
    """
    rm = RMExpansion(index_path, **kwargs)
    return rm.expand_query(query, [], [], num_terms, rm_type="rm3")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Lucene RM Expansion Example")
    print("=" * 30)

    try:
        # Example with actual index
        index_path = "./your_index_path"  # Update this
        rm = RMExpansion(index_path)

        query = "machine learning algorithms"
        print(f"Query: {query}")

        # RM3 expansion
        rm3_terms = rm.expand_query(query, [], [], num_expansion_terms=10, rm_type="rm3")
        print(f"\nRM3 Expansion ({len(rm3_terms)} terms):")
        for i, (term, weight) in enumerate(rm3_terms, 1):
            print(f"  {i:2d}. {term:<15} {weight:.4f}")

        # RM1 expansion
        rm1_terms = rm.expand_query(query, [], [], num_expansion_terms=8, rm_type="rm1")
        print(f"\nRM1 Expansion ({len(rm1_terms)} terms):")
        for i, (term, weight) in enumerate(rm1_terms, 1):
            print(f"  {i:2d}. {term:<15} {weight:.4f}")

        # Statistics
        stats = rm.get_expansion_statistics(rm3_terms)
        print(f"\nExpansion Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure to update index_path to point to your Lucene index")