"""
RM (Relevance Model) Expansion Module - Lucene Backend with Original Term Mapping

FIXED: Now maintains mapping between stemmed terms (for RM3) and original terms (for semantic similarity)
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
    Lucene-based RM3 implementation that maintains original term mappings.

    KEY FIX: Now tracks both stemmed terms (for RM3 weights) and original forms (for embeddings).
    """

    def __init__(self, index_path: str, k1: float = 1.2, b: float = 0.75):
        """Initialize Lucene RM3 scorer."""
        try:
            self.index_path = index_path

            # Get Lucene classes lazily
            classes = get_lucene_classes()
            for name, cls in classes.items():
                setattr(self, name, cls)

            # Open index and setup searcher
            directory = self.FSDirectory.open(self.Path.get(index_path))
            self.reader = self.DirectoryReader.open(directory)

            # Use reader.getContext() to get IndexReaderContext for IndexSearcher
            reader_context = self.reader.getContext()
            self.searcher = self.IndexSearcher(reader_context)
            self.searcher.setSimilarity(self.BM25Similarity(k1, b))

            # Setup analyzer (same as used for indexing)
            self.analyzer = self.EnglishAnalyzer()

            logger.info(f"LuceneRM3Scorer initialized with index: {index_path}")

        except Exception as e:
            logger.error(f"Error initializing LuceneRM3Scorer: {e}")
            raise

    def compute_rm3_expansion_with_originals(self,
                                           query: str,
                                           num_feedback_docs: int = 10,
                                           num_expansion_terms: int = 20,
                                           omit_query_terms: bool = False) -> Tuple[List[Tuple[str, float]], Dict[str, str]]:
        """
        Compute RM3 expansion returning both stemmed terms and original term mappings.

        Returns:
            Tuple of:
            - List of (stemmed_term, weight) tuples for RM3
            - Dict mapping stemmed_term -> best_original_term for embeddings
        """
        try:
            logger.debug(f"Computing RM3 for query: '{query}'")

            # Step 1: Build initial query and search
            initial_query = self._build_boolean_query(query)
            top_docs = self.searcher.search(initial_query, num_feedback_docs)

            if top_docs.totalHits.value() == 0:
                logger.warning(f"No documents found for query: {query}")
                return [], {}

            logger.debug(f"Found {top_docs.totalHits.value()} documents for feedback")

            # Step 2: Extract terms and compute RM3 weights WITH original term tracking
            term_weights, original_mappings = self._compute_rm3_weights_with_originals(
                query, top_docs.scoreDocs, omit_query_terms
            )

            # Step 3: Sort and return top terms
            sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
            result = sorted_terms[:num_expansion_terms]

            # Filter mappings to only include terms in final result
            result_terms = {term for term, _ in result}
            filtered_mappings = {k: v for k, v in original_mappings.items() if k in result_terms}

            logger.debug(f"RM3 expansion completed: {len(result)} terms with {len(filtered_mappings)} original mappings")
            return result, filtered_mappings

        except Exception as e:
            logger.error(f"Error in RM3 expansion: {e}")
            return [], {}

    def _build_boolean_query(self, query_str: str):
        """Build initial boolean query from query string."""
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
        """Tokenize query string using Lucene analyzer."""
        try:
            tokens = []
            # Fixed: Pass string directly to tokenStream, not StringReader
            token_stream = self.analyzer.tokenStream("contents", query_str)
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

    def _compute_rm3_weights_with_originals(self,
                                          query_str: str,
                                          score_docs,
                                          omit_query_terms: bool) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Compute RM3 weights AND track original terms.

        Returns:
            Tuple of:
            - Dict mapping stemmed_term -> weight
            - Dict mapping stemmed_term -> best_original_term
        """
        try:
            term_weights = defaultdict(float)
            # NEW: Track original forms of terms
            stemmed_to_original = {}  # stemmed_term -> original_term
            original_term_counts = defaultdict(lambda: defaultdict(int))  # stemmed -> {original: count}

            # Step 1: Add original query terms with weight 1.0
            if not omit_query_terms:
                query_tokens = self._tokenize_query(query_str)
                original_query_words = query_str.lower().split()

                # Map stemmed query terms to original words
                for i, stemmed_token in enumerate(query_tokens):
                    term_weights[stemmed_token] = 1.0
                    # Try to map to original word (best effort)
                    if i < len(original_query_words):
                        original_word = original_query_words[i]
                        stemmed_to_original[stemmed_token] = original_word
                        original_term_counts[stemmed_token][original_word] += 1

                logger.debug(f"Added {len(query_tokens)} query terms with weight 1.0")

            # Step 2: Determine if we have log scores
            use_log = any(score_doc.score < 0.0 for score_doc in score_docs)

            # Step 3: Compute score normalizer
            normalizer = 0.0
            for score_doc in score_docs:
                if use_log:
                    normalizer += np.exp(score_doc.score)
                else:
                    normalizer += score_doc.score

            if use_log:
                normalizer = np.log(normalizer) if normalizer > 0 else 1.0

            # Step 4: Process each pseudo-relevant document
            for score_doc in score_docs:
                # Compute document weight
                if use_log:
                    doc_weight = score_doc.score - normalizer if normalizer > 0 else 0.0
                else:
                    doc_weight = score_doc.score / normalizer if normalizer > 0 else 0.0

                # Get document content using new API
                stored_fields = self.searcher.storedFields()
                doc = stored_fields.document(score_doc.doc)
                doc_content = doc.get("contents")

                if doc_content:
                    # NEW: Process document with original term tracking
                    self._add_document_terms_with_originals(
                        doc_content, doc_weight, term_weights, original_term_counts
                    )

            # Step 5: Select best original term for each stemmed term
            for stemmed_term, original_counts in original_term_counts.items():
                if original_counts:
                    # Choose the most frequent original form
                    best_original = max(original_counts.items(), key=lambda x: x[1])[0]
                    stemmed_to_original[stemmed_term] = best_original

            logger.debug(f"Processed {len(score_docs)} documents, total terms: {len(term_weights)}")
            logger.debug(f"Original mappings: {len(stemmed_to_original)}")

            return dict(term_weights), stemmed_to_original

        except Exception as e:
            logger.error(f"Error computing RM3 weights with originals: {e}")
            return {}, {}

    def _add_document_terms_with_originals(self,
                                         doc_content: str,
                                         doc_weight: float,
                                         term_weights: Dict[str, float],
                                         original_term_counts: Dict[str, Dict[str, int]]):
        """Add document terms to relevance model AND track original forms."""
        try:
            # Get both stemmed tokens and original tokens
            stemmed_tokens = self._tokenize_document_analyzed(doc_content)
            original_tokens = self._tokenize_document_original(doc_content)

            if not stemmed_tokens:
                return

            # Count stemmed term frequencies
            stemmed_counts = Counter(stemmed_tokens)
            doc_length = len(stemmed_tokens)

            # Create mapping from position to original word
            pos_to_original = {}
            if original_tokens:
                # Simple alignment: assume 1-to-1 mapping where possible
                for i, original_token in enumerate(original_tokens[:len(stemmed_tokens)]):
                    pos_to_original[i] = original_token

            # Add weighted term frequencies
            for stemmed_term, count in stemmed_counts.items():
                term_prob = count / doc_length if doc_length > 0 else 0.0
                term_weights[stemmed_term] += doc_weight * term_prob

                # Track original forms
                for i, token in enumerate(stemmed_tokens):
                    if token == stemmed_term and i in pos_to_original:
                        original_form = pos_to_original[i]
                        original_term_counts[stemmed_term][original_form] += 1

        except Exception as e:
            logger.debug(f"Error processing document with originals: {e}")

    def _tokenize_document_analyzed(self, doc_content: str) -> List[str]:
        """Tokenize document content using Lucene analyzer (returns stemmed terms)."""
        try:
            tokens = []
            # Fixed: Pass string directly to tokenStream
            token_stream = self.analyzer.tokenStream("contents", doc_content)
            char_term_attr = token_stream.addAttribute(self.CharTermAttribute)

            token_stream.reset()
            while token_stream.incrementToken() and len(tokens) < 1000:  # Limit for performance
                token = char_term_attr.toString()
                tokens.append(token)

            token_stream.end()
            token_stream.close()

            return tokens

        except Exception as e:
            logger.debug(f"Error tokenizing document (analyzed): {e}")
            return []

    def _tokenize_document_original(self, doc_content: str) -> List[str]:
        """Tokenize document content without analysis (returns original word forms)."""
        try:
            # Simple whitespace tokenization to preserve original forms
            import re
            # Split on whitespace and punctuation, keep alphabetic words
            words = re.findall(r'\b[a-zA-Z]+\b', doc_content.lower())
            return words[:1000]  # Limit for performance

        except Exception as e:
            logger.debug(f"Error tokenizing document (original): {e}")
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
    RM query expansion with original term preservation for proper semantic similarity.
    """

    def __init__(self, index_path: str, k1: float = 1.2, b: float = 0.75):
        """Initialize RM expansion with Lucene backend."""
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise ValueError(f"Index path does not exist: {index_path}")

        self.lucene_rm3 = LuceneRM3Scorer(str(self.index_path), k1, b)

        logger.info(f"RMExpansion initialized with original term tracking: {index_path}")

    def expand_query_with_originals(self,
                                   query: str,
                                   documents: List[str],  # Ignored - Lucene uses index
                                   scores: List[float],   # Ignored - Lucene computes scores
                                   num_expansion_terms: int = 10,
                                   num_feedback_docs: int = 10,  # NEW: Allow configurable feedback docs
                                   rm_type: str = "rm3") -> Tuple[List[Tuple[str, float]], Dict[str, str]]:
        """
        Expand query using Lucene RM1 or RM3 algorithm.

        NEW: Returns both expansion terms and original term mappings.

        Args:
            query: Original query string
            documents: Ignored (Lucene retrieves from index)
            scores: Ignored (Lucene computes scores)
            num_expansion_terms: Number of expansion terms to return
            num_feedback_docs: Number of pseudo-relevant documents to use
            rm_type: "rm1" (expansion only) or "rm3" (query + expansion)

        Returns:
            Tuple of:
            - List of (stemmed_term, weight) tuples for RM3 weights
            - Dict mapping stemmed_term -> original_term for semantic similarity
        """
        if rm_type not in ["rm1", "rm3"]:
            raise ValueError(f"rm_type must be 'rm1' or 'rm3', got '{rm_type}'")

        logger.debug(f"Expanding query '{query}' using Lucene {rm_type.upper()} with {num_feedback_docs} feedback docs")

        omit_query_terms = (rm_type == "rm1")

        try:
            result, original_mappings = self.lucene_rm3.compute_rm3_expansion_with_originals(
                query=query,
                num_feedback_docs=num_feedback_docs,  # FIXED: Use configurable value
                num_expansion_terms=num_expansion_terms,
                omit_query_terms=omit_query_terms
            )

            logger.debug(f"Lucene {rm_type.upper()} expansion completed: {len(result)} terms, {len(original_mappings)} mappings")
            return result, original_mappings

        except Exception as e:
            logger.error(f"Lucene {rm_type} expansion failed: {e}")
            return [], {}

    def expand_query(self,
                     query: str,
                     documents: List[str],
                     scores: List[float],
                     num_expansion_terms: int = 10,
                     num_feedback_docs: int = 10,  # NEW: Allow configurable feedback docs
                     rm_type: str = "rm3") -> List[Tuple[str, float]]:
        """
        Legacy method for backwards compatibility.
        Returns only the expansion terms (stemmed).
        """
        result, _ = self.expand_query_with_originals(
            query, documents, scores, num_expansion_terms, num_feedback_docs, rm_type
        )
        return result

    def get_expansion_statistics(self,
                                 expansion_terms: List[Tuple[str, float]]) -> dict:
        """Compute statistics about expansion terms."""
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
    """Create RM expansion with original term tracking."""
    return RMExpansion(index_path, **kwargs)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("RM Expansion with Original Term Tracking")
    print("=" * 45)

    try:
        # Example with actual index
        index_path = "./your_index_path"  # Update this
        rm = RMExpansion(index_path)

        query = "machine learning algorithms"
        print(f"Query: {query}")

        # RM3 expansion with original mappings
        rm3_terms, original_mappings = rm.expand_query_with_originals(
            query, [], [], num_expansion_terms=10, rm_type="rm3"
        )

        print(f"\nRM3 Expansion ({len(rm3_terms)} terms):")
        for i, (term, weight) in enumerate(rm3_terms, 1):
            original = original_mappings.get(term, term)
            print(f"  {i:2d}. {term:<15} → {original:<15} {weight:.4f}")

        print(f"\nOriginal mappings for semantic similarity:")
        for stemmed, original in original_mappings.items():
            if stemmed != original:
                print(f"  {stemmed} → {original}")

    except Exception as e:
        print(f"Example failed: {e}")
        print("Make sure to update index_path to point to your Lucene index")