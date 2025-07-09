#!/usr/bin/env python3
"""
CEQE Reranking System

A modular system to rerank TREC run files using CEQE (Contextualized Embeddings for Query Expansion).
Uses ir_datasets for document/query handling, following the same structure as ANCE-PRF.
"""

import os
import json
import pickle
import logging
import argparse
import math
import numpy as np
import operator
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import torch
import ir_datasets
from transformers import BertTokenizer, BertModel, BertConfig
from scipy import spatial
from scipy.special import logsumexp
from tqdm import tqdm

# Import stemmer and NLTK for term processing
try:
    import krovetz
except ImportError:
    print("Warning: krovetz stemmer not found. Install with: pip install krovetz")
    krovetz = None

try:
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    print("Warning: NLTK not found. Install with: pip install nltk")
    nltk = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoolingStrategy:
    """Pooling strategies for BERT embeddings."""
    NONE = 0
    REDUCE_MEAN = 2


class DocumentProcessor:
    """Handles document processing and tokenization for CEQE."""

    def __init__(self, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def convert_example_to_feature(self, text: str, text_id: str, chunk: bool = True) -> Dict[str, Any]:
        """Convert text to features following original extract_features.py"""
        tokens_a = self.tokenizer.tokenize(text)
        tokens_b = None

        if not chunk:
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            return self._get_input_features(tokens_a, tokens_b, text_id)
        else:
            # Chunking for long documents
            chunks_input_features = []
            begin_pointer = 0
            while begin_pointer < len(tokens_a):
                end_pointer = begin_pointer + (self.max_seq_length - 2)
                if end_pointer < len(tokens_a):
                    # Check to see if part of a token will be in the next chunk
                    while end_pointer > begin_pointer and tokens_a[end_pointer].startswith("##"):
                        end_pointer -= 1
                chunked_tokens_a = tokens_a[begin_pointer:end_pointer]
                begin_pointer = end_pointer
                chunks_input_features.append(
                    self._get_input_features(chunked_tokens_a, tokens_b, text_id)
                )
            return chunks_input_features

    def _get_input_features(self, tokens_a: List[str], tokens_b: Optional[List[str]], text_id: str) -> Dict[str, Any]:
        """Get input features following original extract_features.py"""
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        if tokens_b:
            segment_ids = [0] * len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        else:
            segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'input_tokens': tokens,
            'id': text_id
        }


class TRECRunLoader:
    """Loads and processes TREC run files."""

    def __init__(self, run_path: str):
        self.run_path = run_path
        self.run_data = self._load_run()

    def _load_run(self) -> Dict[str, List[Tuple[str, float]]]:
        """Load TREC run file into query_id -> [(doc_id, score), ...] format."""
        run_data = defaultdict(list)

        with open(self.run_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, rank, score, _ = parts[:6]
                    run_data[qid].append((docid, float(score)))
                elif len(parts) >= 3:
                    qid, docid, rank = parts[:3]
                    run_data[qid].append((docid, float(rank)))

        # Sort by score descending for each query
        for qid in run_data:
            run_data[qid].sort(key=lambda x: x[1], reverse=True)

        return dict(run_data)

    def get_query_docs(self, query_id: str, top_k: int = None) -> List[str]:
        """Get document IDs for a query, optionally limited to top_k."""
        if query_id not in self.run_data:
            return []

        docs = [doc_id for doc_id, _ in self.run_data[query_id]]
        return docs[:top_k] if top_k else docs


class CEQEModel:
    """CEQE model following the original implementation."""

    def __init__(self, model_name: str = "bert-base-uncased", max_seq_len: int = 128, device: str = 'cuda'):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device

        # Load BERT model and tokenizer
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=False)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.doc_processor = DocumentProcessor(self.tokenizer, max_seq_len)

        # Load stemmer for term processing
        if krovetz:
            self.stemmer = krovetz.PyKrovetzStemmer()
        else:
            logger.warning("Krovetz stemmer not available, using basic stemming")
            self.stemmer = None

        # Load stopwords
        self.stopwords = self._load_stopwords()

        logger.info(f"CEQE model loaded with {model_name}")

    def _load_stopwords(self) -> set:
        """Load NLTK English stopwords."""
        try:
            if nltk:
                nltk.download('stopwords', quiet=True)
                return set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not load NLTK stopwords ({e}), using basic stopwords")

        # Fallback to basic stopwords
        return set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
        ])

    def stem_word(self, word: str) -> str:
        """Stem a word using available stemmer."""
        if self.stemmer:
            return self.stemmer.stem(word)
        else:
            # Basic stemming fallback
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
            elif word.endswith('s'):
                return word[:-1]
            return word

    def get_embedding_matrix(self, input_features: List[Dict], pooling_strategy: int = PoolingStrategy.NONE):
        """Get embeddings following original graph.py"""
        input_ids = torch.tensor([f['input_ids'] for f in input_features], dtype=torch.long)
        input_mask = torch.tensor([f['input_mask'] for f in input_features], dtype=torch.long)
        segment_ids = torch.tensor([f['segment_ids'] for f in input_features], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            all_encoder_layers = outputs.hidden_states[:-1]  # Exclude last layer

            # Use second to last layer (layer 11) as in original
            encoder_layer = all_encoder_layers[-2]  # Second to last layer

        minus_mask = lambda x, m: x - (1.0 - m.unsqueeze(-1)) * 1e5
        mul_mask = lambda x, m: x * m.unsqueeze(-1)
        masked_reduce_mean = lambda x, m: torch.sum(mul_mask(x, m), dim=1) / (
                torch.sum(m, dim=1, keepdim=True) + 1e-10)

        input_mask = input_mask.float()
        if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
            pooled = masked_reduce_mean(encoder_layer, input_mask)
        elif pooling_strategy == PoolingStrategy.NONE:
            pooled = mul_mask(encoder_layer, input_mask)
        else:
            raise NotImplementedError()

        return pooled

    def remove_zero_tokens(self, embedding_matrix):
        """Remove zero tokens following original implementation"""
        embedding_matrix_non_zero = []
        for i in range(embedding_matrix.shape[0]):
            embedding_matrix_i = embedding_matrix[i]
            embedding_matrix_i_non_zero = []
            for k in embedding_matrix_i:
                if np.array_equal(k, np.zeros(embedding_matrix_i.shape[1])):
                    break
                embedding_matrix_i_non_zero.append(k)
            embedding_matrix_non_zero.append(np.array(embedding_matrix_i_non_zero))
        return embedding_matrix_non_zero

    def get_terms(self, query_info):
        """Group tokens to terms following original tokens_to_terms.py"""
        queries_terms = [[] for i in range(len(query_info['id']))]
        queries_terms_embeds = [[] for i in range(len(query_info['id']))]

        for i in range(len(query_info['id'])):
            q_tokens = query_info['tokens'][i]
            q_embeds = query_info['embedding'][i]

            for j in range(len(q_tokens)):
                token = q_tokens[j]
                token_embeds = np.array(q_embeds[j]).reshape(-1, len(q_embeds[j]))

                if token.startswith('##'):
                    if queries_terms[i]:  # Check if there are previous tokens
                        prev_token = queries_terms[i].pop()
                        if prev_token == "[CLS]":
                            queries_terms[i].append("[CLS]")
                            queries_terms[i].append(token)
                            queries_terms_embeds[i].append(token_embeds)
                            continue
                        new_token = prev_token + token[2:]  # Remove '##'
                        queries_terms[i].append(new_token)
                        prev_embeds = queries_terms_embeds[i].pop()
                        term_embeds = np.append(prev_embeds, token_embeds, axis=0)
                        queries_terms_embeds[i].append(term_embeds)
                    else:
                        # First token is ##, just add it
                        queries_terms[i].append(token)
                        queries_terms_embeds[i].append(token_embeds)
                else:
                    queries_terms[i].append(token)
                    queries_terms_embeds[i].append(token_embeds)

        return queries_terms, queries_terms_embeds

    def get_terms_embeds_pooled(self, queries_terms, queries_terms_embeds):
        """Pool token embeddings to term embeddings following original"""
        queries_terms_embeds_pooled = [[] for i in range(len(queries_terms_embeds))]

        for i in range(len(queries_terms)):
            q_terms_embeds = queries_terms_embeds[i]
            for j in range(len(q_terms_embeds)):
                queries_terms_embeds_pooled[i].append(np.mean(q_terms_embeds[j], axis=0))

        return np.array(queries_terms_embeds_pooled)

    def get_similarity_query_and_docTerms_CAR(self, query_info, prf_docs_terms_info):
        """Get similarity following original get_similarity.py"""
        similarity_q = {}
        q_id = query_info['id']
        query_vec = np.array(query_info['embedding'])

        for doc_j in range(len(prf_docs_terms_info['id'])):
            doc_id = prf_docs_terms_info['id'][doc_j]
            doc_terms_vec = np.array(prf_docs_terms_info['embedding'][doc_j])

            for d_t in range(len(doc_terms_vec)):
                doc_term = prf_docs_terms_info['terms'][doc_j][d_t]
                doc_term_vec = doc_terms_vec[d_t]
                sim = 1 - spatial.distance.cosine(query_vec, doc_term_vec)
                similarity_q.setdefault(doc_id, []).append((doc_term, sim))

        return similarity_q

    def get_similarity_queryTerms_and_docsTerms_CAR(self, query_info, prf_docs_terms_info):
        """Get similarity for query terms following original get_similarity.py"""
        similarity_q = {}
        q_id = query_info['id']
        query_terms_vec = np.array(query_info['embedding'])

        for q_t in range(len(query_terms_vec)):
            term_vec = query_terms_vec[q_t]
            q_t_mention = query_info['terms'][q_t]

            for j in range(len(prf_docs_terms_info['id'])):
                doc_id = prf_docs_terms_info['id'][j]
                doc_terms_vec = np.array(prf_docs_terms_info['embedding'][j])

                for d_t in range(len(doc_terms_vec)):
                    term = prf_docs_terms_info['terms'][j][d_t]
                    doc_term_vec = doc_terms_vec[d_t]
                    sim = 1 - spatial.distance.cosine(term_vec, doc_term_vec)
                    similarity_q.setdefault(q_t_mention, {})
                    similarity_q[q_t_mention].setdefault(doc_id, []).append((term, sim))

        return similarity_q

    def query_vs_per_doc_context_rm(self, similarity_query_to_docs_terms, retrieval_result_query):
        """Following original query_vs_per_doc_context_rm.py"""

        def get_terms(doc_id):
            terms = set()
            for mention in similarity_query_to_docs_terms[doc_id]:
                terms.add(mention[0])
            return list(terms)

        def get_normalizer(doc_id):
            normalizer = 0
            for m in similarity_query_to_docs_terms[doc_id]:
                normalizer += m[1]
            return normalizer

        def get_sum_sim_score_termMentions_to_query(doc_id, term):
            sum_sim_score = 0
            for mention in similarity_query_to_docs_terms[doc_id]:
                if mention[0] == term:
                    sum_sim_score += mention[1]
            return sum_sim_score

        docs_terms_scores = {}
        for doc_id in similarity_query_to_docs_terms:
            doc_terms = get_terms(doc_id)
            normalizer_query = get_normalizer(doc_id)

            for t in doc_terms:
                score = get_sum_sim_score_termMentions_to_query(doc_id, t)
                normalized_score = float(score) / float(normalizer_query) if normalizer_query > 0 else 0
                docs_terms_scores.setdefault(doc_id, {})
                docs_terms_scores[doc_id][t] = normalized_score

        # Convert retrieval scores to posteriors
        doc_scores = []
        for doc_id in retrieval_result_query:
            doc_scores.append(retrieval_result_query[doc_id])

        log_sum_exp = logsumexp(doc_scores)
        posteriors = {}
        for doc_id in retrieval_result_query:
            log_posterior = retrieval_result_query[doc_id] - log_sum_exp
            posteriors[doc_id] = math.exp(log_posterior)

        # Combine term scores with document posteriors
        exp_terms_score_mul_doc_prob = {}
        for doc_id in docs_terms_scores:
            doc_prob = posteriors.get(doc_id, 0)

            for t in docs_terms_scores[doc_id]:
                exp_terms_score_mul_doc_prob.setdefault(t, 0)
                exp_terms_score_mul_doc_prob[t] += (docs_terms_scores[doc_id][t] * doc_prob)

        return exp_terms_score_mul_doc_prob

    def queryTerm_vs_per_doc_context_rm(self, similarity_queryTerms_to_docs_terms, retrieval_result_query,
                                        pooling_method='max'):
        """Following original queryTerm_vs_per_doc_context_rm.py"""
        import sys

        def get_terms(doc_id):
            terms = set()
            sample_query_term = list(similarity_queryTerms_to_docs_terms.keys())[0]
            for mention in similarity_queryTerms_to_docs_terms[sample_query_term][doc_id]:
                terms.add(mention[0])
            return list(terms)

        def get_normalizer(q_t, doc_id):
            normalizer = 0
            for m in similarity_queryTerms_to_docs_terms[q_t][doc_id]:
                normalizer += m[1]
            return normalizer

        def get_sum_sim_score_termMentions_to_queryTerm(q_t, doc_id, term):
            sum_sim_score = 0
            for mention in similarity_queryTerms_to_docs_terms[q_t][doc_id]:
                if mention[0] == term:
                    sum_sim_score += mention[1]
            return sum_sim_score

        docs_terms_scores = {}
        for q_term in similarity_queryTerms_to_docs_terms:
            for doc_id in similarity_queryTerms_to_docs_terms[q_term]:
                doc_terms = get_terms(doc_id)
                normalizer_q_term = get_normalizer(q_term, doc_id)

                for t in doc_terms:
                    score = get_sum_sim_score_termMentions_to_queryTerm(q_term, doc_id, t)
                    normalized_score = float(score) / float(normalizer_q_term) if normalizer_q_term > 0 else 0
                    docs_terms_scores.setdefault(doc_id, {})
                    docs_terms_scores[doc_id].setdefault(t, {})
                    docs_terms_scores[doc_id][t][q_term] = normalized_score

        # Apply pooling strategy
        final_scores = {}
        for doc_id in docs_terms_scores:
            final_scores.setdefault(doc_id, {})
            for t in docs_terms_scores[doc_id]:
                if pooling_method == 'max':
                    final_score_t = -sys.maxsize
                    for q_term_i in docs_terms_scores[doc_id][t]:
                        if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                            if docs_terms_scores[doc_id][t][q_term_i] > final_score_t:
                                final_score_t = docs_terms_scores[doc_id][t][q_term_i]
                elif pooling_method == 'mul':
                    final_score_t = 1
                    for q_term_i in docs_terms_scores[doc_id][t]:
                        if q_term_i != '[SEP]' and q_term_i != '[CLS]':
                            final_score_t *= docs_terms_scores[doc_id][t][q_term_i]

                final_scores[doc_id][t] = final_score_t if final_score_t != -sys.maxsize else 0

        # Convert to posteriors and aggregate
        doc_scores = []
        for doc_id in retrieval_result_query:
            doc_scores.append(retrieval_result_query[doc_id])

        log_sum_exp = logsumexp(doc_scores)
        posteriors = {}
        for doc_id in retrieval_result_query:
            log_posterior = retrieval_result_query[doc_id] - log_sum_exp
            posteriors[doc_id] = math.exp(log_posterior)

        exp_terms_score_mul_doc_prob = {}
        for doc_id in final_scores:
            doc_prob = posteriors.get(doc_id, 0)

            for t in final_scores[doc_id]:
                exp_terms_score_mul_doc_prob.setdefault(t, 0)
                exp_terms_score_mul_doc_prob[t] += (final_scores[doc_id][t] * doc_prob)

        return exp_terms_score_mul_doc_prob

    def get_unique_expansion_terms(self, exp_terms_with_score, query_text, num_exp_terms=10):
        """Get unique expansion terms following original implementation"""
        bow_q_text = set([self.stem_word(q_t) for q_t in query_text.split()])

        most_sim = sorted(exp_terms_with_score.items(), key=operator.itemgetter(1), reverse=True)
        unique_exp_terms = []

        for t in most_sim:
            if len(unique_exp_terms) < num_exp_terms:
                # Filter following original logic: stopwords + BERT tokens + query terms
                if (t[0] not in self.stopwords and
                        t[0] not in ['[CLS]', '[SEP]'] and  # Handle BERT tokens explicitly as in original
                        len(t[0]) > 1 and  # Filter very short terms
                        self.stem_word(t[0]) not in bow_q_text):
                    unique_exp_terms.append(t)
            else:
                break

        return unique_exp_terms


class IRDatasetHandler:
    """Handles ir_datasets integration for any dataset (reused from ANCE-PRF)."""

    def __init__(self, dataset_name: str, cache_dir: Optional[str] = None, lazy_loading: bool = False):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.lazy_loading = lazy_loading

        try:
            self.dataset = ir_datasets.load(dataset_name)
            logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            logger.info("Available datasets can be found at: https://ir-datasets.com/")
            raise

        # Initialize caches
        self.doc_cache = {}
        self.query_cache = {}

        if not lazy_loading:
            self._build_caches()
        else:
            logger.info("Lazy loading enabled - documents and queries will be loaded on demand")

    def _build_caches(self):
        """Build both document and query caches."""
        self._build_doc_cache()
        self._build_query_cache()

    def _build_doc_cache(self):
        """Build document ID to text cache with flexible field handling."""
        logger.info("Building document cache...")

        if not hasattr(self.dataset, 'docs_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have documents")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_docs.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.doc_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.doc_cache)} documents from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load doc cache: {e}")

        # Build cache from dataset
        doc_count = 0
        for doc in tqdm(self.dataset.docs_iter(), desc="Loading documents", total=self.dataset.docs_count()):
            doc_text = self._extract_document_text(doc)
            if doc_text.strip():  # Only cache non-empty documents
                self.doc_cache[doc.doc_id] = doc_text
                doc_count += 1

        logger.info(f"Loaded {doc_count} documents")

        # Save to cache if directory provided
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.doc_cache, f)
                logger.info(f"Saved document cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save doc cache: {e}")

    def _build_query_cache(self):
        """Build query ID to text cache with flexible field handling."""
        logger.info("Building query cache...")

        if not hasattr(self.dataset, 'queries_iter'):
            logger.warning(f"Dataset {self.dataset_name} does not have queries")
            return

        # Load from cache if available
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{self.dataset_name.replace('/', '_')}_queries.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        self.query_cache = pickle.load(f)
                    logger.info(f"Loaded {len(self.query_cache)} queries from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load query cache: {e}")

        # Build cache from dataset
        query_count = 0
        for query in self.dataset.queries_iter():
            query_text = self._extract_query_text(query)
            if query_text.strip():  # Only cache non-empty queries
                self.query_cache[query.query_id] = query_text
                query_count += 1

        logger.info(f"Loaded {query_count} queries")

        # Save to cache if directory provided
        if cache_file:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.query_cache, f)
                logger.info(f"Saved query cache to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save query cache: {e}")

    def _extract_document_text(self, doc) -> str:
        """Extract text from document with flexible field handling."""
        text_parts = []

        # Common document fields to check (in order of preference)
        text_fields = ['title', 'text', 'body', 'content', 'abstract', 'summary']

        for field in text_fields:
            if hasattr(doc, field):
                field_value = getattr(doc, field)
                if field_value and str(field_value).strip():
                    text_parts.append(str(field_value).strip())

        # If no standard fields found, try to get any string attributes
        if not text_parts:
            for attr_name in dir(doc):
                if not attr_name.startswith('_') and attr_name not in ['doc_id']:
                    try:
                        attr_value = getattr(doc, attr_name)
                        if isinstance(attr_value, str) and attr_value.strip():
                            text_parts.append(attr_value.strip())
                    except:
                        continue

        return " ".join(text_parts) if text_parts else ""

    def _extract_query_text(self, query) -> str:
        """Extract text from query with flexible field handling."""
        # Common query fields to check
        text_fields = ['text', 'title', 'query', 'description', 'narrative']

        for field in text_fields:
            if hasattr(query, field):
                field_value = getattr(query, field)
                if field_value and str(field_value).strip():
                    return str(field_value).strip()

        # If no standard fields found, try to get any string attributes
        for attr_name in dir(query):
            if not attr_name.startswith('_') and attr_name not in ['query_id']:
                try:
                    attr_value = getattr(query, attr_name)
                    if isinstance(attr_value, str) and attr_value.strip():
                        return attr_value.strip()
                except:
                    continue

        return ""

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get document text by ID with lazy loading support."""
        if doc_id in self.doc_cache:
            return self.doc_cache[doc_id]

        if self.lazy_loading:
            # Try to find document on demand
            try:
                for doc in self.dataset.docs_iter():
                    if doc.doc_id == doc_id:
                        doc_text = self._extract_document_text(doc)
                        self.doc_cache[doc_id] = doc_text  # Cache for future use
                        return doc_text
            except Exception as e:
                logger.warning(f"Error during lazy loading of document {doc_id}: {e}")

        return None

    def get_query_text(self, query_id: str) -> Optional[str]:
        """Get query text by ID with lazy loading support."""
        if query_id in self.query_cache:
            return self.query_cache[query_id]

        if self.lazy_loading:
            # Try to find query on demand
            try:
                for query in self.dataset.queries_iter():
                    if query.query_id == query_id:
                        query_text = self._extract_query_text(query)
                        self.query_cache[query_id] = query_text  # Cache for future use
                        return query_text
            except Exception as e:
                logger.warning(f"Error during lazy loading of query {query_id}: {e}")

        return None

    def get_documents_text(self, doc_ids: List[str]) -> List[str]:
        """Get multiple document texts with batch lazy loading."""
        doc_texts = []
        missing_ids = []

        # First, get cached documents
        for doc_id in doc_ids:
            if doc_id in self.doc_cache:
                doc_texts.append(self.doc_cache[doc_id])
            else:
                doc_texts.append("")  # Placeholder
                missing_ids.append((len(doc_texts) - 1, doc_id))

        # If lazy loading and we have missing documents, try to find them
        if self.lazy_loading and missing_ids:
            missing_id_set = {doc_id for _, doc_id in missing_ids}

            try:
                for doc in self.dataset.docs_iter():
                    if doc.doc_id in missing_id_set:
                        doc_text = self._extract_document_text(doc)
                        self.doc_cache[doc.doc_id] = doc_text

                        # Update the corresponding positions in doc_texts
                        for idx, doc_id in missing_ids:
                            if doc_id == doc.doc_id:
                                doc_texts[idx] = doc_text

                        missing_id_set.remove(doc.doc_id)
                        if not missing_id_set:
                            break
            except Exception as e:
                logger.warning(f"Error during batch lazy loading: {e}")

        return doc_texts

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        info = {
            'name': self.dataset_name,
            'has_docs': hasattr(self.dataset, 'docs_iter'),
            'has_queries': hasattr(self.dataset, 'queries_iter'),
            'has_qrels': hasattr(self.dataset, 'qrels_iter'),
            'cached_docs': len(self.doc_cache),
            'cached_queries': len(self.query_cache)
        }

        # Try to get additional dataset metadata
        if hasattr(self.dataset, 'documentation'):
            info['documentation'] = self.dataset.documentation()

        return info


class CEQEReranker:
    """Main CEQE reranking system that combines all components."""

    def __init__(self,
                 dataset_name: str,
                 model_name: str = "bert-base-uncased",
                 max_seq_len: int = 128,
                 device: str = 'cuda',
                 num_feedback_docs: int = 5,
                 num_expansion_terms: int = 10,
                 cache_dir: Optional[str] = None,
                 lazy_loading: bool = False):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms

        # Initialize components
        self.model = CEQEModel(model_name, max_seq_len, device)
        self.dataset_handler = IRDatasetHandler(dataset_name, cache_dir, lazy_loading)

        # Print dataset info
        info = self.dataset_handler.get_dataset_info()
        logger.info(f"Dataset info: {info}")

        if not info['has_docs'] or not info['has_queries']:
            logger.warning("Dataset may be missing documents or queries")

    def expand_query_with_ceqe(self, query_text: str, prf_doc_texts: List[str], prf_doc_ids: List[str],
                               retrieval_scores: Dict[str, float], method: str = 'centroid') -> List[Tuple[str, float]]:
        """
        Perform CEQE expansion for a single query.

        Args:
            query_text: Original query text
            prf_doc_texts: PRF document texts
            prf_doc_ids: PRF document IDs
            retrieval_scores: Original retrieval scores for PRF docs
            method: CEQE method ('centroid', 'term_max', 'term_mul')

        Returns:
            List of (term, score) expansion terms
        """
        if not prf_doc_texts:
            return []

        try:
            # Process query
            query_features = [self.model.doc_processor.convert_example_to_feature(query_text, 'query', chunk=False)]

            # Process documents
            doc_features = []
            for i, doc_text in enumerate(prf_doc_texts):
                doc_id = prf_doc_ids[i]
                features = self.model.doc_processor.convert_example_to_feature(doc_text, doc_id, chunk=True)
                if isinstance(features, list):
                    doc_features.extend(features)
                else:
                    doc_features.append(features)

            # Get embeddings
            batch_size = 32

            # Query embeddings - both pooled and token-level
            query_embedding_pooled = self.model.get_embedding_matrix(query_features, PoolingStrategy.REDUCE_MEAN)
            query_embedding_tokens = self.model.get_embedding_matrix(query_features, PoolingStrategy.NONE)

            query_embedding_pooled = query_embedding_pooled.detach().cpu().numpy()
            query_embedding_tokens = query_embedding_tokens.detach().cpu().numpy()

            # Remove zero tokens for token-level query
            query_embedding_tokens_clean = self.model.remove_zero_tokens(query_embedding_tokens)

            # Doc embeddings
            doc_embeddings = []
            for i in range(0, len(doc_features), batch_size):
                batch = doc_features[i:i + batch_size]
                batch_embeddings = self.model.get_embedding_matrix(batch, PoolingStrategy.NONE)
                doc_embeddings.append(batch_embeddings.detach().cpu().numpy())

            doc_embeddings = np.concatenate(doc_embeddings, axis=0)
            doc_embeddings_clean = self.model.remove_zero_tokens(doc_embeddings)

            # Convert to info structures
            query_info_pooled = {
                'id': 'query',
                'embedding': query_embedding_pooled[0]
            }

            query_info_tokens = {
                'id': ['query'],
                'tokens': [query_features[0]['input_tokens']],
                'embedding': query_embedding_tokens_clean
            }

            docs_info = {
                'id': [f['id'] for f in doc_features],
                'tokens': [f['input_tokens'] for f in doc_features],
                'embedding': doc_embeddings_clean
            }

            # Group tokens to terms for documents
            docs_terms, docs_terms_embeds = self.model.get_terms(docs_info)
            docs_terms_embeds_pooled = self.model.get_terms_embeds_pooled(docs_terms, docs_terms_embeds)

            # Create PRF docs terms info
            prf_docs_terms_info = {
                'id': docs_info['id'],
                'terms': docs_terms,
                'embedding': docs_terms_embeds_pooled
            }

            if method == 'centroid':
                # Centroid approach
                similarity = self.model.get_similarity_query_and_docTerms_CAR(query_info_pooled, prf_docs_terms_info)
                exp_terms_score = self.model.query_vs_per_doc_context_rm(similarity, retrieval_scores)

            else:
                # Term-based approach
                query_terms, query_terms_embeds = self.model.get_terms(query_info_tokens)
                query_terms_embeds_pooled = self.model.get_terms_embeds_pooled(query_terms, query_terms_embeds)

                query_info_terms = {
                    'id': 'query',
                    'terms': query_terms[0],
                    'embedding': query_terms_embeds_pooled[0]
                }

                similarity = self.model.get_similarity_queryTerms_and_docsTerms_CAR(query_info_terms,
                                                                                    prf_docs_terms_info)
                pooling_method = 'max' if method == 'term_max' else 'mul'
                exp_terms_score = self.model.queryTerm_vs_per_doc_context_rm(similarity, retrieval_scores,
                                                                             pooling_method)

            # Get expansion terms
            expansion_terms = self.model.get_unique_expansion_terms(exp_terms_score, query_text,
                                                                    self.num_expansion_terms)

            return expansion_terms

        except Exception as e:
            logger.warning(f"Error in CEQE expansion: {e}")
            return []

    def rerank_run(self,
                   run_path: str,
                   output_path: str,
                   method: str = 'centroid',
                   rerank_depth: int = 100,
                   expansion_weight: float = 0.1) -> None:
        """Rerank a TREC run file using CEQE and save results."""

        # Load run file
        run_loader = TRECRunLoader(run_path)
        logger.info(f"Loaded run file with {len(run_loader.run_data)} queries")

        # Collect all unique documents to cache them
        all_doc_ids = set()
        query_doc_mapping = {}

        for qid in run_loader.run_data.keys():
            doc_ids = run_loader.get_query_docs(qid, rerank_depth)
            query_doc_mapping[qid] = doc_ids
            all_doc_ids.update(doc_ids)

        logger.info(f"Found {len(all_doc_ids)} unique documents to process")

        # Pre-load all documents
        all_doc_ids_list = list(all_doc_ids)
        all_doc_texts = self.dataset_handler.get_documents_text(all_doc_ids_list)

        # Create mapping from doc_id to text
        doc_id_to_text = {}
        for doc_id, doc_text in zip(all_doc_ids_list, all_doc_texts):
            if doc_text.strip():
                doc_id_to_text[doc_id] = doc_text

        logger.info(f"Loaded {len(doc_id_to_text)} valid documents")

        # Process each query
        reranked_results = {}

        for qid in tqdm(run_loader.run_data.keys(), desc="Processing queries"):
            query_text = self.dataset_handler.get_query_text(qid)
            if not query_text:
                logger.warning(f"Query {qid} not found in dataset")
                continue

            # Get documents for this query with original scores
            query_docs = run_loader.run_data[qid][:rerank_depth]

            # Filter to only valid documents
            valid_docs = []
            for doc_id, score in query_docs:
                if doc_id in doc_id_to_text:
                    valid_docs.append((doc_id, score))

            if not valid_docs:
                logger.warning(f"No valid documents found for query {qid}")
                continue

            # Get PRF documents and their texts
            prf_docs = valid_docs[:self.num_feedback_docs]
            prf_doc_ids = [doc_id for doc_id, _ in prf_docs]
            prf_doc_texts = [doc_id_to_text[doc_id] for doc_id in prf_doc_ids]
            prf_scores = {doc_id: score for doc_id, score in prf_docs}

            # Perform CEQE expansion
            expansion_terms = self.expand_query_with_ceqe(
                query_text, prf_doc_texts, prf_doc_ids, prf_scores, method
            )

            # Rerank documents based on expansion terms
            expansion_words = set([term for term, _ in expansion_terms])

            reranked_docs = []
            for doc_id, orig_score in valid_docs:
                doc_text = doc_id_to_text[doc_id].lower()

                # Count expansion term matches
                term_matches = sum(1 for term in expansion_words if term.lower() in doc_text)

                # Boost score based on expansion term matches
                boost = expansion_weight * term_matches / max(1, len(expansion_words))
                new_score = orig_score + boost

                reranked_docs.append((doc_id, new_score))

            # Sort by new scores
            reranked_docs.sort(key=lambda x: x[1], reverse=True)

            # Store results
            reranked_results[qid] = reranked_docs

            if expansion_terms:
                logger.debug(f"Query {qid}: {len(expansion_terms)} expansion terms")

        # Save reranked results
        self._save_results(reranked_results, output_path)
        logger.info(f"Reranked results saved to {output_path}")

    def _save_results(self, results: Dict[str, List[Tuple[str, float]]], output_path: str):
        """Save results in TREC format."""
        with open(output_path, 'w') as f:
            for qid, doc_scores in results.items():
                for rank, (doc_id, score) in enumerate(doc_scores, 1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} CEQE\n")


def main():
    parser = argparse.ArgumentParser(description="CEQE Reranking System")
    parser.add_argument("--dataset", required=True, help="IR dataset name (e.g., 'msmarco-passage/dev')")
    parser.add_argument("--run", required=True, help="Path to TREC run file")
    parser.add_argument("--output", required=True, help="Path to output reranked file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--model-name", default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length for BERT")
    parser.add_argument("--method", choices=["centroid", "term_max", "term_mul"], default="centroid",
                        help="CEQE method: centroid (query centroid), term_max (max pooling), term_mul (multiplicative pooling)")
    parser.add_argument("--num-feedback", type=int, default=5, help="Number of PRF documents")
    parser.add_argument("--num-expansion-terms", type=int, default=10, help="Number of expansion terms")
    parser.add_argument("--rerank-depth", type=int, default=100, help="Number of top docs to rerank")
    parser.add_argument("--expansion-weight", type=float, default=0.1, help="Weight for expansion term boost")
    parser.add_argument("--cache-dir", help="Directory to cache dataset files")
    parser.add_argument("--lazy-loading", action="store_true", help="Enable lazy loading for large datasets")
    parser.add_argument("--list-datasets", action="store_true", help="List available ir_datasets and exit")

    args = parser.parse_args()

    if args.list_datasets:
        print("Popular ir_datasets datasets:")
        print("- msmarco-passage/dev, msmarco-passage/eval")
        print("- msmarco-document/dev, msmarco-document/eval")
        print("- trec-dl-2019/passage, trec-dl-2020/passage")
        print("- trec-covid, antique, nfcorpus")
        print("- clueweb09, clueweb12")
        print("- See https://ir-datasets.com/ for complete list")
        return

    # Check for required dependencies
    missing_deps = []
    if krovetz is None:
        missing_deps.append("krovetz (pip install krovetz)")
    if nltk is None:
        missing_deps.append("nltk (pip install nltk)")

    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.warning("Some functionality may be limited, but basic operation should work")

    # Initialize reranker
    reranker = CEQEReranker(
        dataset_name=args.dataset,
        model_name=args.model_name,
        max_seq_len=args.max_seq_len,
        device=args.device,
        num_feedback_docs=args.num_feedback,
        num_expansion_terms=args.num_expansion_terms,
        cache_dir=args.cache_dir,
        lazy_loading=args.lazy_loading
    )

    # Perform reranking
    reranker.rerank_run(
        run_path=args.run,
        output_path=args.output,
        method=args.method,
        rerank_depth=args.rerank_depth,
        expansion_weight=args.expansion_weight
    )

    print(f"CEQE reranking completed. Results saved to {args.output}")


if __name__ == "__main__":
    main()