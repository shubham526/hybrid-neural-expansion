#!/usr/bin/env python3
"""
CEQE Reranking Script

Reranks test data using CEQE (Contextualized Embeddings for Query Expansion)
following the original implementation from the CEQE repository.
Outputs results in TREC run format.
"""

import json
import os
import argparse
import math
import numpy as np
import operator
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from scipy import spatial
from scipy.special import logsumexp, softmax
from typing import Dict, List, Any, Tuple
import sys
import krovetz
import nltk
from nltk.corpus import stopwords


def load_jsonl(file_path):
    """Load JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class PoolingStrategy:
    NONE = 0
    REDUCE_MEAN = 2


class CEQEModel:
    """CEQE model following the original implementation."""

    def __init__(self, model_name="bert-base-uncased", max_seq_len=128, device=None):
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BERT model and tokenizer (following original graph.py)
        config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=False)
        self.model = BertModel.from_pretrained(model_name, config=config)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Load stemmer for term processing
        self.stemmer = krovetz.PyKrovetzStemmer()

        # Load stopwords (following original)
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """Load NLTK English stopwords."""
        try:
            # Download stopwords if not already present
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))

        except Exception as e:
            print(f"Warning: Could not load NLTK stopwords ({e}), using basic stopwords")
            # Fallback to basic stopwords if NLTK fails
            return set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if'
            ])

    def convert_example_to_feature(self, text, text_id, chunk=True):
        """Convert text to features following original extract_features.py"""
        tokens_a = self.tokenizer.tokenize(text)
        tokens_b = None

        if not chunk:
            if len(tokens_a) > self.max_seq_len - 2:
                tokens_a = tokens_a[:(self.max_seq_len - 2)]
            return self._get_input_features(tokens_a, tokens_b, text_id)
        else:
            # Chunking for long documents (following original)
            chunks_input_features = []
            begin_pointer = 0
            while begin_pointer < len(tokens_a):
                end_pointer = begin_pointer + (self.max_seq_len - 2)
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

    def _get_input_features(self, tokens_a, tokens_b, text_id):
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
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'input_tokens': tokens,
            'id': text_id
        }

    def get_embedding_matrix(self, input_features, pooling_strategy=PoolingStrategy.NONE):
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
        masked_reduce_max = lambda x, m: torch.max(minus_mask(x, m), dim=1).values
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
        bow_q_text = set([self.stemmer.stem(q_t) for q_t in query_text.split()])

        most_sim = sorted(exp_terms_with_score.items(), key=operator.itemgetter(1), reverse=True)
        unique_exp_terms = []

        for t in most_sim:
            if len(unique_exp_terms) < num_exp_terms:
                # Filter following original logic: stopwords + BERT tokens + query terms
                if (t[0] not in self.stopwords and
                        t[0] not in ['[CLS]', '[SEP]'] and  # Handle BERT tokens explicitly as in original
                        len(t[0]) > 1 and  # Filter very short terms
                        self.stemmer.stem(t[0]) not in bow_q_text):
                    unique_exp_terms.append(t)
            else:
                break

        return unique_exp_terms

    def expand_and_rerank(self, query_text, candidates, method='centroid', num_expansion_terms=10, num_prf_docs=5):
        """
        Perform CEQE expansion and reranking following original implementation.

        Args:
            query_text: Original query text
            candidates: List of candidate documents with scores
            method: 'centroid', 'term_max', or 'term_mul'
            num_expansion_terms: Number of expansion terms
            num_prf_docs: Number of PRF documents

        Returns:
            Tuple of (expansion_terms, reranked_candidates)
        """
        # Select PRF documents
        prf_docs = candidates[:num_prf_docs]
        prf_texts = [doc['doc_text'] for doc in prf_docs if 'doc_text' in doc]

        if not prf_texts:
            return [], candidates

        # Create retrieval result for PRF docs
        retrieval_result = {}
        for doc in prf_docs:
            retrieval_result[doc['doc_id']] = doc['score']

        try:
            # Process query
            query_features = [self.convert_example_to_feature(query_text, 'query', chunk=False)]

            # Process documents
            doc_features = []
            for i, doc_text in enumerate(prf_texts):
                doc_id = prf_docs[i]['doc_id']
                features = self.convert_example_to_feature(doc_text, doc_id, chunk=True)
                if isinstance(features, list):
                    doc_features.extend(features)
                else:
                    doc_features.append(features)

            # Get embeddings
            batch_size = 32

            # Query embeddings - both pooled and token-level
            query_embedding_pooled = self.get_embedding_matrix(query_features, PoolingStrategy.REDUCE_MEAN)
            query_embedding_tokens = self.get_embedding_matrix(query_features, PoolingStrategy.NONE)

            query_embedding_pooled = query_embedding_pooled.detach().cpu().numpy()
            query_embedding_tokens = query_embedding_tokens.detach().cpu().numpy()

            # Remove zero tokens for token-level query
            query_embedding_tokens_clean = self.remove_zero_tokens(query_embedding_tokens)

            # Doc embeddings
            doc_embeddings = []
            for i in range(0, len(doc_features), batch_size):
                batch = doc_features[i:i + batch_size]
                batch_embeddings = self.get_embedding_matrix(batch, PoolingStrategy.NONE)
                doc_embeddings.append(batch_embeddings.detach().cpu().numpy())

            doc_embeddings = np.concatenate(doc_embeddings, axis=0)
            doc_embeddings_clean = self.remove_zero_tokens(doc_embeddings)

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
            docs_terms, docs_terms_embeds = self.get_terms(docs_info)
            docs_terms_embeds_pooled = self.get_terms_embeds_pooled(docs_terms, docs_terms_embeds)

            # Create PRF docs terms info
            prf_docs_terms_info = {
                'id': docs_info['id'],
                'terms': docs_terms,
                'embedding': docs_terms_embeds_pooled
            }

            if method == 'centroid':
                # Centroid approach
                similarity = self.get_similarity_query_and_docTerms_CAR(query_info_pooled, prf_docs_terms_info)
                exp_terms_score = self.query_vs_per_doc_context_rm(similarity, retrieval_result)

            else:
                # Term-based approach
                query_terms, query_terms_embeds = self.get_terms(query_info_tokens)
                query_terms_embeds_pooled = self.get_terms_embeds_pooled(query_terms, query_terms_embeds)

                query_info_terms = {
                    'id': 'query',
                    'terms': query_terms[0],
                    'embedding': query_terms_embeds_pooled[0]
                }

                similarity = self.get_similarity_queryTerms_and_docsTerms_CAR(query_info_terms, prf_docs_terms_info)
                pooling_method = 'max' if method == 'term_max' else 'mul'
                exp_terms_score = self.queryTerm_vs_per_doc_context_rm(similarity, retrieval_result, pooling_method)

            # Get expansion terms
            expansion_terms = self.get_unique_expansion_terms(exp_terms_score, query_text, num_expansion_terms)

            # Simple reranking: boost scores of documents containing expansion terms
            reranked_candidates = []
            expansion_words = set([term for term, score in expansion_terms])

            for candidate in candidates:
                new_candidate = candidate.copy()
                doc_text_lower = candidate.get('doc_text', '').lower()

                # Count expansion term matches
                term_matches = sum(1 for term in expansion_words if term.lower() in doc_text_lower)

                # Boost score based on expansion term matches
                boost = 0.1 * term_matches / max(1, len(expansion_words))
                new_candidate['score'] = candidate['score'] + boost
                reranked_candidates.append(new_candidate)

            # Sort by new scores
            reranked_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Update ranks
            for i, candidate in enumerate(reranked_candidates):
                candidate['rank'] = i + 1

            return expansion_terms, reranked_candidates

        except Exception as e:
            print(f"Error in CEQE expansion: {e}")
            return [], candidates


def process_single_query_ceqe(query_data, args, work_dir):
    """Process a single query with CEQE expansion and reranking."""
    query_id = query_data['query_id']
    query_text = query_data['query_text']
    candidates = query_data['candidates']

    query_work_dir = work_dir / f"query_{query_id}"
    query_work_dir.mkdir(exist_ok=True)

    try:
        # Initialize CEQE model
        ceqe_model = CEQEModel(
            model_name=args.model_name,
            max_seq_len=args.max_seq_len
        )

        # Perform CEQE expansion and reranking
        expansion_terms, reranked_candidates = ceqe_model.expand_and_rerank(
            query_text=query_text,
            candidates=candidates,
            method=args.method,
            num_expansion_terms=args.num_expansion_terms,
            num_prf_docs=args.prf_depth
        )

        # Convert to results format
        query_results = []
        for candidate in reranked_candidates:
            query_results.append({
                'query_id': query_id,
                'doc_id': candidate['doc_id'],
                'rank': candidate['rank'],
                'score': float(candidate['score'])
            })

        return {
            'query_id': query_id,
            'success': True,
            'results': query_results,
            'num_expansion_terms': len(expansion_terms),
            'num_results': len(query_results),
            'expansion_terms': expansion_terms
        }

    except Exception as e:
        return {
            'query_id': query_id,
            'success': False,
            'error': str(e),
            'results': []
        }

    finally:
        if query_work_dir.exists():
            shutil.rmtree(query_work_dir, ignore_errors=True)


def write_trec_run(all_results, output_file, run_name):
    """Write results in TREC run format."""
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(f"{result['query_id']} Q0 {result['doc_id']} {result['rank']} {result['score']} {run_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rerank test data using CEQE (Contextualized Embeddings for Query Expansion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test.jsonl file from create_train_test_data.py')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for TREC run file')

    # CEQE model arguments
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='BERT model name or path')
    parser.add_argument('--max-seq-len', type=int, default=128,
                        help='Maximum sequence length for BERT')

    # CEQE expansion arguments
    parser.add_argument('--method', type=str, default='centroid',
                        choices=['centroid', 'term_max', 'term_mul'],
                        help='CEQE method: centroid (query centroid), term_max (max pooling), term_mul (multiplicative pooling)')
    parser.add_argument('--prf-depth', type=int, default=5,
                        help='Number of PRF documents to use for expansion')
    parser.add_argument('--num-expansion-terms', type=int, default=10,
                        help='Number of expansion terms to use')

    # Optional arguments
    parser.add_argument('--run-name', type=str, default='ceqe',
                        help='Run name for TREC output')
    parser.add_argument('--max-workers', type=int, default=2,
                        help='Number of parallel workers (CEQE is compute intensive)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    print(f"Loaded {len(test_data)} queries")

    # Create working directory
    work_dir = output_dir / "temp_work"
    work_dir.mkdir(exist_ok=True)

    print(f"Starting CEQE reranking...")
    print(f"  Model: {args.model_name}")
    print(f"  Method: {args.method}")
    print(f"  PRF depth: {args.prf_depth}")
    print(f"  Expansion terms: {args.num_expansion_terms}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Run name: {args.run_name}")

    all_query_results = []
    successful_queries = 0
    failed_queries = 0

    try:
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_query = {
                executor.submit(process_single_query_ceqe, query_data, args, work_dir): query_data['query_id']
                for query_data in test_data
            }

            for future in as_completed(future_to_query):
                query_id = future_to_query[future]

                try:
                    result = future.result()
                    all_query_results.append(result)

                    if result['success']:
                        successful_queries += 1
                        print(f"✓ {query_id}: {result['num_results']} results, "
                              f"{result['num_expansion_terms']} expansion terms")

                        # Print some expansion terms for debugging
                        if result.get('expansion_terms'):
                            expansion_preview = result['expansion_terms'][:3]
                            terms_str = ", ".join([f"{term}({score:.3f})" for term, score in expansion_preview])
                            print(f"  Expansion preview: {terms_str}")
                    else:
                        failed_queries += 1
                        print(f"✗ {query_id}: {result['error']}")

                except Exception as e:
                    failed_queries += 1
                    print(f"✗ {query_id}: {e}")

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    # Collect all results and write TREC run file
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

        print(f"\n✅ CEQE reranking completed!")
        print(f"   Successful queries: {successful_queries}")
        print(f"   Failed queries: {failed_queries}")
        print(f"   Total results: {len(all_results)}")
        print(f"   TREC run file: {trec_file}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main()