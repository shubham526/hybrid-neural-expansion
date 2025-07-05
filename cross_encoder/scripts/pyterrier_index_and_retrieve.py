#!/usr/bin/env python3
"""
PyTerrier Index and Retrieve Script

This script can index documents and perform retrieval using BM25 on BEIR datasets.
Supports beir/dbpedia-entity and beir/nq datasets.

Usage:
    # Index a dataset
    python pyterrier_script.py --mode index --dataset beir/dbpedia-entity --index_path ./dbpedia-index

    # Retrieve from indexed dataset
    python pyterrier_script.py --mode retrieve --dataset beir/dbpedia-entity --index_path ./dbpedia-index --output_file run.txt

    # Index and then retrieve
    python pyterrier_script.py --mode both --dataset beir/nq --index_path ./nq-index --output_file run.txt
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import pyterrier as pt
    import pandas as pd
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install python-terrier pandas")
    sys.exit(1)


class PyTerrierProcessor:
    def __init__(self, dataset_name, index_path):
        """
        Initialize PyTerrier processor

        Args:
            dataset_name: Name of the dataset (e.g., 'beir/dbpedia-entity', 'beir/nq')
            index_path: Path where the index will be stored/loaded from
        """
        self.dataset_name = dataset_name
        self.index_path = index_path
        self.dataset = None
        self.index = None
        self.retriever = None

        # Initialize PyTerrier
        if not pt.started():
            pt.init()
            print("PyTerrier initialized")

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load the specified dataset"""
        try:
            print(f"Loading dataset: {self.dataset_name}")
            self.dataset = pt.datasets.get_dataset(f'irds:{self.dataset_name}')
            print(f"Dataset loaded successfully")
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")
            print("Available datasets include: beir/dbpedia-entity, beir/nq, beir/hotpotqa, etc.")
            sys.exit(1)

    def index_dataset(self, overwrite=True):
        """
        Index the dataset documents

        Args:
            overwrite: Whether to overwrite existing index
        """
        try:
            print(f"Starting indexing for dataset: {self.dataset_name}")
            print(f"Index will be stored at: {self.index_path}")

            # Create index directory if it doesn't exist
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

            # Check if index already exists
            if os.path.exists(self.index_path) and not overwrite:
                print(f"Index already exists at {self.index_path}. Use --overwrite to recreate.")
                return

            # Create indexer
            indexer = pt.index.IterDictIndexer(self.index_path, overwrite=overwrite)

            # Get corpus iterator
            corpus_iter = self.dataset.get_corpus_iter()

            print("Starting indexing process...")
            start_time = time.time()

            # Index the documents
            index_ref = indexer.index(corpus_iter)

            end_time = time.time()
            print(f"Indexing completed in {end_time - start_time:.2f} seconds")

            # Load the index to get statistics
            index = pt.IndexFactory.of(index_ref)
            stats = index.getCollectionStatistics()

            print(f"Index statistics:")
            print(f"  Number of documents: {stats.getNumberOfDocuments():,}")
            print(f"  Number of terms: {stats.getNumberOfUniqueTerms():,}")
            print(f"  Number of tokens: {stats.getNumberOfTokens():,}")
            print(f"  Average document length: {stats.getAverageDocumentLength():.2f}")

            return index_ref

        except Exception as e:
            print(f"Error during indexing: {e}")
            sys.exit(1)

    def load_index(self):
        """Load existing index"""
        try:
            if not os.path.exists(self.index_path):
                print(f"Index not found at {self.index_path}. Please run indexing first.")
                return None

            print(f"Loading index from: {self.index_path}")
            self.index = pt.IndexFactory.of(self.index_path)

            # Print index statistics
            stats = self.index.getCollectionStatistics()
            print(f"Loaded index with {stats.getNumberOfDocuments():,} documents")

            return self.index

        except Exception as e:
            print(f"Error loading index: {e}")
            return None

    def setup_retriever(self, wmodel="BM25", num_results=1000, qe_method="none", qe_params=None, **kwargs):
        """
        Setup BM25 retriever with optional query expansion

        Args:
            wmodel: Weighting model (default: BM25)
            num_results: Number of results to retrieve per query
            qe_method: Query expansion method ('none', 'bo1', 'rm3', 'kl')
            qe_params: Query expansion parameters dict
            **kwargs: Additional parameters for the retriever
        """
        try:
            if self.index is None:
                self.load_index()
                if self.index is None:
                    return None

            print(f"Setting up {wmodel} retriever...")

            # Create retriever with specified parameters
            controls = {}
            if 'k1' in kwargs:
                controls['bm25.k_1'] = str(kwargs['k1'])
            if 'b' in kwargs:
                controls['bm25.b'] = str(kwargs['b'])

            base_retriever = pt.terrier.Retriever(
                self.index,
                wmodel=wmodel,
                num_results=num_results,
                controls=controls if controls else None
            )

            # Set up query expansion if requested
            if qe_method != "none":
                print(f"Setting up query expansion: {qe_method}")

                if qe_params is None:
                    qe_params = {}

                if qe_method == "bo1":
                    # Bo1 Query Expansion
                    qe = pt.rewrite.Bo1QueryExpansion(
                        self.index,
                        fb_terms=qe_params.get('qe_terms', 10),
                        fb_docs=qe_params.get('qe_docs', 3)
                    )
                    self.retriever = base_retriever >> qe >> base_retriever

                elif qe_method == "rm3":
                    # RM3 Query Expansion
                    qe = pt.rewrite.RM3(
                        self.index,
                        fb_terms=qe_params.get('qe_terms', 10),
                        fb_docs=qe_params.get('qe_docs', 3),
                        fb_lambda=qe_params.get('fb_lambda', 0.5)
                    )
                    self.retriever = base_retriever >> qe >> base_retriever

                elif qe_method == "kl":
                    # KL Query Expansion
                    qe = pt.rewrite.KLQueryExpansion(
                        self.index,
                        fb_terms=qe_params.get('qe_terms', 10),
                        fb_docs=qe_params.get('qe_docs', 3),
                        fb_mu=qe_params.get('fb_mu', 500)
                    )
                    self.retriever = base_retriever >> qe >> base_retriever

                print(f"Query expansion pipeline created: BM25 >> {qe_method.upper()} >> BM25")
                print(f"  Expansion terms: {qe_params.get('qe_terms', 10)}")
                print(f"  Feedback docs: {qe_params.get('qe_docs', 3)}")
                if qe_method == "rm3":
                    print(f"  FB Lambda: {qe_params.get('fb_lambda', 0.5)}")
                elif qe_method == "kl":
                    print(f"  FB Mu: {qe_params.get('fb_mu', 500)}")
            else:
                self.retriever = base_retriever
                print(f"Base retriever setup complete (no query expansion)")

            return self.retriever

        except Exception as e:
            print(f"Error setting up retriever: {e}")
            return None

    def retrieve_queries(self, output_file, run_name="BM25_run", custom_queries=None):
        """
        Perform retrieval on dataset queries

        Args:
            output_file: Output file for TREC run
            run_name: Name for the run
            custom_queries: Optional custom queries DataFrame
        """
        try:
            if self.retriever is None:
                self.setup_retriever()
                if self.retriever is None:
                    return

            # Get queries
            if custom_queries is not None:
                queries = custom_queries
                print(f"Using {len(queries)} custom queries")
            else:
                print("Loading dataset queries...")
                queries = self.dataset.get_topics()
                print(f"Loaded {len(queries)} queries from dataset")

            # Display sample queries
            print("\nSample queries:")
            for i, row in queries.head(3).iterrows():
                qid = row.get('qid', row.get('query_id', 'unknown'))
                query_text = row.get('query', row.get('text', 'unknown'))
                print(f"  {qid}: {query_text}")

            print(f"\nStarting retrieval...")
            start_time = time.time()

            # Perform retrieval
            results = self.retriever.transform(queries)

            end_time = time.time()
            print(f"Retrieval completed in {end_time - start_time:.2f} seconds")

            # Display results summary
            print(f"\nResults summary:")
            print(f"  Total query-document pairs: {len(results):,}")
            print(f"  Unique queries: {results['qid'].nunique()}")
            print(f"  Average results per query: {len(results) / results['qid'].nunique():.1f}")

            # Show sample results
            print(f"\nSample results:")
            sample_results = results.head(5)
            for _, row in sample_results.iterrows():
                print(f"  Query {row['qid']}: Doc {row['docno']} (rank {row['rank']}, score {row['score']:.4f})")

            # Save results
            print(f"\nSaving results to: {output_file}")
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            pt.io.write_results(results, output_file, format="trec", run_name=run_name)
            print(f"TREC run file saved successfully")

            return results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return None

    def search_single_query(self, query_text, num_results=10, qe_method="none", qe_params=None):
        """
        Search a single query

        Args:
            query_text: Query string
            num_results: Number of results to return
            qe_method: Query expansion method
            qe_params: Query expansion parameters
        """
        try:
            if self.retriever is None:
                self.setup_retriever(qe_method=qe_method, qe_params=qe_params)
                if self.retriever is None:
                    return None

            print(f"Searching: '{query_text}'")
            if qe_method != "none":
                print(f"Using query expansion: {qe_method}")

            results = self.retriever.search(query_text)

            print(f"\nTop {min(num_results, len(results))} results:")
            for i, row in results.head(num_results).iterrows():
                print(f"  {row['rank'] + 1}. Doc {row['docno']} (score: {row['score']:.4f})")

            return results.head(num_results)

        except Exception as e:
            print(f"Error searching query: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='PyTerrier Index and Retrieve Script')

    # Required arguments
    parser.add_argument('--mode', required=True, choices=['index', 'retrieve', 'both', 'search'],
                        help='Mode: index, retrieve, both, or search')
    parser.add_argument('--qe-method', choices=['none', 'bo1', 'rm3', 'kl'], default='none',
                        help='Query expansion method: none, bo1, rm3, or kl')
    parser.add_argument('--dataset', required=True,
                        # choices=['beir/dbpedia-entity', 'beir/nq', 'beir/hotpotqa'],
                        help='Dataset to use')
    parser.add_argument('--index-path', required=True, help='Path for the index')

    # Optional arguments
    parser.add_argument('--output-file', help='Output TREC run file (required for retrieve/both modes)')
    parser.add_argument('--run-name', default='BM25_run', help='Run name for TREC file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing index')
    parser.add_argument('--num-results', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--query', help='Single query for search mode')

    # BM25 parameters
    parser.add_argument('--k1', type=float, default=1.2, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter')

    # Query expansion parameters
    parser.add_argument('--qe-terms', type=int, default=10, help='Number of expansion terms (default: 10)')
    parser.add_argument('--qe-docs', type=int, default=3, help='Number of feedback documents (default: 3)')
    parser.add_argument('--fb-lambda', type=float, default=0.5, help='Feedback lambda for RM3 (default: 0.5)')
    parser.add_argument('--fb-mu', type=float, default=500, help='Feedback mu for KL (default: 500)')

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ['retrieve', 'both'] and not args.output_file:
        print("Error: --output_file is required for retrieve and both modes")
        sys.exit(1)

    if args.mode == 'search' and not args.query:
        print("Error: --query is required for search mode")
        sys.exit(1)

    # Initialize processor
    processor = PyTerrierProcessor(args.dataset, args.index_path)

    try:
        # Execute based on mode
        if args.mode == 'index':
            print("=== INDEXING MODE ===")
            processor.index_dataset(overwrite=args.overwrite)

        elif args.mode == 'retrieve':
            print("=== RETRIEVAL MODE ===")
            qe_params = {
                'qe_terms': args.qe_terms,
                'qe_docs': args.qe_docs,
                'fb_lambda': args.fb_lambda,
                'fb_mu': args.fb_mu
            }
            processor.setup_retriever(
                num_results=args.num_results,
                qe_method=args.qe_method,
                qe_params=qe_params,
                k1=args.k1,
                b=args.b
            )
            processor.retrieve_queries(args.output_file, args.run_name)

        elif args.mode == 'both':
            print("=== INDEX AND RETRIEVE MODE ===")
            processor.index_dataset(overwrite=args.overwrite)
            qe_params = {
                'qe_terms': args.qe_terms,
                'qe_docs': args.qe_docs,
                'fb_lambda': args.fb_lambda,
                'fb_mu': args.fb_mu
            }
            processor.setup_retriever(
                num_results=args.num_results,
                qe_method=args.qe_method,
                qe_params=qe_params,
                k1=args.k1,
                b=args.b
            )
            processor.retrieve_queries(args.output_file, args.run_name)

        elif args.mode == 'search':
            print("=== SEARCH MODE ===")
            qe_params = {
                'qe_terms': args.qe_terms,
                'qe_docs': args.qe_docs,
                'fb_lambda': args.fb_lambda,
                'fb_mu': args.fb_mu
            }
            processor.search_single_query(
                args.query,
                args.num_results,
                qe_method=args.qe_method,
                qe_params=qe_params
            )

        print("\n=== COMPLETED SUCCESSFULLY ===")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()