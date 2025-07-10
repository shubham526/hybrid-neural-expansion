#!/usr/bin/env python3
"""
PyTerrier Index and Retrieve Script

This script can index documents and perform retrieval using BM25 on BEIR datasets.
Supports beir/dbpedia-entity and beir/nq datasets.
Now supports custom queries from TSV files.

Usage:
    # Index a dataset
    python pyterrier_script.py --mode index --dataset beir/dbpedia-entity --index_path ./dbpedia-index

    # Retrieve from indexed dataset using dataset queries
    python pyterrier_script.py --mode retrieve --dataset beir/dbpedia-entity --index_path ./dbpedia-index --output_file run.txt

    # Retrieve using custom queries from TSV file
    python pyterrier_script.py --mode retrieve --dataset beir/dbpedia-entity --index_path ./dbpedia-index --output_file run.txt --queries_file queries.tsv

    # Index and then retrieve with custom queries
    python pyterrier_script.py --mode both --dataset beir/nq --index_path ./nq-index --output_file run.txt --queries_file my_queries.tsv
"""

import argparse
import os
import sys
import time
import pandas as pd
from pathlib import Path
import ir_datasets

try:
    import pyterrier as pt
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install: pip install python-terrier pandas tqdm")
    sys.exit(1)


def preprocess_queries(queries_df):
    """
    Preprocess queries to handle special characters that cause parsing errors.

    Args:
        queries_df: DataFrame with 'qid' and 'query' columns

    Returns:
        DataFrame with cleaned queries
    """

    def clean_query(query_text):
        """Clean individual query text"""
        if pd.isna(query_text):
            return ""

        # Convert to string
        query = str(query_text)

        # More aggressive cleaning for Terrier compatibility
        import re

        # Remove all punctuation except spaces, hyphens, and periods in numbers
        # Keep only: letters, numbers, spaces, hyphens
        query = re.sub(r'[^\w\s\-]', ' ', query)

        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)

        # Remove leading/trailing whitespace
        query = query.strip()

        # Ensure query is not empty
        if not query:
            return "empty query"

        return query

    print("Preprocessing queries for Terrier compatibility...")

    # Store original queries for logging
    original_queries = queries_df['query'].tolist()

    # Clean queries
    queries_df = queries_df.copy()
    queries_df['query'] = queries_df['query'].apply(clean_query)

    # Log changes
    changes_count = 0
    for i, (orig, cleaned) in enumerate(zip(original_queries, queries_df['query'].tolist())):
        if orig != cleaned:
            changes_count += 1
            if changes_count <= 5:  # Show first 5 changes
                qid = queries_df.iloc[i]['qid']
                print(f"  Query {qid}: '{orig}' → '{cleaned}'")

    if changes_count > 5:
        print(f"  ... and {changes_count - 5} more queries were cleaned")
    elif changes_count > 0:
        print(f"  Total queries cleaned: {changes_count}")
    else:
        print("  No queries needed cleaning")

    return queries_df


def load_queries_from_tsv(queries_file):
    """
    Load queries from TSV file.

    Supports files with or without headers.
    Expected formats:
    1. No header: qid\tquery (first column = qid, second column = query)
    2. With header: any column names, auto-detected
    3. Single column: treats as query with auto-generated qids

    Args:
        queries_file: Path to TSV file

    Returns:
        pandas DataFrame with 'qid' and 'query' columns
    """
    try:
        print(f"Loading queries from: {queries_file}")

        # First, try to detect if file has headers by reading first few lines
        with open(queries_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip() if f else ""

        # Split by tab to see structure
        first_parts = first_line.split('\t')
        second_parts = second_line.split('\t') if second_line else []

        print(f"First line: {len(first_parts)} columns")
        print(f"Second line: {len(second_parts)} columns")

        # Heuristic to detect headers:
        # If first line contains common header words, assume it has headers
        header_keywords = ['qid', 'query', 'id', 'text', 'question', 'title', 'queryid', 'query_id']
        has_headers = any(keyword.lower() in first_line.lower() for keyword in header_keywords)

        if has_headers:
            print("Detected headers in file")
            # Read with headers
            df = pd.read_csv(queries_file, sep='\t', dtype=str)

            print(f"TSV shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")

            # Auto-detect column names
            qid_col = None
            query_col = None

            # Look for qid column (case insensitive)
            for col in df.columns:
                if col.lower() in ['qid', 'query_id', 'id', 'queryid']:
                    qid_col = col
                    break

            # Look for query column (case insensitive)
            for col in df.columns:
                if col.lower() in ['query', 'text', 'query_text', 'question']:
                    query_col = col
                    break

            if qid_col is None:
                # If no clear qid column, use first column
                qid_col = df.columns[0]
                print(f"Warning: No clear qid column found, using first column: {qid_col}")

            if query_col is None:
                # If no clear query column, use second column
                if len(df.columns) > 1:
                    query_col = df.columns[1]
                    print(f"Warning: No clear query column found, using second column: {query_col}")
                else:
                    raise ValueError("Could not find query column in TSV file")

            print(f"Using columns: qid='{qid_col}', query='{query_col}'")

            # Create standardized dataframe
            queries_df = pd.DataFrame({
                'qid': df[qid_col].astype(str),
                'query': df[query_col].astype(str)
            })

        else:
            print("No headers detected, assuming first column = qid, second column = query")

            # Read without headers
            df = pd.read_csv(queries_file, sep='\t', header=None, dtype=str)

            print(f"TSV shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")

            if len(df.columns) == 1:
                # Single column - treat as queries with auto-generated IDs
                print("Single column detected, generating automatic qids")
                queries_df = pd.DataFrame({
                    'qid': [f"q{i + 1}" for i in range(len(df))],
                    'query': df[0].astype(str)
                })
            elif len(df.columns) >= 2:
                # Multiple columns - use first as qid, second as query
                print("Multiple columns detected, using first=qid, second=query")
                queries_df = pd.DataFrame({
                    'qid': df[0].astype(str),
                    'query': df[1].astype(str)
                })
            else:
                raise ValueError("Empty or invalid TSV file")

        # Remove any rows with missing values
        original_len = len(queries_df)
        queries_df = queries_df.dropna()
        if len(queries_df) < original_len:
            print(f"Warning: Removed {original_len - len(queries_df)} rows with missing values")

        # Remove empty queries
        queries_df = queries_df[queries_df['query'].str.strip() != '']
        if len(queries_df) < original_len:
            print(f"Warning: Removed {original_len - len(queries_df)} rows with empty queries")

        # Preprocess queries to handle special characters
        queries_df = preprocess_queries(queries_df)

        print(f"Loaded {len(queries_df)} queries successfully")

        # Show sample queries
        print("\nSample queries from TSV:")
        for i, row in queries_df.head(3).iterrows():
            print(f"  {row['qid']}: {row['query']}")

        return queries_df

    except Exception as e:
        print(f"Error loading queries from TSV file: {e}")
        print("\nExpected TSV formats:")
        print("\n1. Without headers:")
        print("1\\tWhat is the capital of France?")
        print("2\\tHow does photosynthesis work?")
        print("\n2. With headers:")
        print("qid\\tquery")
        print("1\\tWhat is the capital of France?")
        print("2\\tHow does photosynthesis work?")
        print("\n3. Single column (auto-generated IDs):")
        print("What is the capital of France?")
        print("How does photosynthesis work?")
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
            self.dataset = ir_datasets.load(self.dataset_name)
            print(f"Dataset loaded successfully")
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")
            print("Available datasets include: beir/dbpedia-entity, beir/nq, beir/hotpotqa, etc.")
            sys.exit(1)

    def index_dataset(self, overwrite=True):
        """
        Index the dataset documents with text storage for neural rerankers

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

            def ir_dataset_generate():
                """Generator that yields documents in PyTerrier format with text storage."""
                print("Generating documents from ir_datasets...")


                for doc in tqdm(self.dataset.docs_iter(), desc="Processing documents", total=self.dataset.docs_count()):

                    # print(doc.doc_text)
                    # print(doc.doc_id)


                    # Extract document text using the same logic as IRDatasetHandler
                    doc_text = self._extract_document_text_for_indexing(doc)

                    if doc_text.strip():  # Only index non-empty documents
                        yield {
                            'docno': doc.doc_id,
                            'text': doc_text
                        }

            # Create indexer with metadata storage for document text
            print("Creating indexer with text storage...")
            indexer = pt.index.IterDictIndexer(
                self.index_path,
                overwrite=overwrite,
                meta={
                    'docno': 64,  # Store document ID (up to 64 chars)
                    'text': 8192  # Store full text (up to 8KB per document)
                }
            )

            print("Starting indexing process with text storage...")
            start_time = time.time()

            # Index the documents using our generator
            index_ref = indexer.index(ir_dataset_generate())

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

            # Test that text storage is working
            print("\nTesting text storage...")
            index_obj = pt.IndexFactory.of(index_ref)
            meta_index = index_obj.getMetaIndex()

            try:
                if len(meta_index.getKeys())  > 0:
                    print("✓ Metadata storage confirmed - document text will be available")

                    # Show sample stored text
                    sample_keys = ['docno', 'text']
                    if stats.getNumberOfDocuments() > 0:
                        try:
                            sample_meta = meta_index.getItem('docno', 0)
                            sample_text = meta_index.getItem('text', 0)
                            print(f"  Sample document: {sample_meta}")
                            print(f"  Sample text: {sample_text[:100]}...")
                        except:
                            print("  Could not retrieve sample text (may be normal)")
                else:
                    print("⚠ Warning: No metadata stored - text may not be available for reranking")
            except:
                print(" Error when accessing metadata storage")

            return index_ref

        except Exception as e:
            print(f"Error during indexing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _extract_document_text_for_indexing(self, doc) -> str:
        """
        Extract text from document for indexing (similar to IRDatasetHandler but for indexing).

        Args:
            doc: Document object from ir_datasets

        Returns:
            Combined document text
        """
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

    def retrieve_queries(self, output_file, run_name="BM25_run", custom_queries=None, queries_file=None,
                         query_field="auto"):
        """
        Perform retrieval on dataset queries with progress tracking

        Args:
            output_file: Output file for TREC run
            run_name: Name for the run
            custom_queries: Optional custom queries DataFrame
            queries_file: Optional path to TSV file with queries
            query_field: Which field to use for dataset queries ('title', 'description', 'narrative', or 'auto')
        """
        try:
            if self.retriever is None:
                self.setup_retriever()
                if self.retriever is None:
                    return

            # Get queries (priority: custom_queries > queries_file > dataset queries)
            if custom_queries is not None:
                queries = custom_queries
                print(f"Using {len(queries)} custom queries from DataFrame")
            elif queries_file is not None:
                queries = load_queries_from_tsv(queries_file)
                print(f"Using {len(queries)} queries from TSV file: {queries_file}")
            else:
                # ==================================================================
                # START: IMPROVED LOGIC FOR LOADING DATASET QUERIES
                # ==================================================================
                print("Loading dataset queries...")
                queries_list = []
                # 1. Load all relevant fields from the ir_datasets query objects
                for query in self.dataset.queries_iter():
                    # Use getattr to safely access attributes that might not exist in all datasets
                    queries_list.append({
                        'qid': getattr(query, 'query_id', None),
                        'title': getattr(query, 'title', ''),
                        'description': getattr(query, 'description', ''),
                        'narrative': getattr(query, 'narrative', ''),
                        'text': getattr(query, 'text', '')  # Keep for datasets that use .text
                    })
                queries = pd.DataFrame(queries_list)
                print(f"Loaded {len(queries)} potential query fields from dataset")

                # 2. Now, select the correct field to use as the 'query'
                available_fields = [col for col in ['title', 'description', 'narrative', 'text'] if
                                    col in queries.columns and queries[col].str.strip().any()]
                print(f"Available query fields with content: {available_fields}")

                selected_field = query_field
                if selected_field == "auto":
                    if 'title' in available_fields:
                        selected_field = 'title'
                    elif 'text' in available_fields:
                        selected_field = 'text'
                    elif 'description' in available_fields:
                        selected_field = 'description'
                    else:
                        selected_field = available_fields[0]

                if selected_field not in available_fields:
                    print(f"Error: Requested query field '{selected_field}' not available or has no content.")
                    print(f"Available fields: {available_fields}")
                    return None

                print(f"Using '{selected_field}' as the query field.")

                # 3. Create the final 'query' column
                queries['query'] = queries[selected_field]

                # Ensure we have a valid qid column
                if 'qid' not in queries.columns or queries['qid'].isnull().any():
                    queries['qid'] = queries.index.astype(str)

                # Select only the essential columns for the rest of the pipeline
                queries = queries[['qid', 'query']]
                # ==================================================================
                # END: IMPROVED LOGIC
                # ==================================================================

            if queries_file is None:  # Only preprocess if not already done by load_queries_from_tsv
                queries = preprocess_queries(queries)

            # Display sample queries
            print("\nSample queries for retrieval:")
            for i, row in queries.head(3).iterrows():
                qid = row.get('qid', row.get('query_id', 'unknown'))
                query_text = row.get('query', row.get('text', 'unknown'))
                print(f"  {qid}: {query_text}")

            print(f"\nStarting retrieval for {len(queries)} queries...")
            start_time = time.time()

            # Create progress bar
            progress_bar = tqdm(
                total=len(queries),
                desc="Processing queries",
                unit="query",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} queries [{elapsed}<{remaining}, {rate_fmt}]"
            )

            # Process queries in batches to show progress
            batch_size = max(1, len(queries) // 20)  # 20 updates for the progress bar
            all_results = []

            try:
                for i in range(0, len(queries), batch_size):
                    batch_queries = queries.iloc[i:i + batch_size]

                    # Perform retrieval on this batch
                    batch_results = self.retriever.transform(batch_queries)
                    all_results.append(batch_results)

                    # Update progress bar
                    progress_bar.update(len(batch_queries))

                    # Optional: Show current query being processed
                    if len(batch_queries) > 0:
                        current_qid = batch_queries.iloc[0].get('qid', 'unknown')
                        progress_bar.set_postfix({"Current": current_qid})

                # Combine all results
                if all_results:
                    results = pd.concat(all_results, ignore_index=True)
                else:
                    results = pd.DataFrame()

            finally:
                progress_bar.close()

            end_time = time.time()
            print(f"\nRetrieval completed in {end_time - start_time:.2f} seconds")

            # Display results summary
            if not results.empty:
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
            else:
                print("\nWarning: No results generated!")

            return results

        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
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
    parser = argparse.ArgumentParser(description='PyTerrier Index and Retrieve Script with Custom Queries Support')

    # Required arguments
    parser.add_argument('--mode', required=True, choices=['index', 'retrieve', 'both', 'search'],
                        help='Mode: index, retrieve, both, or search')
    parser.add_argument('--dataset', required=True,
                        help='Dataset to use (e.g., beir/dbpedia-entity, beir/nq)')
    parser.add_argument('--index-path', required=True, help='Path for the index')

    # Optional arguments
    parser.add_argument('--output-file', help='Output TREC run file (required for retrieve/both modes)')
    parser.add_argument('--queries-file', help='Path to TSV file containing custom queries (qid, query)')
    parser.add_argument('--run-name', default='BM25_run', help='Run name for TREC file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing index')
    parser.add_argument('--num-results', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--query', help='Single query for search mode')
    parser.add_argument('--query-field', choices=['title', 'description', 'narrative', 'auto'],
                        default='auto', help='Query field to use from dataset (default: auto)')

    # Query expansion parameters
    parser.add_argument('--qe-method', choices=['none', 'bo1', 'rm3', 'kl'], default='none',
                        help='Query expansion method: none, bo1, rm3, or kl')
    parser.add_argument('--qe-terms', type=int, default=10, help='Number of expansion terms (default: 10)')
    parser.add_argument('--qe-docs', type=int, default=3, help='Number of feedback documents (default: 3)')
    parser.add_argument('--fb-lambda', type=float, default=0.5, help='Feedback lambda for RM3 (default: 0.5)')
    parser.add_argument('--fb-mu', type=float, default=500, help='Feedback mu for KL (default: 500)')

    # BM25 parameters
    parser.add_argument('--k1', type=float, default=1.2, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, default=0.75, help='BM25 b parameter')

    args = parser.parse_args()

    # Validate arguments
    if args.mode in ['retrieve', 'both'] and not args.output_file:
        print("Error: --output-file is required for retrieve and both modes")
        sys.exit(1)

    if args.mode == 'search' and not args.query:
        print("Error: --query is required for search mode")
        sys.exit(1)

    if args.queries_file and not os.path.exists(args.queries_file):
        print(f"Error: Queries file not found: {args.queries_file}")
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
            processor.retrieve_queries(
                output_file=args.output_file,
                run_name=args.run_name,
                queries_file=args.queries_file,
                query_field=args.query_field
            )

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
            processor.retrieve_queries(
                output_file=args.output_file,
                run_name=args.run_name,
                queries_file=args.queries_file,
                query_field=args.query_field
            )

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