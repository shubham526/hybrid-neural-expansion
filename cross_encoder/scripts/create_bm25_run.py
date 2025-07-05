#!/usr/bin/env python3
"""
TREC BM25 Run File Generator

This script searches a Lucene index using BM25 similarity and generates
a TREC-style run file with the results.

Usage:
    python trec_bm25_runner.py --index_path /path/to/index --queries_file queries.txt --output_file run.txt

TREC Run File Format:
    query_id Q0 doc_id rank score run_name
"""

import argparse
import os
import sys
import logging
from cross_encoder.src.utils.lucene_utils import initialize_lucene, get_lucene_classes

logger = logging.getLogger(__name__)


class TrecBM25Runner:
    def __init__(self, index_path, lucene_path, search_field='contents', analyzer_type='standard'):
        """
        Initialize the TREC BM25 Runner

        Args:
            index_path: Path to the Lucene index
            search_field: Field name to search in (default: 'contents')
            analyzer_type: Type of analyzer ('standard' or 'english')
        """
        initialize_lucene(lucene_path)
        self.index_path = index_path
        self.search_field = search_field
        self.analyzer_type = analyzer_type
        classes = get_lucene_classes()
        for name, cls in classes.items():
            setattr(self, name, cls)

        # Initialize Lucene components
        self.directory = None
        self.reader = None
        self.searcher = None
        self.analyzer = None
        self.parser = None

        self._setup_lucene()

    def _setup_lucene(self):
        """Setup Lucene components"""
        try:
            # Open directory and reader
            self.directory = self.FSDirectory.open(self.Path.get(self.index_path))
            self.reader = self.DirectoryReader.open(self.directory)

            # Create searcher with BM25 similarity
            self.searcher = self.IndexSearcher(self.reader.getContext())
            self.searcher.setSimilarity(self.BM25Similarity())

            # Setup analyzer and parser
            if self.analyzer_type.lower() == 'english':
                self.analyzer = self.EnglishAnalyzer()
            else:
                self.analyzer = self.StandardAnalyzer()

            self.parser = self.QueryParser(self.search_field, self.analyzer)

            print(f"Successfully initialized Lucene with index: {self.index_path}")
            print(f"Index contains {self.reader.numDocs()} documents")

        except Exception as e:
            print(f"Error setting up Lucene: {e}")
            sys.exit(1)

    def search_query(self, query_text, max_results=1000):
        """
        Search for a query and return results

        Args:
            query_text: Query string to search
            max_results: Maximum number of results to return

        Returns:
            List of tuples (doc_id, score)
        """
        try:
            # Parse query
            query = self.parser.parse(query_text)

            # Search
            doc_hits = self.searcher.search(query, max_results)

            results = []
            for score_doc in doc_hits.scoreDocs:
                # Get the document
                document = self.searcher.doc(score_doc.doc)
                collection_doc_id = document.get('id')
                # If no collection ID found, use Lucene internal ID as fallback
                if not collection_doc_id:
                    collection_doc_id = str(score_doc.doc)
                    print(f"Warning: No collection document ID found, using Lucene ID: {collection_doc_id}")
                results.append((collection_doc_id, score_doc.score))

            return results

        except Exception as e:
            print(f"Error searching query '{query_text}': {e}")
            return []

    def load_queries(self, queries_file):
        """
        Load queries from file

        Supports formats:
        - Simple text file (one query per line, auto-numbered)
        - TREC format: query_id query_text
        - TSV format: query_id\tquery_text

        Args:
            queries_file: Path to queries file

        Returns:
            List of tuples (query_id, query_text)
        """
        queries = []

        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse as query_id query_text format
                    parts = line.split('\t') if '\t' in line else line.split(' ', 1)

                    if len(parts) >= 2:
                        query_id = parts[0]
                        query_text = parts[1]
                    else:
                        # Single query text, auto-number
                        query_id = str(line_num)
                        query_text = line

                    queries.append((query_id, query_text))

            print(f"Loaded {len(queries)} queries from {queries_file}")
            return queries

        except Exception as e:
            print(f"Error loading queries from {queries_file}: {e}")
            return []

    def generate_run_file(self, queries, output_file, run_name="BM25_run", max_results=1000):
        """
        Generate TREC run file

        Args:
            queries: List of (query_id, query_text) tuples
            output_file: Output file path
            run_name: Name for the run
            max_results: Maximum results per query
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for query_id, query_text in queries:
                    print(f"Processing query {query_id}: {query_text}")

                    results = self.search_query(query_text, max_results)

                    if not results:
                        print(f"No results found for query {query_id}")
                        continue

                    # Write results in TREC format
                    for rank, (doc_id, score) in enumerate(results, 1):
                        # TREC format: query_id Q0 doc_id rank score run_name
                        f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

                    print(f"Query {query_id}: {len(results)} results written")

            print(f"Run file generated: {output_file}")

        except Exception as e:
            print(f"Error generating run file: {e}")

    def close(self):
        """Close Lucene resources"""
        try:
            if self.reader:
                self.reader.close()
            if self.directory:
                self.directory.close()
        except Exception as e:
            print(f"Error closing Lucene resources: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate TREC BM25 run file from Lucene index')
    parser.add_argument('--index-path', required=True, help='Path to Lucene index')
    parser.add_argument('--lucene-path', required=True, help='Path to Lucene modules')
    parser.add_argument('--queries-file', required=True, help='Path to queries file')
    parser.add_argument('--output-file', required=True, help='Output run file path')
    parser.add_argument('--run-name', default='BM25_run', help='Run name (default: BM25_run)')
    parser.add_argument('--max-results', type=int, default=1000, help='Max results per query (default: 1000)')
    parser.add_argument('--search-field', default='contents', help='Field to search in (default: contents)')
    parser.add_argument('--analyzer', choices=['standard', 'english'], default='english',
                        help='Analyzer type (default: standard)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.index_path):
        print(f"Error: Index path does not exist: {args.index_path}")
        sys.exit(1)

    if not os.path.exists(args.queries_file):
        print(f"Error: Queries file does not exist: {args.queries_file}")
        sys.exit(1)

    # Initialize runner
    runner = TrecBM25Runner(
        index_path=args.index_path,
        search_field=args.search_field,
        analyzer_type=args.analyzer
    )

    try:
        # Load queries
        queries = runner.load_queries(args.queries_file)
        if not queries:
            print("No queries loaded. Exiting.")
            sys.exit(1)

        # Generate run file
        runner.generate_run_file(
            queries=queries,
            output_file=args.output_file,
            run_name=args.run_name,
            max_results=args.max_results
        )

        print("Done!")

    finally:
        runner.close()


if __name__ == "__main__":
    main()