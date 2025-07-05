#!/usr/bin/env python3
"""
MS MARCO vs Robust Index Comparison

This script compares the structure and content of MS MARCO and Robust indexes
to identify why RM3 works on one but not the other.
"""

import sys
from pathlib import Path
import re
from collections import Counter

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.utils.lucene_utils import initialize_lucene, get_lucene_classes


def analyze_index(index_path: str, index_name: str, test_query: str = "what is baby boomers means"):
    """Analyze an index structure and content."""

    print(f"\n{'=' * 60}")
    print(f"ANALYZING {index_name.upper()} INDEX")
    print(f"{'=' * 60}")
    print(f"Path: {index_path}")

    classes = get_lucene_classes()
    FSDirectory = classes['FSDirectory']
    Paths = classes['Path']
    DirectoryReader = classes['DirectoryReader']
    IndexSearcher = classes['IndexSearcher']
    BM25Similarity = classes['BM25Similarity']
    BooleanQueryBuilder = classes['BooleanQueryBuilder']
    TermQuery = classes['TermQuery']
    Term = classes['Term']
    Occur = classes['BooleanClauseOccur']
    EnglishAnalyzer = classes['EnglishAnalyzer']
    CharTermAttribute = classes['CharTermAttribute']

    try:
        # Open index
        directory = FSDirectory.open(Paths.get(index_path))
        reader = DirectoryReader.open(directory)
        searcher = IndexSearcher(reader.getContext())
        searcher.setSimilarity(BM25Similarity())
        analyzer = EnglishAnalyzer()

        print(f"✓ Index opened successfully")
        print(f"  Documents: {reader.numDocs():,}")
        print(f"  Deleted docs: {reader.numDeletedDocs():,}")

    except Exception as e:
        print(f"❌ Failed to open index: {e}")
        return {}

    # Analyze document structure
    print(f"\n📄 DOCUMENT STRUCTURE")
    print("-" * 30)

    field_analysis = {}

    try:
        # Sample multiple documents to get a good picture
        sample_docs = min(5, reader.numDocs())

        for doc_id in range(sample_docs):
            doc = searcher.storedFields().document(doc_id)
            fields = [field.name() for field in doc.getFields()]

            print(f"Document {doc_id} fields: {fields}")

            # Analyze each field
            for field_name in fields:
                if field_name not in field_analysis:
                    field_analysis[field_name] = {
                        'count': 0,
                        'total_length': 0,
                        'samples': []
                    }

                field_value = doc.get(field_name)
                if field_value:
                    field_analysis[field_name]['count'] += 1
                    field_analysis[field_name]['total_length'] += len(field_value)

                    if len(field_analysis[field_name]['samples']) < 2:
                        field_analysis[field_name]['samples'].append(field_value[:100])

        print(f"\nField Analysis:")
        for field_name, stats in field_analysis.items():
            avg_length = stats['total_length'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {field_name}:")
            print(f"    Documents with content: {stats['count']}/{sample_docs}")
            print(f"    Average length: {avg_length:.1f} chars")
            if stats['samples']:
                print(f"    Sample: '{stats['samples'][0]}...'")

    except Exception as e:
        print(f"❌ Error analyzing document structure: {e}")
        return {}

    # Test RM3 feedback document search
    print(f"\n🔍 RM3 FEEDBACK SEARCH TEST")
    print("-" * 30)
    print(f"Test query: '{test_query}'")

    def tokenize_query(query_str: str):
        tokens = []
        try:
            token_stream = analyzer.tokenStream("contents", query_str)
            char_term_attr = token_stream.addAttribute(CharTermAttribute)
            token_stream.reset()
            while token_stream.incrementToken():
                tokens.append(char_term_attr.toString())
            token_stream.close()
        except Exception as e:
            print(f"    Tokenization error: {e}")
        return tokens

    try:
        query_terms = tokenize_query(test_query)
        print(f"Query terms: {query_terms}")

        if not query_terms:
            print("❌ No query terms after analysis!")
            return field_analysis

        # Try searching in different fields
        search_results = {}

        for field_name in field_analysis.keys():
            if field_analysis[field_name]['count'] > 0:  # Only try fields with content
                try:
                    builder = BooleanQueryBuilder()
                    for term in query_terms:
                        term_query = TermQuery(Term(field_name, term))
                        builder.add(term_query, Occur.SHOULD)

                    boolean_query = builder.build()
                    top_docs = searcher.search(boolean_query, 10)

                    search_results[field_name] = {
                        'total_hits': top_docs.totalHits.value,
                        'retrieved': len(top_docs.scoreDocs),
                        'top_score': top_docs.scoreDocs[0].score if top_docs.scoreDocs else 0
                    }

                    print(f"  Search in '{field_name}': {top_docs.totalHits.value} hits")

                except Exception as e:
                    print(f"  Error searching '{field_name}': {e}")

        # Find best field for feedback
        best_field = max(search_results.keys(),
                         key=lambda f: search_results[f]['total_hits']) if search_results else 'contents'

        print(f"Best field for search: '{best_field}'")

        # Test term extraction from feedback documents
        if best_field in search_results and search_results[best_field]['total_hits'] > 0:
            print(f"\n🔬 TERM EXTRACTION TEST (using '{best_field}' field)")
            print("-" * 30)

            # Get feedback documents
            builder = BooleanQueryBuilder()
            for term in query_terms:
                term_query = TermQuery(Term(best_field, term))
                builder.add(term_query, Occur.SHOULD)

            boolean_query = builder.build()
            top_docs = searcher.search(boolean_query, 5)  # Test with 5 docs

            total_terms_extracted = 0
            docs_with_terms = 0

            for i, score_doc in enumerate(top_docs.scoreDocs):
                doc = searcher.storedFields().document(score_doc.doc)
                doc_content = doc.get(best_field)

                if doc_content:
                    print(f"  Doc {i} (score: {score_doc.score:.3f}):")
                    print(f"    Content length: {len(doc_content)}")
                    print(f"    Preview: '{doc_content[:80]}...'")

                    # Extract terms like RM3 does
                    original_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', doc_content.lower())
                    print(f"    Original tokens: {len(original_tokens)}")

                    if original_tokens:
                        # Test stemming
                        stemmed_tokens = []
                        for token in original_tokens[:50]:  # Test first 50
                            stemmed = tokenize_query(token)
                            stemmed_tokens.extend(stemmed)

                        if stemmed_tokens:
                            docs_with_terms += 1
                            total_terms_extracted += len(stemmed_tokens)
                            unique_terms = len(set(stemmed_tokens))
                            print(f"    Stemmed tokens: {len(stemmed_tokens)} ({unique_terms} unique)")

                            # Show most common terms
                            term_counts = Counter(stemmed_tokens)
                            top_terms = term_counts.most_common(5)
                            print(f"    Top terms: {dict(top_terms)}")
                        else:
                            print(f"    ❌ No stemmed tokens produced!")
                    else:
                        print(f"    ❌ No original tokens found!")

            print(f"\nExtraction Summary:")
            print(f"  Documents with extracted terms: {docs_with_terms}/{len(top_docs.scoreDocs)}")
            print(f"  Total terms extracted: {total_terms_extracted}")

            if docs_with_terms == 0:
                print(f"  ❌ ROOT CAUSE: No terms extracted from {index_name} feedback documents!")
            elif total_terms_extracted < 10:
                print(f"  ⚠️  Very few terms extracted from {index_name}")
            else:
                print(f"  ✓ Term extraction successful for {index_name}")

    except Exception as e:
        print(f"❌ Error in feedback search test: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    reader.close()
    directory.close()

    return field_analysis


def compare_indexes(msmarco_path: str, robust_path: str, lucene_path: str):
    """Compare MS MARCO and Robust indexes."""

    print("🔄 INITIALIZING LUCENE")
    if not initialize_lucene(lucene_path):
        print("❌ Failed to initialize Lucene")
        return

    print("✓ Lucene initialized")

    # Analyze both indexes
    msmarco_analysis = analyze_index(msmarco_path, "MS MARCO", "what is baby boomers means")
    robust_analysis = analyze_index(robust_path, "ROBUST", "information retrieval")

    # Compare results
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")

    print("\nField Comparison:")
    all_fields = set(msmarco_analysis.keys()) | set(robust_analysis.keys())

    for field in sorted(all_fields):
        msmarco_has = field in msmarco_analysis
        robust_has = field in robust_analysis

        status = ""
        if msmarco_has and robust_has:
            ms_avg = msmarco_analysis[field]['total_length'] / max(1, msmarco_analysis[field]['count'])
            rb_avg = robust_analysis[field]['total_length'] / max(1, robust_analysis[field]['count'])
            status = f"Both (MS: {ms_avg:.1f} chars, Robust: {rb_avg:.1f} chars)"
        elif msmarco_has:
            status = "MS MARCO only"
        elif robust_has:
            status = "Robust only"

        print(f"  {field}: {status}")

    print("\nRecommendations:")
    if 'contents' in msmarco_analysis and 'contents' in robust_analysis:
        ms_content = msmarco_analysis['contents']['total_length'] / max(1, msmarco_analysis['contents']['count'])
        rb_content = robust_analysis['contents']['total_length'] / max(1, robust_analysis['contents']['count'])

        if ms_content < rb_content * 0.1:
            print("  ⚠️  MS MARCO 'contents' field is much shorter than Robust")
            print("     This might cause insufficient term extraction for RM3")

        if msmarco_analysis['contents']['count'] == 0:
            print("  ❌ MS MARCO 'contents' field is empty!")
            print("     Try a different field name for MS MARCO")

    if 'contents' not in msmarco_analysis:
        print("  ❌ MS MARCO index doesn't have 'contents' field")
        print("     Update RM3 code to use the correct field name")
        if msmarco_analysis:
            best_field = max(msmarco_analysis.keys(),
                             key=lambda f: msmarco_analysis[f]['total_length'])
            print(f"     Suggested field: '{best_field}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare MS MARCO and Robust indexes")
    parser.add_argument('--msmarco-index', required=True, help='Path to MS MARCO index')
    parser.add_argument('--robust-index', required=True, help='Path to Robust index')
    parser.add_argument('--lucene-path', required=True, help='Path to Lucene JAR files')

    args = parser.parse_args()

    compare_indexes(args.msmarco_index, args.robust_index, args.lucene_path)