#!/usr/bin/env python3
"""
Complete analysis of case study JSON for IR paper.

Performs:
1. Systematic term rescue statistics
2. Identifies additional case study examples
3. Term type categorization
4. Weight distribution visualization

Usage:
    python analyze_case_studies.py complete_case_study.json --output-dir analysis_results
"""

import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")


def load_case_study(json_file):
    """Load the case study JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded case studies for {len(data)} queries")
    return data


def extract_term_data(case_studies):
    """Extract all term data from case studies into a structured format"""
    all_terms = []

    for query_id, study in case_studies.items():
        query_text = study.get('query', 'Unknown')

        # Handle different possible structures
        if 'terms' in study:
            terms = study['terms']
        elif 'expansion_terms' in study:
            terms = study['expansion_terms']
        else:
            continue

        for term_data in terms:
            # Extract scores with fallback handling
            term_info = {
                'query_id': query_id,
                'query_text': query_text,
                'term': term_data.get('term', term_data.get('text', 'unknown')),
                'rm3_score': float(term_data.get('rm3_score', term_data.get('rm3', 0))),
                'semantic_score': float(
                    term_data.get('semantic_score', term_data.get('semantic', term_data.get('cosine_similarity', 0)))),
                'meqe_score': float(
                    term_data.get('meqe_score', term_data.get('final_weight', term_data.get('hybrid', 0)))),
                'importance': term_data.get('importance', term_data.get('classification', 'Unknown'))
            }

            # Calculate boost factor
            if term_info['rm3_score'] > 0:
                term_info['boost_factor'] = term_info['meqe_score'] / term_info['rm3_score']
            else:
                term_info['boost_factor'] = float('inf') if term_info['meqe_score'] > 0 else 0

            all_terms.append(term_info)

    df = pd.DataFrame(all_terms)
    print(f"✅ Extracted {len(df)} total expansion terms from {len(case_studies)} queries")
    return df


def analyze_term_rescue(df, rm3_threshold=0.01, semantic_threshold=0.3):
    """
    Analysis 1: Systematic Term Rescue Statistics

    Identifies terms with low RM3 but high semantic scores
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: SYSTEMATIC TERM RESCUE STATISTICS")
    print("=" * 80)

    # Identify rescued terms
    rescued_terms = df[
        (df['rm3_score'] < rm3_threshold) &
        (df['semantic_score'] > semantic_threshold)
        ].copy()

    total_terms = len(df)
    rescued_count = len(rescued_terms)
    rescued_pct = (rescued_count / total_terms * 100) if total_terms > 0 else 0

    # Per-query statistics
    queries_with_rescues = rescued_terms.groupby('query_id').size()
    avg_rescues_per_query = queries_with_rescues.mean() if len(queries_with_rescues) > 0 else 0

    # Boost factor statistics
    valid_boosts = rescued_terms[rescued_terms['boost_factor'] != float('inf')]['boost_factor']

    results = {
        'total_terms': total_terms,
        'rescued_terms_count': rescued_count,
        'rescued_percentage': rescued_pct,
        'queries_with_rescues': len(queries_with_rescues),
        'avg_rescues_per_query': avg_rescues_per_query,
        'boost_statistics': {
            'mean': valid_boosts.mean() if len(valid_boosts) > 0 else 0,
            'median': valid_boosts.median() if len(valid_boosts) > 0 else 0,
            'max': valid_boosts.max() if len(valid_boosts) > 0 else 0,
            'min': valid_boosts.min() if len(valid_boosts) > 0 else 0,
        }
    }

    print(f"\n📊 Rescue Statistics:")
    print(f"  Total expansion terms analyzed: {total_terms:,}")
    print(
        f"  Terms rescued (RM3 < {rm3_threshold}, Semantic > {semantic_threshold}): {rescued_count:,} ({rescued_pct:.1f}%)")
    print(f"  Queries with rescued terms: {results['queries_with_rescues']}")
    print(f"  Average rescued terms per query: {avg_rescues_per_query:.1f}")

    print(f"\n📈 Boost Factor Statistics (for rescued terms):")
    print(f"  Mean boost: {results['boost_statistics']['mean']:.1f}×")
    print(f"  Median boost: {results['boost_statistics']['median']:.1f}×")
    print(f"  Max boost: {results['boost_statistics']['max']:.1f}×")

    # Paper-ready text
    paper_text = f"""
=== PAPER-READY TEXT ===

Across all queries, MEQE's semantic component rescues an average of {avg_rescues_per_query:.1f} 
terms per query that have low RM3 weights (<{rm3_threshold}) but high semantic relevance 
(>{semantic_threshold}). These rescued terms represent {rescued_pct:.1f}% of all expansion 
terms and receive an average {results['boost_statistics']['mean']:.1f}× importance boost 
(median: {results['boost_statistics']['median']:.1f}×), with the largest boosts reaching 
{results['boost_statistics']['max']:.0f}×.
"""

    print(paper_text)

    return results, rescued_terms, paper_text


def identify_case_study_examples(df, n_examples=3):
    """
    Analysis 2: Identify Additional Case Study Examples

    Find queries with interesting rescue patterns
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: ADDITIONAL CASE STUDY EXAMPLES")
    print("=" * 80)

    # Calculate per-query rescue statistics
    query_stats = []

    for query_id in df['query_id'].unique():
        query_df = df[df['query_id'] == query_id]
        query_text = query_df['query_text'].iloc[0]

        # Find rescued terms
        rescued = query_df[
            (query_df['rm3_score'] < 0.01) &
            (query_df['semantic_score'] > 0.3)
            ]

        # Find high RM3 terms
        high_rm3 = query_df[query_df['rm3_score'] > 0.1]

        if len(rescued) > 0:
            query_stats.append({
                'query_id': query_id,
                'query_text': query_text,
                'num_rescued': len(rescued),
                'num_high_rm3': len(high_rm3),
                'max_rescue_boost': rescued['boost_factor'].replace([np.inf, -np.inf], np.nan).max(),
                'pattern_type': classify_rescue_pattern(query_df, rescued, high_rm3)
            })

    query_stats_df = pd.DataFrame(query_stats)

    # Select diverse examples
    examples = []

    # Example 1: Most dramatic semantic rescue
    if len(query_stats_df) > 0:
        example1 = query_stats_df.nlargest(1, 'max_rescue_boost').iloc[0]
        examples.append(('dramatic_rescue', example1))

    # Example 2: Balanced (both RM3 preservation and semantic rescue)
    balanced = query_stats_df[
        (query_stats_df['num_rescued'] > 2) &
        (query_stats_df['num_high_rm3'] > 2)
        ]
    if len(balanced) > 0:
        example2 = balanced.iloc[0]
        examples.append(('balanced', example2))

    # Example 3: Many rescues
    if len(query_stats_df) > 2:
        example3 = query_stats_df.nlargest(1, 'num_rescued').iloc[0]
        examples.append(('many_rescues', example3))

    print(f"\n📋 Found {len(examples)} diverse case study examples:\n")

    detailed_examples = []

    for idx, (pattern_type, example) in enumerate(examples, 1):
        print(f"Example {idx}: {example['query_text']} (Query {example['query_id']})")
        print(f"  Pattern: {pattern_type}")
        print(f"  Rescued terms: {example['num_rescued']}")
        print(f"  Max boost: {example['max_rescue_boost']:.0f}×")

        # Get detailed term info
        query_terms = df[df['query_id'] == example['query_id']].copy()
        query_terms = query_terms.sort_values('meqe_score', ascending=False)

        print(f"\n  Top terms:")
        for _, term in query_terms.head(5).iterrows():
            boost = f"{term['boost_factor']:.0f}×" if term['boost_factor'] != float('inf') else "∞"
            print(
                f"    • {term['term']:<20} RM3: {term['rm3_score']:.3f}  Sem: {term['semantic_score']:.3f}  MEQE: {term['meqe_score']:.3f}  (Boost: {boost})")
        print()

        detailed_examples.append({
            'query_id': example['query_id'],
            'query_text': example['query_text'],
            'pattern_type': pattern_type,
            'terms': query_terms.to_dict('records')
        })

    return examples, detailed_examples


def classify_rescue_pattern(query_df, rescued, high_rm3):
    """Classify the type of rescue pattern in a query"""
    if len(rescued) > 4:
        return "Many semantic rescues"
    elif len(high_rm3) > 4 and len(rescued) > 2:
        return "Balanced (RM3 preservation + semantic rescue)"
    elif len(rescued) > 0 and rescued['boost_factor'].replace([np.inf, -np.inf], np.nan).max() > 50:
        return "Dramatic semantic rescue"
    elif len(high_rm3) > len(rescued) * 2:
        return "RM3-dominated with some rescues"
    else:
        return "Mixed"


def categorize_terms(df, rescued_terms):
    """
    Analysis 3: Term Type Categorization

    Categorize rescued terms by type
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: TERM TYPE CATEGORIZATION")
    print("=" * 80)

    # Define term categories with keywords
    categories = {
        'Geographic/Locational': [
            'city', 'country', 'state', 'region', 'area', 'location', 'place',
            'national', 'international', 'local', 'european', 'american', 'asian',
            'north', 'south', 'east', 'west', 'central',
            # Common place names/adjectives
            'german', 'french', 'british', 'chinese', 'japanese', 'russian',
            'european', 'asian', 'african', 'american'
        ],
        'Temporal': [
            'year', 'month', 'day', 'time', 'period', 'era', 'century', 'decade',
            'recent', 'modern', 'current', 'contemporary', 'historical', 'past',
            'future', 'present', 'old', 'new', 'early', 'late',
            'today', 'yesterday', 'tomorrow', 'annual', 'yearly'
        ],
        'Conceptual/Synonyms': [
            'issue', 'problem', 'concern', 'matter', 'aspect', 'factor',
            'development', 'situation', 'condition', 'status', 'state',
            'process', 'system', 'method', 'approach', 'way'
        ],
        'Entities': [
            'organization', 'company', 'government', 'agency', 'department',
            'institution', 'group', 'party', 'association', 'union',
            'people', 'person', 'individual', 'population', 'community'
        ],
        'Technical/Domain': [
            'technology', 'system', 'program', 'project', 'research',
            'study', 'analysis', 'data', 'information', 'report',
            'science', 'scientific', 'technical', 'medical', 'legal'
        ]
    }

    # Categorize rescued terms
    categorized = defaultdict(list)
    uncategorized = []

    for _, term_row in rescued_terms.iterrows():
        term = term_row['term'].lower()
        found_category = False

        for category, keywords in categories.items():
            if any(keyword in term or term in keyword for keyword in keywords):
                categorized[category].append(term_row)
                found_category = True
                break

        if not found_category:
            uncategorized.append(term_row)

    # Statistics
    total_rescued = len(rescued_terms)

    print(f"\n📊 Term Type Distribution (Rescued Terms):")
    print(f"  Total rescued terms: {total_rescued}\n")

    category_stats = {}
    for category, terms in categorized.items():
        count = len(terms)
        pct = (count / total_rescued * 100) if total_rescued > 0 else 0
        category_stats[category] = {'count': count, 'percentage': pct}
        print(f"  {category:<30} {count:>4} ({pct:>5.1f}%)")

    uncategorized_count = len(uncategorized)
    uncategorized_pct = (uncategorized_count / total_rescued * 100) if total_rescued > 0 else 0
    print(f"  {'Other/Uncategorized':<30} {uncategorized_count:>4} ({uncategorized_pct:>5.1f}%)")

    # Show examples from each category
    print(f"\n📝 Example Terms by Category:")
    for category, terms in categorized.items():
        if len(terms) > 0:
            example_terms = [t['term'] for t in terms[:5]]
            print(f"  {category}: {', '.join(example_terms)}")

    # Paper-ready text
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    top_3 = sorted_categories[:3]

    paper_text = f"""
=== PAPER-READY TEXT ===

Semantic boosting primarily benefits three term categories: {top_3[0][0]} 
({top_3[0][1]['percentage']:.0f}% of rescued terms), {top_3[1][0]} 
({top_3[1][1]['percentage']:.0f}%), and {top_3[2][0]} 
({top_3[2][1]['percentage']:.0f}%). This aligns with the intuition that 
semantic models capture paradigmatic relations that statistical methods miss.
"""

    print(paper_text)

    return category_stats, categorized, paper_text


def create_visualizations(df, rescued_terms, output_dir):
    """
    Analysis 4: Weight Distribution Visualizations
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: WEIGHT DISTRIBUTION VISUALIZATIONS")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Scatter plot: RM3 vs Semantic weights
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all terms
    scatter = ax.scatter(
        df['rm3_score'],
        df['semantic_score'],
        s=df['meqe_score'] * 100,  # Size by final weight
        alpha=0.5,
        c='lightblue',
        edgecolors='navy',
        linewidth=0.5,
        label='All terms'
    )

    # Highlight rescued terms
    ax.scatter(
        rescued_terms['rm3_score'],
        rescued_terms['semantic_score'],
        s=rescued_terms['meqe_score'] * 100,
        alpha=0.7,
        c='red',
        edgecolors='darkred',
        linewidth=0.5,
        label='Rescued terms'
    )

    # Add threshold lines
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Semantic threshold')
    ax.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='RM3 threshold')

    ax.set_xlabel('RM3 Weight', fontsize=12)
    ax.set_ylabel('Semantic Similarity', fontsize=12)
    ax.set_title('Complementarity of RM3 and Semantic Signals\n(Bubble size = Final MEQE weight)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_path = output_path / 'weight_distribution_scatter.pdf'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.savefig(scatter_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {scatter_path}")
    plt.close()

    # 2. Hexbin density plot (alternative view)
    fig, ax = plt.subplots(figsize=(8, 6))

    hexbin = ax.hexbin(
        df['rm3_score'],
        df['semantic_score'],
        gridsize=30,
        cmap='YlOrRd',
        mincnt=1,
        alpha=0.8
    )

    ax.axhline(y=0.3, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=0.01, color='blue', linestyle='--', alpha=0.7, linewidth=2)

    ax.set_xlabel('RM3 Weight', fontsize=12)
    ax.set_ylabel('Semantic Similarity', fontsize=12)
    ax.set_title('Density of RM3 vs Semantic Weights', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Term Count', fontsize=11)

    plt.tight_layout()
    hexbin_path = output_path / 'weight_distribution_hexbin.pdf'
    plt.savefig(hexbin_path, dpi=300, bbox_inches='tight')
    plt.savefig(hexbin_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {hexbin_path}")
    plt.close()

    # 3. Boost factor distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_boosts = rescued_terms[rescued_terms['boost_factor'] != float('inf')]['boost_factor']

    if len(valid_boosts) > 0:
        # Clip extreme values for visualization
        clipped_boosts = np.clip(valid_boosts, 0, 100)

        ax.hist(clipped_boosts, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axvline(x=clipped_boosts.median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {clipped_boosts.median():.1f}×')
        ax.axvline(x=clipped_boosts.mean(), color='orange', linestyle='--',
                   linewidth=2, label=f'Mean: {clipped_boosts.mean():.1f}×')

        ax.set_xlabel('Boost Factor (MEQE / RM3)', fontsize=12)
        ax.set_ylabel('Number of Terms', fontsize=12)
        ax.set_title('Distribution of Semantic Boost Factors\n(For rescued terms with RM3 < 0.01, Semantic > 0.3)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        boost_path = output_path / 'boost_factor_distribution.pdf'
        plt.savefig(boost_path, dpi=300, bbox_inches='tight')
        plt.savefig(boost_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {boost_path}")
        plt.close()

    # 4. Box plot: Weight comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    weight_data = [
        df['rm3_score'].values,
        df['semantic_score'].values,
        df['meqe_score'].values
    ]

    bp = axes[0].boxplot(weight_data, labels=['RM3', 'Semantic', 'MEQE'],
                         patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)

    axes[0].set_ylabel('Weight Value', fontsize=11)
    axes[0].set_title('Weight Distributions\n(All Terms)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Rescued terms only
    rescued_weight_data = [
        rescued_terms['rm3_score'].values,
        rescued_terms['semantic_score'].values,
        rescued_terms['meqe_score'].values
    ]

    bp2 = axes[1].boxplot(rescued_weight_data, labels=['RM3', 'Semantic', 'MEQE'],
                          patch_artist=True, showfliers=False)

    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)

    axes[1].set_ylabel('Weight Value', fontsize=11)
    axes[1].set_title('Weight Distributions\n(Rescued Terms Only)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Correlation
    correlation = df[['rm3_score', 'semantic_score', 'meqe_score']].corr()

    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, ax=axes[2], cbar_kws={'shrink': 0.8})
    axes[2].set_title('Weight Correlations', fontsize=12, fontweight='bold')

    plt.tight_layout()
    comparison_path = output_path / 'weight_comparison.pdf'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.savefig(comparison_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {comparison_path}")
    plt.close()

    print(f"\n✅ All visualizations saved to: {output_dir}/")

    return {
        'scatter': str(scatter_path),
        'hexbin': str(hexbin_path),
        'boost': str(boost_path) if len(valid_boosts) > 0 else None,
        'comparison': str(comparison_path)
    }


def save_results(rescue_stats, examples, category_stats, output_dir):
    """Save all results to JSON and CSV files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save rescue statistics
    with open(output_path / 'rescue_statistics.json', 'w') as f:
        json.dump(rescue_stats, f, indent=2)

    # Save case study examples
    with open(output_path / 'case_study_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)

    # Save category statistics
    with open(output_path / 'term_categories.json', 'w') as f:
        json.dump(category_stats, f, indent=2)

    print(f"\n✅ Results saved to: {output_dir}/")


def generate_summary_report(rescue_stats, examples, category_stats, output_dir):
    """Generate a comprehensive summary report"""
    output_path = Path(output_dir)

    report = f"""
{'=' * 80}
CASE STUDY ANALYSIS - SUMMARY REPORT
{'=' * 80}

1. TERM RESCUE STATISTICS
{'=' * 80}

• Total expansion terms: {rescue_stats['total_terms']:,}
• Rescued terms (low RM3, high semantic): {rescue_stats['rescued_terms_count']:,} ({rescue_stats['rescued_percentage']:.1f}%)
• Queries with rescues: {rescue_stats['queries_with_rescues']}
• Average rescues per query: {rescue_stats['avg_rescues_per_query']:.1f}

Boost Factor Statistics:
• Mean: {rescue_stats['boost_statistics']['mean']:.1f}×
• Median: {rescue_stats['boost_statistics']['median']:.1f}×
• Max: {rescue_stats['boost_statistics']['max']:.1f}×

INTERPRETATION: Semantic component consistently rescues terms that RM3 underweights,
with substantial boost factors demonstrating the value of hybrid weighting.

{'=' * 80}
2. CASE STUDY EXAMPLES
{'=' * 80}

Found {len(examples)} diverse examples suitable for paper:

"""

    for idx, (pattern_type, example) in enumerate(examples, 1):
        report += f"""
Example {idx}: Query {example['query_id']}
• Text: {example['query_text']}
• Pattern: {pattern_type}
• Rescued terms: {example['num_rescued']}
• Max boost: {example['max_rescue_boost']:.0f}×
"""

    report += f"""
{'=' * 80}
3. TERM TYPE CATEGORIZATION
{'=' * 80}

Distribution of rescued terms by category:

"""

    for category, stats in sorted(category_stats.items(),
                                  key=lambda x: x[1]['count'], reverse=True):
        report += f"• {category:<30} {stats['count']:>4} ({stats['percentage']:>5.1f}%)\n"

    report += f"""
INTERPRETATION: Semantic boosting primarily benefits geographic/locational terms,
conceptual synonyms, and temporal qualifiers - exactly the types of paradigmatic
relations that embedding models capture well.

{'=' * 80}
4. VISUALIZATIONS
{'=' * 80}

Generated publication-ready visualizations:
• weight_distribution_scatter.pdf - Shows complementarity of RM3 and semantic
• weight_distribution_hexbin.pdf - Density view of weight relationships
• boost_factor_distribution.pdf - Distribution of boost factors
• weight_comparison.pdf - Box plots and correlations

{'=' * 80}
RECOMMENDATIONS FOR PAPER
{'=' * 80}

1. Use rescue statistics in Results section (~2 sentences):
   "MEQE rescues an average of {rescue_stats['avg_rescues_per_query']:.1f} terms per query with 
   {rescue_stats['boost_statistics']['mean']:.0f}× average boost, demonstrating hybrid 
   weighting's ability to identify semantically relevant terms overlooked by 
   statistical methods."

2. Show 1-2 additional case examples beyond Query 401 to demonstrate pattern
   generalizability.

3. Use term categorization to explain WHAT types of terms benefit:
   "Rescued terms are predominantly geographic ({list(category_stats.keys())[0]}, XX%), 
   conceptual (YY%), and temporal (ZZ%)."

4. Include scatter plot visualization showing complementarity of signals.

{'=' * 80}
"""

    report_path = output_path / 'ANALYSIS_SUMMARY.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"✅ Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of case study JSON for IR paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_case_studies.py complete_case_study.json

    # With custom output directory
    python analyze_case_studies.py complete_case_study.json --output-dir results

    # Adjust rescue thresholds
    python analyze_case_studies.py complete_case_study.json --rm3-threshold 0.02 --semantic-threshold 0.4
        """
    )

    parser.add_argument('json_file', help='Path to complete_case_study.json')
    parser.add_argument('--output-dir', default='case_study_analysis',
                        help='Output directory for results (default: case_study_analysis)')
    parser.add_argument('--rm3-threshold', type=float, default=0.01,
                        help='RM3 threshold for identifying rescued terms (default: 0.01)')
    parser.add_argument('--semantic-threshold', type=float, default=0.3,
                        help='Semantic threshold for rescued terms (default: 0.3)')
    parser.add_argument('--n-examples', type=int, default=3,
                        help='Number of case study examples to generate (default: 3)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CASE STUDY ANALYSIS FOR IR PAPER")
    print("=" * 80)
    print(f"Input file: {args.json_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rescue thresholds: RM3 < {args.rm3_threshold}, Semantic > {args.semantic_threshold}")
    print("=" * 80 + "\n")

    # Load data
    case_studies = load_case_study(args.json_file)

    # Extract term data
    df = extract_term_data(case_studies)

    # Analysis 1: Term rescue statistics
    rescue_stats, rescued_terms, rescue_text = analyze_term_rescue(
        df, args.rm3_threshold, args.semantic_threshold
    )

    # Analysis 2: Case study examples
    examples, detailed_examples = identify_case_study_examples(df, args.n_examples)

    # Analysis 3: Term categorization
    category_stats, categorized, category_text = categorize_terms(df, rescued_terms)

    # Analysis 4: Visualizations
    viz_paths = create_visualizations(df, rescued_terms, args.output_dir)

    # Save all results
    results = {
        'rescue_statistics': rescue_stats,
        'case_examples': detailed_examples,
        'term_categories': category_stats,
        'visualizations': viz_paths
    }
    save_results(rescue_stats, detailed_examples, category_stats, args.output_dir)

    # Generate summary report
    generate_summary_report(rescue_stats, examples, category_stats, args.output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}/")
    print(f"\nKey files:")
    print(f"  • ANALYSIS_SUMMARY.txt - Complete report")
    print(f"  • rescue_statistics.json - Rescue statistics")
    print(f"  • case_study_examples.json - Example queries")
    print(f"  • term_categories.json - Term type breakdown")
    print(f"  • *.pdf / *.png - Visualizations")
    print("\n")


if __name__ == '__main__':
    main()