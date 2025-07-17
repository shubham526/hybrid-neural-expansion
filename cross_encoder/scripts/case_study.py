#!/usr/bin/env python3
"""
MEQE Case Study: Hybrid Re-weighting Analysis

This script analyzes pre-extracted features to show how MEQE intelligently
re-weights RM3 candidate terms using the hybrid approach.

Uses:
1. Pre-extracted features from create_features.py output
2. Trained model weights from checkpoint
3. Hybrid weighting formula: α * RM3_weight + β * semantic_score
"""

import argparse
import logging
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from cross_encoder.src.models.reranker2 import create_neural_reranker
from cross_encoder.src.utils.file_utils import ensure_dir, save_json, load_jsonl
from cross_encoder.src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

logger = logging.getLogger(__name__)


class MEQECaseStudyAnalyzer:
    """
    Analyzer for generating MEQE case study data using pre-extracted features.
    """

    def __init__(self,
                 features_file: str,
                 output_dir: Path,
                 model_checkpoint: str = None,
                 semantic_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the case study analyzer.

        Args:
            features_file: Path to pre-extracted features JSONL file
            output_dir: Output directory for results
            model_checkpoint: Path to trained MEQE model (optional)
            semantic_model: Semantic similarity model name (for model loading)
        """
        self.features_file = Path(features_file)
        self.output_dir = ensure_dir(output_dir)
        self.model_checkpoint = model_checkpoint

        # Load pre-extracted features
        logger.info(f"Loading pre-extracted features from: {features_file}")
        self.features_data = self._load_features()
        logger.info(f"Loaded features for {len(self.features_data)} queries")

        # Load trained model weights if provided
        self.learned_alpha = 0.5  # Default values
        self.learned_beta = 0.5
        self.learned_lambda = 0.3

        if model_checkpoint and Path(model_checkpoint).exists():
            logger.info(f"Loading trained model weights from: {model_checkpoint}")
            self._load_model_weights(model_checkpoint, semantic_model)
        else:
            logger.warning("No model checkpoint provided - using default weights")
            logger.warning("For accurate results, provide a trained model checkpoint!")

    def _load_features(self) -> Dict[str, Dict]:
        """Load features from JSONL file and index by query_id."""
        features_list = load_jsonl(self.features_file)
        features_dict = {}

        for feature_entry in features_list:
            query_id = feature_entry['query_id']
            features_dict[query_id] = feature_entry

        return features_dict

    def _load_model_weights(self, checkpoint_path: str, semantic_model: str):
        """Load learned weights from trained model checkpoint."""
        try:
            # Create model architecture (needed for loading weights)
            # model = create_neural_reranker(
            #     model_name=semantic_model,
            #     scoring_method='neural',
            #     device='cpu'
            # )
            model = create_neural_reranker(
                model_name='all-MiniLM-L6-v2',
                max_expansion_terms=15,
                hidden_dim=384,
                dropout=0.1,
                scoring_method='cosine',
                device='cpu'
            )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint)

            # Extract learned weights
            self.learned_alpha, self.learned_beta, self.learned_lambda = model.get_learned_weights()

            logger.info(
                f"Loaded weights: α={self.learned_alpha:.4f}, β={self.learned_beta:.4f}, λ={self.learned_lambda:.4f}")

        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.warning("Using default weights instead")

    def analyze_query_reweighting(self, query_id: str) -> Dict[str, Any]:
        """
        Analyze how MEQE re-weights expansion terms for a specific query.

        Args:
            query_id: Query ID to analyze

        Returns:
            Dictionary with analysis results
        """
        if query_id not in self.features_data:
            logger.error(f"Query {query_id} not found in features data")
            return {}

        query_data = self.features_data[query_id]
        query_text = query_data['query_text']
        term_features = query_data['term_features']

        logger.info(f"Analyzing query re-weighting for: '{query_text}' (ID: {query_id})")

        # Analyze each expansion term
        term_analysis = []

        for term, term_data in term_features.items():
            rm_weight = term_data['rm_weight']
            semantic_score = term_data['semantic_score']
            original_term = term_data.get('original_term', term)

            # Compute MEQE hybrid importance using learned weights
            meqe_importance = (self.learned_alpha * rm_weight +
                               self.learned_beta * semantic_score)

            # Categorize scores for analysis
            rm_category = self._categorize_score(rm_weight, 'rm3')
            semantic_category = self._categorize_score(semantic_score, 'semantic')
            meqe_category = self._categorize_score(meqe_importance, 'meqe')

            term_analysis.append({
                'term': term,
                'original_term': original_term,
                'rm_weight': rm_weight,
                'semantic_score': semantic_score,
                'meqe_importance': meqe_importance,
                'rm_category': rm_category,
                'semantic_category': semantic_category,
                'meqe_category': meqe_category,
                'importance_ratio': meqe_importance / max(rm_weight, 1e-6),  # How much MEQE changes the weight
                'semantic_boost': semantic_score > rm_weight,  # Whether semantic score helps
                'rm_vs_semantic_diff': semantic_score - rm_weight
            })

        # Sort by MEQE importance (descending)
        term_analysis.sort(key=lambda x: x['meqe_importance'], reverse=True)

        # Generate insights
        insights = self._generate_insights(term_analysis)

        analysis_result = {
            'query': query_text,
            'query_id': query_id,
            'learned_weights': {
                'alpha': self.learned_alpha,
                'beta': self.learned_beta,
                'lambda': self.learned_lambda
            },
            'num_terms': len(term_analysis),
            'term_analysis': term_analysis,
            'insights': insights,
            'statistics': self._compute_statistics(term_analysis)
        }

        return analysis_result

    def _categorize_score(self, score: float, score_type: str) -> str:
        """Categorize scores into human-readable categories."""
        if score_type == 'rm3':
            # RM3 weights are typically small, adjust thresholds accordingly
            if score >= 0.1:
                return "Very High"
            elif score >= 0.05:
                return "High"
            elif score >= 0.02:
                return "Medium"
            elif score >= 0.01:
                return "Low"
            else:
                return "Very Low"

        elif score_type == 'semantic':
            # Cosine similarity scores range from 0-1
            if score >= 0.8:
                return "Very High"
            elif score >= 0.6:
                return "High"
            elif score >= 0.4:
                return "Medium"
            elif score >= 0.2:
                return "Low"
            else:
                return "Very Low"

        elif score_type == 'meqe':
            # MEQE importance is combination, adjust based on typical ranges
            if score >= 0.15:
                return "Very High"
            elif score >= 0.08:
                return "High"
            elif score >= 0.04:
                return "Medium"
            elif score >= 0.02:
                return "Low"
            else:
                return "Very Low"

        return "Unknown"

    def _generate_insights(self, term_analysis: List[Dict]) -> Dict[str, Any]:
        """Generate insights about the re-weighting behavior."""
        insights = {
            'top_terms_by_meqe': term_analysis[:5],
            'terms_boosted_by_semantic': [],
            'terms_penalized_by_semantic': [],
            'high_rm3_low_semantic': [],
            'high_semantic_low_rm3': [],
            'most_reweighted': []
        }

        for term_data in term_analysis:
            # Terms significantly boosted by semantic similarity
            if term_data['semantic_boost'] and term_data['rm_vs_semantic_diff'] > 0.1:
                insights['terms_boosted_by_semantic'].append(term_data)

            # Terms penalized by semantic similarity
            if not term_data['semantic_boost'] and term_data['rm_vs_semantic_diff'] < -0.1:
                insights['terms_penalized_by_semantic'].append(term_data)

            # High RM3 but low semantic (potentially noisy)
            if (term_data['rm_category'] in ['High', 'Very High'] and
                    term_data['semantic_category'] in ['Low', 'Very Low']):
                insights['high_rm3_low_semantic'].append(term_data)

            # High semantic but low RM3 (potentially missed by statistical methods)
            if (term_data['semantic_category'] in ['High', 'Very High'] and
                    term_data['rm_category'] in ['Low', 'Very Low']):
                insights['high_semantic_low_rm3'].append(term_data)

            # Most re-weighted terms (biggest change from RM3 to MEQE)
            if abs(term_data['importance_ratio'] - 1.0) > 0.5:
                insights['most_reweighted'].append(term_data)

        # Sort insights by relevance
        for key in insights:
            if isinstance(insights[key], list) and insights[key]:
                if key == 'top_terms_by_meqe':
                    continue  # Already sorted
                elif key in ['terms_boosted_by_semantic', 'high_semantic_low_rm3']:
                    insights[key].sort(key=lambda x: x['semantic_score'], reverse=True)
                elif key in ['terms_penalized_by_semantic', 'high_rm3_low_semantic']:
                    insights[key].sort(key=lambda x: x['rm_weight'], reverse=True)
                elif key == 'most_reweighted':
                    insights[key].sort(key=lambda x: abs(x['importance_ratio'] - 1.0), reverse=True)

        return insights

    def _compute_statistics(self, term_analysis: List[Dict]) -> Dict[str, float]:
        """Compute statistical summaries of the re-weighting."""
        if not term_analysis:
            return {}

        rm_weights = [t['rm_weight'] for t in term_analysis]
        semantic_scores = [t['semantic_score'] for t in term_analysis]
        meqe_scores = [t['meqe_importance'] for t in term_analysis]
        importance_ratios = [t['importance_ratio'] for t in term_analysis]

        import numpy as np

        return {
            'rm_weight_mean': np.mean(rm_weights),
            'rm_weight_std': np.std(rm_weights),
            'semantic_score_mean': np.mean(semantic_scores),
            'semantic_score_std': np.std(semantic_scores),
            'meqe_importance_mean': np.mean(meqe_scores),
            'meqe_importance_std': np.std(meqe_scores),
            'importance_ratio_mean': np.mean(importance_ratios),
            'importance_ratio_std': np.std(importance_ratios),
            'correlation_rm_semantic': np.corrcoef(rm_weights, semantic_scores)[0, 1],
            'terms_with_semantic_boost': sum(1 for t in term_analysis if t['semantic_boost']),
            'terms_with_semantic_penalty': sum(1 for t in term_analysis if not t['semantic_boost'])
        }

    def find_query_by_text(self, query_text: str) -> str:
        """Find query ID by searching for query text."""
        for query_id, data in self.features_data.items():
            if query_text.lower() in data['query_text'].lower():
                return query_id
        return None

    def list_available_queries(self, limit: int = 20) -> List[Tuple[str, str]]:
        """List available queries in the features file."""
        queries = []
        for query_id, data in list(self.features_data.items())[:limit]:
            queries.append((query_id, data['query_text']))
        return queries

    def create_case_study_table(self,
                                analysis_result: Dict[str, Any],
                                top_n: int = 10) -> pd.DataFrame:
        """
        Create a formatted table suitable for the paper.

        Args:
            analysis_result: Result from analyze_query_reweighting
            top_n: Number of top terms to include

        Returns:
            Pandas DataFrame with formatted results
        """
        term_analysis = analysis_result['term_analysis'][:top_n]

        table_data = []
        for term_data in term_analysis:
            table_data.append({
                'Candidate Term': term_data['original_term'],
                'RM3 Score': term_data['rm_category'],
                'Semantic Score (cos sim)': term_data['semantic_category'],
                'Final MEQE Importance': term_data['meqe_category'],
                'RM3 Weight (raw)': f"{term_data['rm_weight']:.4f}",
                'Semantic Weight (raw)': f"{term_data['semantic_score']:.4f}",
                'MEQE Weight (raw)': f"{term_data['meqe_importance']:.4f}",
                'Importance Ratio': f"{term_data['importance_ratio']:.2f}x"
            })

        df = pd.DataFrame(table_data)
        return df

    def generate_latex_table(self,
                             analysis_result: Dict[str, Any],
                             top_n: int = 8,
                             table_style: str = "paper") -> str:
        """
        Generate LaTeX table code suitable for the paper.

        Args:
            analysis_result: Result from analyze_query_reweighting
            top_n: Number of terms to include
            table_style: Style of table ("paper" or "detailed")

        Returns:
            LaTeX table code as string
        """
        term_analysis = analysis_result['term_analysis'][:top_n]
        query = analysis_result['query']

        if table_style == "paper":
            # Simplified table for paper
            latex_code = "\\begin{table}[h!]\n"
            latex_code += "\\centering\n"
            latex_code += f"\\caption{{Illustration of MEQE's re-weighting for the query \"{query}.\" All terms are sourced from RM3, but MEQE assigns higher importance to semantically aligned terms.}}\n"
            latex_code += "\\label{tab:case-study}\n"
            latex_code += "\\begin{tabular}{lccc}\n"
            latex_code += "\\hline\n"
            latex_code += "\\textbf{Candidate Term} & \\textbf{RM3 Score} & \\textbf{Semantic Score (cos sim)} & \\textbf{Final MEQE Importance} \\\\ \\hline\n"

            for term_data in term_analysis:
                term = term_data['original_term']
                rm_cat = term_data['rm_category']
                sem_cat = term_data['semantic_category']
                meqe_cat = term_data['meqe_category']

                # Format term name (bold for top terms)
                if term_data['meqe_category'] in ['Very High', 'High']:
                    term_formatted = f"\\textbf{{{term}}}"
                else:
                    term_formatted = term

                # Format final importance (bold for high importance)
                if meqe_cat in ['Very High', 'High']:
                    meqe_formatted = f"\\textbf{{{meqe_cat}}}"
                else:
                    meqe_formatted = meqe_cat

                latex_code += f"{term_formatted} & {rm_cat} & {sem_cat} & {meqe_formatted} \\\\\n"

            latex_code += "\\hline\n"
            latex_code += "\\end{tabular}\n"
            latex_code += "\\end{table}\n"

        else:  # detailed table
            latex_code = "\\begin{table}[h!]\n"
            latex_code += "\\centering\n"
            latex_code += "\\caption{Detailed MEQE re-weighting analysis}\n"
            latex_code += "\\begin{tabular}{lcccccc}\n"
            latex_code += "\\hline\n"
            latex_code += "\\textbf{Term} & \\textbf{RM3} & \\textbf{Semantic} & \\textbf{MEQE} & \\textbf{RM3 Cat} & \\textbf{Sem Cat} & \\textbf{MEQE Cat} \\\\ \\hline\n"

            for term_data in term_analysis:
                latex_code += f"{term_data['original_term']} & "
                latex_code += f"{term_data['rm_weight']:.4f} & "
                latex_code += f"{term_data['semantic_score']:.4f} & "
                latex_code += f"{term_data['meqe_importance']:.4f} & "
                latex_code += f"{term_data['rm_category']} & "
                latex_code += f"{term_data['semantic_category']} & "
                latex_code += f"{term_data['meqe_category']} \\\\\n"

            latex_code += "\\hline\n"
            latex_code += "\\end{tabular}\n"
            latex_code += "\\end{table}\n"

        return latex_code

    def create_visualizations(self,
                              analysis_result: Dict[str, Any],
                              save_plots: bool = True) -> Dict[str, Any]:
        """
        Create visualizations showing the re-weighting effects.

        Args:
            analysis_result: Result from analyze_query_reweighting
            save_plots: Whether to save plot files

        Returns:
            Dictionary with plot information
        """
        term_analysis = analysis_result['term_analysis']
        query = analysis_result['query']
        query_id = analysis_result['query_id']

        # Prepare data
        terms = [t['original_term'] for t in term_analysis[:15]]  # Top 15 for readability
        rm_weights = [t['rm_weight'] for t in term_analysis[:15]]
        semantic_scores = [t['semantic_score'] for t in term_analysis[:15]]
        meqe_scores = [t['meqe_importance'] for t in term_analysis[:15]]

        plots_created = {}

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("colorblind")

        # Plot 1: Comparison of RM3 vs MEQE weights
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))

        x = range(len(terms))
        width = 0.35

        bars1 = ax1.bar([i - width / 2 for i in x], rm_weights, width,
                        label='RM3 Weight', alpha=0.8, color='skyblue')
        bars2 = ax1.bar([i + width / 2 for i in x], meqe_scores, width,
                        label='MEQE Importance', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('Expansion Terms')
        ax1.set_ylabel('Weight/Importance Score')
        ax1.set_title(f'RM3 vs MEQE Re-weighting for Query: "{query}"')
        ax1.set_xticks(x)
        ax1.set_xticklabels(terms, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plots:
            plot1_path = self.output_dir / f"rm3_vs_meqe_comparison_{query_id}.png"
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plots_created['rm3_vs_meqe'] = str(plot1_path)
        plt.close()

        # Plot 2: Scatter plot showing relationship between RM3 and semantic scores
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

        all_rm = [t['rm_weight'] for t in term_analysis]
        all_semantic = [t['semantic_score'] for t in term_analysis]
        all_meqe = [t['meqe_importance'] for t in term_analysis]

        scatter = ax2.scatter(all_rm, all_semantic, c=all_meqe, cmap='viridis',
                              s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax2.set_xlabel('RM3 Weight')
        ax2.set_ylabel('Semantic Similarity Score')
        ax2.set_title(f'RM3 vs Semantic Scores (colored by MEQE importance)\nQuery: "{query}"')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('MEQE Importance')

        # Add annotations for top terms
        for i, (rm, sem, term) in enumerate(zip(all_rm[:5], all_semantic[:5],
                                                [t['original_term'] for t in term_analysis[:5]])):
            ax2.annotate(term, (rm, sem), xytext=(5, 5), textcoords='offset points',
                         fontsize=9, alpha=0.8)

        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            plot2_path = self.output_dir / f"rm3_vs_semantic_scatter_{query_id}.png"
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plots_created['scatter_plot'] = str(plot2_path)
        plt.close()

        return plots_created

    def run_case_study(self,
                       query_ids: List[str] = None,
                       query_text_search: str = None) -> Dict[str, Any]:
        """
        Run complete case study analysis for specified queries.

        Args:
            query_ids: List of query IDs to analyze
            query_text_search: Search for queries containing this text

        Returns:
            Complete case study results
        """
        if query_ids is None and query_text_search is None:
            # Default behavior - analyze first 5 queries
            query_ids = list(self.features_data.keys())[:5]
            logger.info(f"No queries specified - analyzing first {len(query_ids)} queries")

        elif query_text_search:
            # Search for queries containing the text
            found_queries = []
            for query_id, data in self.features_data.items():
                if query_text_search.lower() in data['query_text'].lower():
                    found_queries.append(query_id)

            if not found_queries:
                logger.error(f"No queries found containing text: '{query_text_search}'")
                return {}

            query_ids = found_queries
            logger.info(f"Found {len(query_ids)} queries containing '{query_text_search}'")

        logger.info(f"Running case study for {len(query_ids)} queries")

        all_results = {}

        for query_id in query_ids:
            if query_id not in self.features_data:
                logger.warning(f"Query ID {query_id} not found in features data")
                continue

            query_text = self.features_data[query_id]['query_text']
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing Query: {query_id} - '{query_text}'")
            logger.info(f"{'=' * 60}")

            # Analyze this query
            analysis = self.analyze_query_reweighting(query_id)

            if analysis:
                # Create visualizations
                plots = self.create_visualizations(analysis, save_plots=True)
                analysis['plots'] = plots

                # Generate LaTeX table
                latex_table = self.generate_latex_table(analysis, top_n=8, table_style="paper")
                analysis['latex_table'] = latex_table

                # Create pandas table
                df_table = self.create_case_study_table(analysis, top_n=10)

                # Store analysis without non-serializable objects
                analysis_for_json = {k: v for k, v in analysis.items()
                                     if k not in ['pandas_table']}  # Exclude DataFrame
                all_results[query_id] = analysis_for_json

                # Save individual analysis (without DataFrame)
                query_file = self.output_dir / f"analysis_{query_id}.json"
                save_json(analysis_for_json, query_file)

                # Save LaTeX table to file
                latex_file = self.output_dir / f"latex_table_{query_id}.tex"
                with open(latex_file, 'w') as f:
                    f.write(latex_table)

                # Save pandas table to CSV
                csv_file = self.output_dir / f"table_{query_id}.csv"
                df_table.to_csv(csv_file, index=False)

                logger.info(f"✓ Analysis complete for '{query_text}'")
                logger.info(f"  - JSON: {query_file}")
                logger.info(f"  - LaTeX: {latex_file}")
                logger.info(f"  - CSV: {csv_file}")
                logger.info(f"  - Plots: {len(plots)} created")
            else:
                logger.warning(f"⚠ Analysis failed for query {query_id}")

        # Save complete results
        complete_results = {
            'case_study_config': {
                'num_queries': len(query_ids),
                'queries_analyzed': list(all_results.keys()),
                'learned_weights': {
                    'alpha': self.learned_alpha,
                    'beta': self.learned_beta,
                    'lambda': self.learned_lambda
                },
                'features_source': str(self.features_file)
            },
            'individual_analyses': all_results,
            'summary_statistics': self._compute_cross_query_statistics(all_results)
        }

        # Save complete results
        complete_file = self.output_dir / "complete_case_study.json"
        save_json(complete_results, complete_file)

        # Create summary report
        self._create_summary_report(complete_results)

        logger.info(f"\n{'=' * 60}")
        logger.info("CASE STUDY COMPLETE!")
        logger.info(f"{'=' * 60}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Complete analysis: {complete_file}")
        logger.info(f"Queries analyzed: {len(all_results)}")
        logger.info(f"{'=' * 60}")

        return complete_results

    def _compute_cross_query_statistics(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics across all analyzed queries."""
        if not all_results:
            return {}

        all_importance_ratios = []
        all_semantic_boosts = []
        terms_by_category = defaultdict(int)

        for query_id, analysis in all_results.items():
            for term_data in analysis['term_analysis']:
                all_importance_ratios.append(term_data['importance_ratio'])
                all_semantic_boosts.append(term_data['semantic_boost'])
                terms_by_category[term_data['meqe_category']] += 1

        import numpy as np

        return {
            'total_terms_analyzed': len(all_importance_ratios),
            'avg_importance_ratio': np.mean(all_importance_ratios),
            'semantic_boost_rate': np.mean(all_semantic_boosts),
            'terms_by_meqe_category': dict(terms_by_category),
            'importance_ratio_stats': {
                'mean': np.mean(all_importance_ratios),
                'std': np.std(all_importance_ratios),
                'min': np.min(all_importance_ratios),
                'max': np.max(all_importance_ratios)
            }
        }

    def _create_summary_report(self, complete_results: Dict[str, Any]):
        """Create a human-readable summary report."""
        report_file = self.output_dir / "case_study_summary.md"

        with open(report_file, 'w') as f:
            f.write("# MEQE Case Study: Hybrid Re-weighting Analysis\n\n")

            # Configuration
            config = complete_results['case_study_config']
            f.write("## Configuration\n\n")
            f.write(f"- **Queries Analyzed**: {config['num_queries']}\n")
            f.write(f"- **Features Source**: {config['features_source']}\n")
            f.write(
                f"- **Learned Weights**: α={config['learned_weights']['alpha']:.3f}, β={config['learned_weights']['beta']:.3f}, λ={config['learned_weights']['lambda']:.3f}\n\n")

            # Summary statistics
            if 'summary_statistics' in complete_results:
                stats = complete_results['summary_statistics']
                f.write("## Summary Statistics\n\n")
                f.write(f"- **Total Terms Analyzed**: {stats['total_terms_analyzed']}\n")
                f.write(f"- **Average Importance Ratio**: {stats['avg_importance_ratio']:.2f}x\n")
                f.write(f"- **Semantic Boost Rate**: {stats['semantic_boost_rate']:.1%}\n\n")

                f.write("### Terms by MEQE Category\n\n")
                for category, count in stats['terms_by_meqe_category'].items():
                    f.write(f"- **{category}**: {count} terms\n")
                f.write("\n")

            # Individual query results
            f.write("## Individual Query Analysis\n\n")
            for query_id, analysis in complete_results['individual_analyses'].items():
                f.write(f"### Query {query_id}: {analysis['query']}\n\n")

                # Top terms
                f.write("**Top MEQE Terms**:\n")
                for i, term_data in enumerate(analysis['term_analysis'][:5], 1):
                    f.write(f"{i}. **{term_data['original_term']}** - ")
                    f.write(f"RM3: {term_data['rm_category']}, ")
                    f.write(f"Semantic: {term_data['semantic_category']}, ")
                    f.write(f"MEQE: {term_data['meqe_category']}\n")
                f.write("\n")

                # Key insights
                insights = analysis['insights']
                if insights['high_rm3_low_semantic']:
                    f.write("**Terms Penalized by Semantic (High RM3, Low Semantic)**:\n")
                    for term_data in insights['high_rm3_low_semantic'][:3]:
                        f.write(
                            f"- {term_data['original_term']} (RM3: {term_data['rm_category']}, Semantic: {term_data['semantic_category']})\n")
                    f.write("\n")

                if insights['high_semantic_low_rm3']:
                    f.write("**Terms Boosted by Semantic (Low RM3, High Semantic)**:\n")
                    for term_data in insights['high_semantic_low_rm3'][:3]:
                        f.write(
                            f"- {term_data['original_term']} (RM3: {term_data['rm_category']}, Semantic: {term_data['semantic_category']})\n")
                    f.write("\n")

                f.write("---\n\n")

        logger.info(f"Summary report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MEQE case study using pre-extracted features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--features-file', type=str, required=True,
                        help='Path to pre-extracted features JSONL file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for case study results')

    # Optional arguments
    parser.add_argument('--model-checkpoint', type=str,
                        help='Path to trained MEQE model checkpoint')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                        help='Semantic similarity model name (for model loading)')

    # Query specification
    parser.add_argument('--query-ids', nargs='+',
                        help='Specific query IDs to analyze')
    parser.add_argument('--query-search', type=str,
                        help='Search for queries containing this text (e.g., "Edison rival")')
    parser.add_argument('--list-queries', action='store_true',
                        help='List available queries and exit')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Setup logging
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("meqe_case_study", args.log_level,
                                      str(output_dir / 'case_study.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # Initialize analyzer
        analyzer = MEQECaseStudyAnalyzer(
            features_file=args.features_file,
            output_dir=output_dir,
            model_checkpoint=args.model_checkpoint,
            semantic_model=args.semantic_model
        )

        # List queries if requested
        if args.list_queries:
            print("\nAvailable Queries (first 20):")
            print("=" * 60)
            queries = analyzer.list_available_queries(20)
            for query_id, query_text in queries:
                print(f"{query_id:<10} {query_text}")
            print(f"\nTotal queries available: {len(analyzer.features_data)}")
            return

        # Run case study
        if args.query_search:
            # Search for queries containing specific text
            results = analyzer.run_case_study(query_text_search=args.query_search)
        elif args.query_ids:
            # Analyze specific query IDs
            results = analyzer.run_case_study(query_ids=args.query_ids)
        else:
            # Default behavior - analyze first few queries
            logger.info("No specific queries provided - analyzing first 3 queries")
            results = analyzer.run_case_study(query_ids=list(analyzer.features_data.keys())[:3])

        if not results or not results.get('individual_analyses'):
            logger.error("No results generated - check your query specifications")
            return

        # Print summary for the main results
        print("\n" + "=" * 60)
        print("CASE STUDY RESULTS FOR PAPER")
        print("=" * 60)

        config = results['case_study_config']
        print(f"Learned Weights: α={config['learned_weights']['alpha']:.3f}, "
              f"β={config['learned_weights']['beta']:.3f}, "
              f"λ={config['learned_weights']['lambda']:.3f}")
        print(f"Queries Analyzed: {len(results['individual_analyses'])}")

        # Show details for each analyzed query
        for query_id, analysis in results['individual_analyses'].items():
            print(f"\n--- Query {query_id}: {analysis['query']} ---")
            print("Top 5 Terms by MEQE Importance:")
            print("-" * 40)

            for i, term_data in enumerate(analysis['term_analysis'][:5], 1):
                print(f"{i}. {term_data['original_term']:<20} "
                      f"RM3: {term_data['rm_category']:<10} "
                      f"Semantic: {term_data['semantic_category']:<10} "
                      f"MEQE: {term_data['meqe_category']}")

            print(f"\nLaTeX table: {output_dir}/latex_table_{query_id}.tex")
            print(f"Analysis JSON: {output_dir}/analysis_{query_id}.json")

            # Show key insights
            insights = analysis['insights']
            if insights['high_rm3_low_semantic']:
                print(f"\nTerms penalized by semantic similarity:")
                for term_data in insights['high_rm3_low_semantic'][:2]:
                    print(f"  - {term_data['original_term']} (high RM3, low semantic)")

            if insights['high_semantic_low_rm3']:
                print(f"\nTerms boosted by semantic similarity:")
                for term_data in insights['high_semantic_low_rm3'][:2]:
                    print(f"  - {term_data['original_term']} (low RM3, high semantic)")

        print(f"\nComplete results: {output_dir}/complete_case_study.json")
        print(f"Summary report: {output_dir}/case_study_summary.md")
        print("=" * 60)

        print(f"\nCase study complete! Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Case study failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()