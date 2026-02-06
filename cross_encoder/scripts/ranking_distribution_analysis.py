import pandas as pd
from collections import defaultdict
import argparse
import numpy as np


def load_trec_run(run_file_path: str) -> dict:
    """
    Loads a TREC-style run file into a dictionary.

    Args:
        run_file_path: Path to the TREC run file.

    Returns:
        A dictionary mapping query IDs to a list of ranked document IDs with scores.
    """
    run_data = defaultdict(list)
    print(f"Loading run file from: {run_file_path}")
    with open(run_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, _ = parts
                run_data[query_id].append((doc_id, int(rank), float(score)))
            elif len(parts) >= 4:
                query_id, _, doc_id, rank = parts[:4]
                score = float(parts[4]) if len(parts) > 4 else 0.0
                run_data[query_id].append((doc_id, int(rank), score))

    # Sort by rank to ensure proper ordering
    for query_id in run_data:
        run_data[query_id].sort(key=lambda x: x[1])  # Sort by rank

    print(f"Loaded rankings for {len(run_data)} queries.")
    return dict(run_data)


def load_qrels(qrels_file_path: str) -> dict:
    """
    Loads a qrels file into a dictionary.

    Args:
        qrels_file_path: Path to the qrels file.

    Returns:
        A dictionary mapping query IDs to another dictionary of {doc_id: relevance_grade}.
    """
    qrels_data = defaultdict(dict)
    print(f"Loading qrels from: {qrels_file_path}")
    with open(qrels_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id, _, doc_id, grade = parts[:4]
                qrels_data[query_id][doc_id] = int(grade)
    print(f"Loaded relevance judgments for {len(qrels_data)} queries.")
    return dict(qrels_data)


def calculate_average_ranks(run_data: dict, qrels_data: dict) -> dict:
    """
    Calculate average ranks for different relevance grades.

    Returns:
        Dictionary with average rank statistics.
    """
    grade_ranks = {
        1: [],  # Relevant (Grade 1)
        2: [],  # Highly Relevant (Grade 2)
        'all': []  # All relevant
    }

    for query_id, ranked_docs in run_data.items():
        if query_id not in qrels_data:
            continue

        query_qrels = qrels_data[query_id]

        for doc_id, rank, score in ranked_docs:
            if doc_id in query_qrels:
                relevance_grade = query_qrels[doc_id]
                if relevance_grade > 0:
                    grade_ranks['all'].append(rank)
                    if relevance_grade in grade_ranks:
                        grade_ranks[relevance_grade].append(rank)

    avg_ranks = {}
    for grade, ranks in grade_ranks.items():
        if ranks:
            avg_ranks[grade] = {
                'mean': np.mean(ranks),
                'median': np.median(ranks),
                'count': len(ranks)
            }
        else:
            avg_ranks[grade] = {'mean': 0, 'median': 0, 'count': 0}

    return avg_ranks


def analyze_ranking_distribution_with_return(run_data: dict, qrels_data: dict, baseline_run_data: dict = None):
    """
    Performs comprehensive ranking distribution analysis and returns results for saving.
    """
    # Get overall distribution analysis
    distribution_stats, avg_ranks, baseline_avg_ranks, total_relevant_docs, queries_processed = analyze_ranking_distribution(
        run_data, qrels_data, baseline_run_data)

    # Get per-query detailed analysis
    per_query_results = analyze_per_query_improvements(run_data, qrels_data, baseline_run_data)

    return distribution_stats, avg_ranks, baseline_avg_ranks, total_relevant_docs, queries_processed, per_query_results


def analyze_ranking_distribution(run_data: dict, qrels_data: dict, baseline_run_data: dict = None):
    """
    Performs comprehensive ranking distribution analysis for the paper.

    Args:
        run_data: The loaded run file data (REGENT results).
        qrels_data: The loaded qrels data.
        baseline_run_data: Optional baseline run data for comparison.
    """

    # Define the ranking bins for analysis
    bins = {
        "Top 10": (1, 10),
        "11-50": (11, 50),
        "51-100": (51, 100),
        "Beyond 100": (101, 1000)
    }

    # Initialize counters for the analysis
    distribution_stats = {
        "Overall": {bin_name: 0 for bin_name in bins},
        "Highly Relevant (Grade 2)": {bin_name: 0 for bin_name in bins},
        "Relevant (Grade 1)": {bin_name: 0 for bin_name in bins},
    }

    total_relevant_docs = 0
    queries_processed = 0

    # Iterate through each query in the run file
    for query_id, ranked_docs in run_data.items():
        if query_id not in qrels_data:
            continue

        queries_processed += 1
        query_qrels = qrels_data[query_id]

        # Iterate through the ranked list of documents for the current query
        for doc_id, rank, score in ranked_docs:
            if doc_id in query_qrels:
                relevance_grade = query_qrels[doc_id]

                if relevance_grade > 0:
                    total_relevant_docs += 1
                    # Find which bin the document's rank falls into
                    for bin_name, (start_rank, end_rank) in bins.items():
                        if start_rank <= rank <= end_rank:
                            distribution_stats["Overall"][bin_name] += 1
                            if relevance_grade == 2:
                                distribution_stats["Highly Relevant (Grade 2)"][bin_name] += 1
                            elif relevance_grade == 1:
                                distribution_stats["Relevant (Grade 1)"][bin_name] += 1
                            break

    # Calculate average ranks
    avg_ranks = calculate_average_ranks(run_data, qrels_data)
    baseline_avg_ranks = None
    if baseline_run_data:
        baseline_avg_ranks = calculate_average_ranks(baseline_run_data, qrels_data)

    # --- Print the analysis results ---
    print("\n" + "=" * 80)
    print("REGENT: Ranking Distribution Analysis for Paper Section")
    print("=" * 80)

    # Print overall distribution
    print(f"\nProcessed {queries_processed} queries with {total_relevant_docs} total relevant documents")

    print("\nOverall Distribution of Relevant Documents:")
    for bin_name, count in distribution_stats["Overall"].items():
        percentage = (count / total_relevant_docs * 100) if total_relevant_docs > 0 else 0
        print(f"  - {bin_name}: {count} documents ({percentage:.1f}%)")

    top_10_count = distribution_stats["Overall"]["Top 10"]
    top_50_count = top_10_count + distribution_stats["Overall"]["11-50"]
    beyond_100_count = distribution_stats["Overall"]["Beyond 100"]

    print(f"\n🎯 KEY STATISTICS FOR PAPER:")
    print(f" → {top_10_count} relevant documents appear within the first 10 positions")
    print(
        f" → {top_50_count} documents ({top_50_count / total_relevant_docs:.0%} of all relevant content) are promoted to top 50")
    print(f" → Only {beyond_100_count} relevant documents appear beyond rank 100")

    # Print distribution by relevance grade with detailed statistics
    print(f"\n📊 DISTRIBUTION BY RELEVANCE LEVEL:")

    for grade_label, stats in distribution_stats.items():
        if grade_label != "Overall":
            grade_total = sum(stats.values())
            if grade_total == 0:
                continue

            print(f"\n--- {grade_label} ({grade_total} total documents) ---")

            top_10_grade = stats["Top 10"]
            top_11_50_grade = stats["11-50"]
            top_50_grade = top_10_grade + top_11_50_grade

            print(f"  • Top 10: {top_10_grade} documents ({top_10_grade / grade_total * 100:.1f}%)")
            print(f"  • Ranks 11-50: {top_11_50_grade} documents ({top_11_50_grade / grade_total * 100:.1f}%)")
            print(f"  • Top 50 total: {top_50_grade} documents ({top_50_grade / grade_total * 100:.1f}%)")
            print(f"  • Beyond 100: {stats['Beyond 100']} documents")

    # Print average rank improvements
    print(f"\n📈 AVERAGE RANK ANALYSIS:")

    if baseline_avg_ranks:
        print(f"Comparison with baseline:")
        for grade in [2, 1, 'all']:
            if grade in avg_ranks and grade in baseline_avg_ranks:
                current_avg = avg_ranks[grade]['mean']
                baseline_avg = baseline_avg_ranks[grade]['mean']
                improvement = baseline_avg - current_avg

                grade_name = {2: "Highly Relevant (Grade 2)", 1: "Relevant (Grade 1)", 'all': "All Relevant"}[grade]
                print(f"  • {grade_name}:")
                print(f"    - REGENT average rank: {current_avg:.0f}")
                print(f"    - Baseline average rank: {baseline_avg:.0f}")
                print(f"    - Improvement: {improvement:.0f} positions")
    else:
        print(f"REGENT average ranks:")
        for grade in [2, 1, 'all']:
            if grade in avg_ranks:
                avg_rank = avg_ranks[grade]['mean']
                count = avg_ranks[grade]['count']
                grade_name = {2: "Highly Relevant (Grade 2)", 1: "Relevant (Grade 1)", 'all': "All Relevant"}[grade]
                print(f"  • {grade_name}: {avg_rank:.0f} (n={count})")

    # Generate paper-ready text
    print(f"\n" + "=" * 80)
    print("📝 PAPER-READY TEXT:")
    print("=" * 80)

    paper_text = f"""The strong performance is further validated by examining REGENT's ranking distribution. 
{top_10_count} relevant documents appear within the first 10 positions, while {top_50_count} documents 
(over {top_50_count / total_relevant_docs:.0%} of all relevant content) are successfully promoted to the 
top 50 positions. This sharp top-rank concentration is underscored by only {beyond_100_count} relevant 
documents appearing beyond rank 100, indicating REGENT's consistent ability to identify and elevate 
relevant content.

The impact becomes even more pronounced when analyzing relevance levels. For highly relevant documents 
(Grade 2), {distribution_stats["Highly Relevant (Grade 2)"]["Top 10"]} appear in the top 10, with 
{distribution_stats["Highly Relevant (Grade 2)"]["Top 10"] + distribution_stats["Highly Relevant (Grade 2)"]["11-50"]} 
in the top 50. For relevant documents (Grade 1), {distribution_stats["Relevant (Grade 1)"]["Top 10"]} 
appear in the top 10 and {distribution_stats["Relevant (Grade 1)"]["11-50"]} between ranks 11-50."""

    if baseline_avg_ranks:
        grade2_improvement = baseline_avg_ranks[2]['mean'] - avg_ranks[2]['mean']
        grade1_improvement = baseline_avg_ranks[1]['mean'] - avg_ranks[1]['mean']
        paper_text += f""" REGENT improves the average rank of highly relevant documents from 
{baseline_avg_ranks[2]['mean']:.0f} to {avg_ranks[2]['mean']:.0f}, and relevant documents from 
{baseline_avg_ranks[1]['mean']:.0f} to {avg_ranks[1]['mean']:.0f}."""

    print(paper_text)
    print("=" * 80)

    # Return results for saving
    return distribution_stats, avg_ranks, baseline_avg_ranks, total_relevant_docs, queries_processed


def analyze_per_query_improvements(run_data: dict, qrels_data: dict, baseline_run_data: dict = None):
    """
    Analyze improvements on a per-query basis.

    Returns:
        Dictionary with detailed per-query analysis.
    """
    per_query_results = {}

    for query_id in run_data.keys():
        if query_id not in qrels_data:
            continue

        query_qrels = qrels_data[query_id]
        relevant_docs = {doc_id: grade for doc_id, grade in query_qrels.items() if grade > 0}

        if not relevant_docs:
            continue

        # Get REGENT rankings
        regent_rankings = {doc_id: rank for doc_id, rank, score in run_data[query_id]}

        # Get baseline rankings if available
        baseline_rankings = {}
        if baseline_run_data and query_id in baseline_run_data:
            baseline_rankings = {doc_id: rank for doc_id, rank, score in baseline_run_data[query_id]}

        # Analyze this query's results
        query_analysis = {
            "query_id": query_id,
            "total_relevant_docs": len(relevant_docs),
            "relevant_docs_found": 0,
            "relevant_docs_in_top_10": 0,
            "relevant_docs_in_top_50": 0,
            "average_relevant_rank": 0,
            "improved_documents": [],
            "degraded_documents": [],
            "new_documents_found": [],
            "lost_documents": [],
            "rank_improvements": [],
            "documents_by_grade": {1: [], 2: []},
            "baseline_comparison": {}
        }

        relevant_ranks = []

        # Analyze each relevant document
        for doc_id, relevance_grade in relevant_docs.items():
            doc_analysis = {
                "doc_id": doc_id,
                "relevance_grade": relevance_grade,
                "regent_rank": None,
                "baseline_rank": None,
                "rank_improvement": None,
                "in_regent_results": False,
                "in_baseline_results": False
            }

            # Check REGENT ranking
            if doc_id in regent_rankings:
                doc_analysis["regent_rank"] = regent_rankings[doc_id]
                doc_analysis["in_regent_results"] = True
                relevant_ranks.append(regent_rankings[doc_id])
                query_analysis["relevant_docs_found"] += 1

                if regent_rankings[doc_id] <= 10:
                    query_analysis["relevant_docs_in_top_10"] += 1
                if regent_rankings[doc_id] <= 50:
                    query_analysis["relevant_docs_in_top_50"] += 1

            # Check baseline ranking
            if doc_id in baseline_rankings:
                doc_analysis["baseline_rank"] = baseline_rankings[doc_id]
                doc_analysis["in_baseline_results"] = True

            # Calculate improvement
            if doc_analysis["regent_rank"] and doc_analysis["baseline_rank"]:
                improvement = doc_analysis["baseline_rank"] - doc_analysis["regent_rank"]
                doc_analysis["rank_improvement"] = improvement
                query_analysis["rank_improvements"].append(improvement)

                if improvement > 0:
                    query_analysis["improved_documents"].append(doc_analysis.copy())
                elif improvement < 0:
                    query_analysis["degraded_documents"].append(doc_analysis.copy())

            # Handle new/lost documents
            if doc_analysis["in_regent_results"] and not doc_analysis["in_baseline_results"]:
                query_analysis["new_documents_found"].append(doc_analysis.copy())
            elif doc_analysis["in_baseline_results"] and not doc_analysis["in_regent_results"]:
                query_analysis["lost_documents"].append(doc_analysis.copy())

            # Group by relevance grade
            if relevance_grade in query_analysis["documents_by_grade"]:
                query_analysis["documents_by_grade"][relevance_grade].append(doc_analysis.copy())

        # Calculate summary statistics
        if relevant_ranks:
            query_analysis["average_relevant_rank"] = np.mean(relevant_ranks)

        if baseline_run_data and query_id in baseline_run_data:
            # Baseline comparison for this query
            baseline_relevant_ranks = []
            baseline_top_10 = 0
            baseline_top_50 = 0

            for doc_id, grade in relevant_docs.items():
                if doc_id in baseline_rankings:
                    baseline_relevant_ranks.append(baseline_rankings[doc_id])
                    if baseline_rankings[doc_id] <= 10:
                        baseline_top_10 += 1
                    if baseline_rankings[doc_id] <= 50:
                        baseline_top_50 += 1

            query_analysis["baseline_comparison"] = {
                "baseline_relevant_found": len(baseline_relevant_ranks),
                "baseline_top_10": baseline_top_10,
                "baseline_top_50": baseline_top_50,
                "baseline_avg_rank": np.mean(baseline_relevant_ranks) if baseline_relevant_ranks else 0,
                "improvement_top_10": query_analysis["relevant_docs_in_top_10"] - baseline_top_10,
                "improvement_top_50": query_analysis["relevant_docs_in_top_50"] - baseline_top_50,
                "avg_rank_improvement": np.mean(query_analysis["rank_improvements"]) if query_analysis[
                    "rank_improvements"] else 0
            }

        per_query_results[query_id] = query_analysis

    return per_query_results


def save_detailed_results(distribution_stats, avg_ranks, baseline_avg_ranks, total_relevant_docs,
                          queries_processed, per_query_results, output_dir="ranking_analysis_results"):
    """
    Save detailed analysis results to disk in multiple formats.
    """
    import os
    import json

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save distribution statistics as JSON
    results_dict = {
        "summary": {
            "total_queries_processed": queries_processed,
            "total_relevant_documents": total_relevant_docs,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        },
        "distribution_by_rank_bins": distribution_stats,
        "average_ranks": avg_ranks,
        "baseline_average_ranks": baseline_avg_ranks if baseline_avg_ranks else None
    }

    with open(f"{output_dir}/ranking_analysis_results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"📁 Detailed results saved to: {output_dir}/ranking_analysis_results.json")

    # 2. Save distribution table as CSV
    distribution_df_data = []
    for category, bins in distribution_stats.items():
        for bin_name, count in bins.items():
            percentage = (count / total_relevant_docs * 100) if total_relevant_docs > 0 else 0
            distribution_df_data.append({
                "Category": category,
                "Rank_Range": bin_name,
                "Document_Count": count,
                "Percentage": round(percentage, 2)
            })

    distribution_df = pd.DataFrame(distribution_df_data)
    distribution_df.to_csv(f"{output_dir}/distribution_by_ranks.csv", index=False)
    print(f"📊 Distribution table saved to: {output_dir}/distribution_by_ranks.csv")

    # 3. Save average rank comparison as CSV
    if avg_ranks:
        avg_rank_data = []
        for grade, stats in avg_ranks.items():
            row = {
                "Relevance_Grade": grade,
                "REGENT_Average_Rank": round(stats['mean'], 1),
                "REGENT_Median_Rank": round(stats['median'], 1),
                "Document_Count": stats['count']
            }

            if baseline_avg_ranks and grade in baseline_avg_ranks:
                baseline_stats = baseline_avg_ranks[grade]
                row["Baseline_Average_Rank"] = round(baseline_stats['mean'], 1)
                row["Baseline_Median_Rank"] = round(baseline_stats['median'], 1)
                row["Rank_Improvement"] = round(baseline_stats['mean'] - stats['mean'], 1)

            avg_rank_data.append(row)

        avg_rank_df = pd.DataFrame(avg_rank_data)
        avg_rank_df.to_csv(f"{output_dir}/average_rank_comparison.csv", index=False)
        print(f"📈 Average rank comparison saved to: {output_dir}/average_rank_comparison.csv")

    # 4. Save per-query detailed results
    if per_query_results:
        # Save complete per-query data as JSON
        with open(f"{output_dir}/per_query_detailed_results.json", 'w') as f:
            json.dump(per_query_results, f, indent=2)
        print(f"🔍 Per-query detailed results saved to: {output_dir}/per_query_detailed_results.json")

        # Create per-query summary CSV
        per_query_summary = []
        for query_id, analysis in per_query_results.items():
            row = {
                "Query_ID": query_id,
                "Total_Relevant_Docs": analysis["total_relevant_docs"],
                "Relevant_Found": analysis["relevant_docs_found"],
                "Top_10_Count": analysis["relevant_docs_in_top_10"],
                "Top_50_Count": analysis["relevant_docs_in_top_50"],
                "Avg_Relevant_Rank": round(analysis["average_relevant_rank"], 1),
                "Documents_Improved": len(analysis["improved_documents"]),
                "Documents_Degraded": len(analysis["degraded_documents"]),
                "New_Documents_Found": len(analysis["new_documents_found"]),
                "Documents_Lost": len(analysis["lost_documents"])
            }

            if analysis["baseline_comparison"]:
                bc = analysis["baseline_comparison"]
                row.update({
                    "Baseline_Top_10": bc["baseline_top_10"],
                    "Baseline_Top_50": bc["baseline_top_50"],
                    "Baseline_Avg_Rank": round(bc["baseline_avg_rank"], 1),
                    "Top_10_Improvement": bc["improvement_top_10"],
                    "Top_50_Improvement": bc["improvement_top_50"],
                    "Avg_Rank_Improvement": round(bc["avg_rank_improvement"], 1)
                })

            per_query_summary.append(row)

        per_query_df = pd.DataFrame(per_query_summary)
        per_query_df.to_csv(f"{output_dir}/per_query_summary.csv", index=False)
        print(f"📋 Per-query summary saved to: {output_dir}/per_query_summary.csv")

        # Create document-level improvements CSV
        doc_improvements = []
        for query_id, analysis in per_query_results.items():
            for doc in analysis["improved_documents"]:
                doc_improvements.append({
                    "Query_ID": query_id,
                    "Doc_ID": doc["doc_id"],
                    "Relevance_Grade": doc["relevance_grade"],
                    "REGENT_Rank": doc["regent_rank"],
                    "Baseline_Rank": doc["baseline_rank"],
                    "Rank_Improvement": doc["rank_improvement"]
                })

        if doc_improvements:
            improvements_df = pd.DataFrame(doc_improvements)
            improvements_df.to_csv(f"{output_dir}/document_improvements.csv", index=False)
            print(f"⬆️ Document improvements saved to: {output_dir}/document_improvements.csv")

    # 5. Save paper-ready LaTeX snippet
    top_10_count = distribution_stats["Overall"]["Top 10"]
    top_50_count = top_10_count + distribution_stats["Overall"]["11-50"]
    beyond_100_count = distribution_stats["Overall"]["Beyond 100"]

    latex_text = f"""% REGENT Ranking Distribution Results
% Generated automatically from ranking analysis

The strong performance is further validated by examining REGENT's ranking distribution. 
{top_10_count} relevant documents appear within the first 10 positions, while {top_50_count} documents 
(over {top_50_count / total_relevant_docs:.0%} of all relevant content) are successfully promoted to the 
top 50 positions. This sharp top-rank concentration is underscored by only {beyond_100_count} relevant 
documents appearing beyond rank 100, indicating REGENT's consistent ability to identify and elevate 
relevant content.

The impact becomes even more pronounced when analyzing relevance levels. For highly relevant documents 
(Grade 2), {distribution_stats["Highly Relevant (Grade 2)"]["Top 10"]} appear in the top 10, with 
{distribution_stats["Highly Relevant (Grade 2)"]["Top 10"] + distribution_stats["Highly Relevant (Grade 2)"]["11-50"]} 
in the top 50. For relevant documents (Grade 1), {distribution_stats["Relevant (Grade 1)"]["Top 10"]} 
appear in the top 10 and {distribution_stats["Relevant (Grade 1)"]["11-50"]} between ranks 11-50."""

    if baseline_avg_ranks:
        grade2_improvement = baseline_avg_ranks[2]['mean'] - avg_ranks[2]['mean']
        grade1_improvement = baseline_avg_ranks[1]['mean'] - avg_ranks[1]['mean']
        latex_text += f""" REGENT improves the average rank of highly relevant documents from 
{baseline_avg_ranks[2]['mean']:.0f} to {avg_ranks[2]['mean']:.0f}, and relevant documents from 
{baseline_avg_ranks[1]['mean']:.0f} to {avg_ranks[1]['mean']:.0f}."""

    with open(f"{output_dir}/paper_text.tex", 'w') as f:
        f.write(latex_text)
    print(f"📝 Paper-ready LaTeX text saved to: {output_dir}/paper_text.tex")

    # 6. Save summary statistics for quick reference
    summary_stats = {
        "Key Statistics": {
            "Total Queries": queries_processed,
            "Total Relevant Documents": total_relevant_docs,
            "Documents in Top 10": top_10_count,
            "Documents in Top 50": top_50_count,
            "Top 50 Percentage": f"{top_50_count / total_relevant_docs:.0%}",
            "Documents Beyond Rank 100": beyond_100_count
        }
    }

    # Add grade-specific statistics
    for grade in [1, 2]:
        grade_name = f"Grade {grade}"
        if grade in avg_ranks:
            summary_stats[f"Relevance {grade_name}"] = {
                "Average Rank": round(avg_ranks[grade]['mean'], 1),
                "Top 10 Count": distribution_stats[f"{'Highly ' if grade == 2 else ''}Relevant (Grade {grade})"][
                    "Top 10"],
                "Top 50 Count": distribution_stats[f"{'Highly ' if grade == 2 else ''}Relevant (Grade {grade})"][
                                    "Top 10"] +
                                distribution_stats[f"{'Highly ' if grade == 2 else ''}Relevant (Grade {grade})"][
                                    "11-50"]
            }

            if baseline_avg_ranks and grade in baseline_avg_ranks:
                improvement = baseline_avg_ranks[grade]['mean'] - avg_ranks[grade]['mean']
                summary_stats[f"Relevance {grade_name}"]["Rank Improvement"] = round(improvement, 1)

    # Add per-query summary statistics
    if per_query_results:
        query_improvements = [len(analysis["improved_documents"]) for analysis in per_query_results.values()]
        query_degradations = [len(analysis["degraded_documents"]) for analysis in per_query_results.values()]

        summary_stats["Per-Query Analysis"] = {
            "Queries Analyzed": len(per_query_results),
            "Avg Documents Improved per Query": round(np.mean(query_improvements), 1),
            "Avg Documents Degraded per Query": round(np.mean(query_degradations), 1),
            "Total Document Improvements": sum(query_improvements),
            "Total Document Degradations": sum(query_degradations)
        }

    with open(f"{output_dir}/summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"📋 Summary statistics saved to: {output_dir}/summary_statistics.json")

    print(f"\n✅ All results saved to directory: {output_dir}/")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the distribution of relevant documents in a TREC run file for REGENT paper."
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to the TREC-formatted run file generated by REGENT."
    )
    parser.add_argument(
        "--qrels",
        required=True,
        help="Path to the corresponding qrels file for relevance judgments."
    )
    parser.add_argument(
        "--baseline",
        help="Path to baseline run file (e.g., BM25) for comparison."
    )
    parser.add_argument(
        "--output-dir",
        default="ranking_analysis_results",
        help="Directory to save detailed results (default: ranking_analysis_results)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to disk"
    )

    args = parser.parse_args()

    # Load the data
    run_data = load_trec_run(args.run)
    qrels_data = load_qrels(args.qrels)

    baseline_run_data = None
    if args.baseline:
        baseline_run_data = load_trec_run(args.baseline)

    # Perform and print the analysis
    distribution_stats, avg_ranks, baseline_avg_ranks, total_relevant_docs, queries_processed, per_query_results = analyze_ranking_distribution_with_return(
        run_data, qrels_data, baseline_run_data
    )

    # Save results to disk if requested
    if args.save_results:
        save_detailed_results(
            distribution_stats, avg_ranks, baseline_avg_ranks,
            total_relevant_docs, queries_processed, per_query_results, args.output_dir
        )


if __name__ == "__main__":
    main()