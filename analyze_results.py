import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set
import networkx as nx
import numpy as np
from collections import defaultdict


def load_results(csv_file: str = "unique_companies.csv") -> pd.DataFrame:
    """Load the results from the CSV file."""
    return pd.read_csv(csv_file)


def create_match_network(matches: List[Dict]) -> nx.Graph:
    """Create a network graph of company matches."""
    G = nx.Graph()

    # Add edges between matched companies
    for match in matches:
        G.add_edge(
            match["company1"],
            match["company2"],
            weight=match["similarity"],
            block_key=match["block_key"],
        )

    return G


def visualize_matches(
    df: pd.DataFrame, matches: List[Dict], output_file: str = "match_visualization.png"
):
    """Create visualizations of the matching results."""
    # Set style
    plt.style.use("seaborn")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Distribution of match counts
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x="match_count", bins=30)
    plt.title("Distribution of Match Counts")
    plt.xlabel("Number of Matches")
    plt.ylabel("Count")

    # 2. Match counts by country
    plt.subplot(2, 2, 2)
    country_matches = (
        df.groupby("main_country_code")["match_count"]
        .mean()
        .sort_values(ascending=False)
    )
    country_matches.head(10).plot(kind="bar")
    plt.title("Average Matches by Country (Top 10)")
    plt.xlabel("Country Code")
    plt.ylabel("Average Number of Matches")
    plt.xticks(rotation=45)

    # 3. Match counts by industry
    plt.subplot(2, 2, 3)
    industry_matches = (
        df.groupby("main_industry")["match_count"].mean().sort_values(ascending=False)
    )
    industry_matches.head(10).plot(kind="bar")
    plt.title("Average Matches by Industry (Top 10)")
    plt.xlabel("Industry")
    plt.ylabel("Average Number of Matches")
    plt.xticks(rotation=45)

    # 4. Similarity score distribution
    plt.subplot(2, 2, 4)
    similarities = [match["similarity"] for match in matches]
    sns.histplot(similarities, bins=30)
    plt.title("Distribution of Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def analyze_match_quality(matches: List[Dict], df: pd.DataFrame) -> Dict:
    """Analyze the quality of matches based on various criteria."""
    quality_metrics = {
        "total_matches": len(matches),
        "unique_companies": len(df),
        "exact_matches": 0,
        "high_similarity_matches": 0,
        "matches_by_country": defaultdict(int),
        "matches_by_industry": defaultdict(int),
        "similarity_distribution": defaultdict(int),
        "similarity_bins": {
            "0.90-0.92": 0,
            "0.93-0.94": 0,
            "0.95-0.96": 0,
            "0.97-0.98": 0,
            "0.99-0.999": 0,
            "1.0": 0,
        },
        "validation_stats": {
            "country_match": 0,
            "industry_match": 0,
            "website_match": 0,
            "all_fields_match": 0,
        },
    }

    for match in matches:
        # Count exact matches
        if match["similarity"] == 1.0:
            quality_metrics["exact_matches"] += 1

        # Count high similarity matches
        if match["similarity"] >= 0.95:
            quality_metrics["high_similarity_matches"] += 1

        # Count matches by country using processed names
        company1_processed = match["company1_processed"]
        company2_processed = match["company2_processed"]

        # Get country information using processed names
        company1_country = df[df["company_name_processed"] == company1_processed][
            "main_country_code"
        ].iloc[0]
        company2_country = df[df["company_name_processed"] == company2_processed][
            "main_country_code"
        ].iloc[0]
        if company1_country == company2_country:
            quality_metrics["matches_by_country"][company1_country] += 1
            quality_metrics["validation_stats"]["country_match"] += 1

        # Get industry information using processed names
        company1_industry = df[df["company_name_processed"] == company1_processed][
            "main_industry"
        ].iloc[0]
        company2_industry = df[df["company_name_processed"] == company2_processed][
            "main_industry"
        ].iloc[0]
        if company1_industry == company2_industry:
            quality_metrics["matches_by_industry"][company1_industry] += 1
            quality_metrics["validation_stats"]["industry_match"] += 1

        # Check website match
        company1_website = df[df["company_name_processed"] == company1_processed][
            "website_processed"
        ].iloc[0]
        company2_website = df[df["company_name_processed"] == company2_processed][
            "website_processed"
        ].iloc[0]
        if (
            company1_website
            and company2_website
            and company1_website == company2_website
        ):
            quality_metrics["validation_stats"]["website_match"] += 1

        # Check if all fields match
        if (
            company1_country == company2_country
            and company1_industry == company2_industry
            and company1_website
            and company2_website
            and company1_website == company2_website
        ):
            quality_metrics["validation_stats"]["all_fields_match"] += 1

        # Track similarity distribution
        similarity = match["similarity"]
        if similarity == 1.0:
            quality_metrics["similarity_bins"]["1.0"] += 1
        elif similarity >= 0.99:
            quality_metrics["similarity_bins"]["0.99-0.999"] += 1
        elif similarity >= 0.97:
            quality_metrics["similarity_bins"]["0.97-0.98"] += 1
        elif similarity >= 0.95:
            quality_metrics["similarity_bins"]["0.95-0.96"] += 1
        elif similarity >= 0.93:
            quality_metrics["similarity_bins"]["0.93-0.94"] += 1
        else:
            quality_metrics["similarity_bins"]["0.90-0.92"] += 1

    return quality_metrics


def print_quality_report(metrics: Dict):
    """Print a detailed report of match quality metrics."""
    print("\nMatch Quality Report")
    print("=" * 50)
    print(f"Total number of matches: {metrics['total_matches']}")
    print(f"Number of unique companies: {metrics['unique_companies']}")
    print(f"Exact matches (similarity = 1.0): {metrics['exact_matches']}")
    print(f"High similarity matches (>= 0.95): {metrics['high_similarity_matches']}")

    print("\nTop 5 Countries by Match Count:")
    for country, count in sorted(
        metrics["matches_by_country"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{country}: {count} matches")

    print("\nTop 5 Industries by Match Count:")
    for industry, count in sorted(
        metrics["matches_by_industry"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{industry}: {count} matches")

    print("\nSimilarity Score Distribution:")
    for bin_name, count in sorted(metrics["similarity_bins"].items()):
        percentage = (count / metrics["total_matches"]) * 100
        print(f"{bin_name}: {count} matches ({percentage:.1f}%)")

    print("\nValidation Statistics:")
    total_matches = metrics["total_matches"]
    print(
        f"Matches with same country: {metrics['validation_stats']['country_match']} ({metrics['validation_stats']['country_match']/total_matches*100:.1f}%)"
    )
    print(
        f"Matches with same industry: {metrics['validation_stats']['industry_match']} ({metrics['validation_stats']['industry_match']/total_matches*100:.1f}%)"
    )
    print(
        f"Matches with same website: {metrics['validation_stats']['website_match']} ({metrics['validation_stats']['website_match']/total_matches*100:.1f}%)"
    )
    print(
        f"Matches with all fields matching: {metrics['validation_stats']['all_fields_match']} ({metrics['validation_stats']['all_fields_match']/total_matches*100:.1f}%)"
    )


def main():
    # Load results
    print("Loading results...")
    df = load_results()

    # Load matches from the original script
    matches = pd.read_csv("matches.csv").to_dict("records")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_matches(df, matches)

    # Analyze match quality
    print("\nAnalyzing match quality...")
    quality_metrics = analyze_match_quality(matches, df)

    # Print quality report
    print_quality_report(quality_metrics)


if __name__ == "__main__":
    main()
