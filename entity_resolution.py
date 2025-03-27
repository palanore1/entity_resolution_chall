import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from jellyfish import jaro_winkler_similarity
from collections import defaultdict


def preprocess_company_name(name: str) -> str:
    """
    Preprocess company name by:
    1. Converting to lowercase
    2. Removing special characters
    3. Removing common legal entity types
    4. Removing extra whitespace
    5. Converting accented characters to ASCII
    """
    if pd.isna(name):
        return ""

    # Convert to lowercase and remove accents
    name = unidecode(str(name).lower())

    # Remove common legal entity types
    legal_entities = [
        "llc",
        "ltd",
        "inc",
        "corp",
        "corporation",
        "co",
        "company",
        "limited",
        "incorporated",
        "gmbh",
        "ag",
        "sa",
        "srl",
        "bv",
        "nv",
        "spa",
        "plc",
        "pte",
        "pvt",
    ]

    # Remove legal entity types that are separate words
    words = name.split()
    words = [w for w in words if w not in legal_entities]
    name = " ".join(words)

    # Remove special characters but keep spaces
    name = re.sub(r"[^a-z0-9\s]", "", name)

    # Remove extra whitespace
    name = " ".join(name.split())

    return name


def preprocess_phone(phone: str) -> str:
    """
    Preprocess phone number by:
    1. Removing all non-numeric characters
    2. Removing country codes if present
    """
    if pd.isna(phone):
        return ""

    # Remove all non-numeric characters
    phone = re.sub(r"[^0-9]", "", str(phone))

    # Remove common country codes
    common_codes = ["1", "44", "61", "81", "86", "91"]
    for code in common_codes:
        if phone.startswith(code):
            phone = phone[len(code) :]
            break

    return phone


def preprocess_address(address: str) -> str:
    """
    Preprocess address by:
    1. Converting to lowercase
    2. Removing special characters
    3. Standardizing common terms
    4. Removing extra whitespace
    """
    if pd.isna(address):
        return ""

    # Convert to lowercase
    address = str(address).lower()

    # Standardize common terms
    replacements = {
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "road": "rd",
        "drive": "dr",
        "lane": "ln",
        "suite": "ste",
        "apartment": "apt",
    }

    for old, new in replacements.items():
        address = address.replace(old, new)

    # Remove special characters but keep spaces
    address = re.sub(r"[^a-z0-9\s]", "", address)

    # Remove extra whitespace
    address = " ".join(address.split())

    return address


def preprocess_website(url: str) -> str:
    """
    Preprocess website URL by:
    1. Removing protocol (http/https)
    2. Removing www
    3. Converting to lowercase
    4. Removing trailing slashes
    """
    if pd.isna(url):
        return ""

    url = str(url).lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    url = url.rstrip("/")
    return url


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Preprocess the entire dataframe and return both the preprocessed dataframe
    and a dictionary of original values for reference
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()

    # Store original values for later reference
    original_values = {
        "company_name": df["company_name"].tolist(),
        "company_legal_names": df["company_legal_names"].tolist(),
        "company_commercial_names": df["company_commercial_names"].tolist(),
        "website_url": df["website_url"].tolist(),
        "primary_phone": df["primary_phone"].tolist(),
        "main_address_raw_text": df["main_address_raw_text"].tolist(),
    }

    # Preprocess company names
    df_processed["company_name_processed"] = df["company_name"].apply(
        preprocess_company_name
    )
    df_processed["company_legal_names_processed"] = df["company_legal_names"].apply(
        preprocess_company_name
    )
    df_processed["company_commercial_names_processed"] = df[
        "company_commercial_names"
    ].apply(preprocess_company_name)

    # Preprocess contact information
    df_processed["phone_processed"] = df["primary_phone"].apply(preprocess_phone)
    df_processed["website_processed"] = df["website_url"].apply(preprocess_website)
    df_processed["address_processed"] = df["main_address_raw_text"].apply(
        preprocess_address
    )

    # Create combined features for better matching
    df_processed["name_variations"] = df_processed.apply(
        lambda x: " | ".join(
            filter(
                None,
                [
                    x["company_name_processed"],
                    x["company_legal_names_processed"],
                    x["company_commercial_names_processed"],
                ],
            )
        ),
        axis=1,
    )

    # Handle missing values in key columns
    df_processed = df_processed.fillna(
        {
            "main_country_code": "UNKNOWN",
            "main_city": "UNKNOWN",
            "main_region": "UNKNOWN",
            "business_model": "UNKNOWN",
            "main_sector": "UNKNOWN",
        }
    )

    return df_processed, original_values


def create_blocks(df: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Create blocks based on different blocking keys using preprocessed data.
    Returns a dictionary where keys are blocking keys and values are sets of indices.
    """
    blocks = {}

    # Block by country code and first word of company name
    for country in df["main_country_code"].unique():
        if country == "UNKNOWN":
            continue

        country_mask = df["main_country_code"] == country
        country_df = df[country_mask]

        # Create blocks by first word within each country
        for idx, name in country_df["company_name_processed"].items():
            if not name:
                continue

            words = name.split()
            if not words:
                continue

            first_word = words[0]
            if len(first_word) <= 2:  # Skip very short words
                continue

            # Create block key with country and first word
            block_key = f"country_{country}_word_{first_word}"
            if block_key not in blocks:
                blocks[block_key] = set()
            blocks[block_key].add(idx)

    # Block by website domain (only for companies with websites)
    website_mask = df["website_processed"].notna() & (df["website_processed"] != "")
    website_df = df[website_mask]

    for idx, website in website_df["website_processed"].items():
        if not website:
            continue

        domain = website.split("/")[0]
        if not domain:
            continue

        block_key = f"website_{domain}"
        if block_key not in blocks:
            blocks[block_key] = set()
        blocks[block_key].add(idx)

    # Block by phone number prefix (only for companies with phone numbers)
    phone_mask = df["phone_processed"].notna() & (df["phone_processed"] != "")
    phone_df = df[phone_mask]

    for idx, phone in phone_df["phone_processed"].items():
        if not phone or len(phone) < 3:
            continue

        prefix = phone[:3]
        block_key = f"phone_{prefix}"
        if block_key not in blocks:
            blocks[block_key] = set()
        blocks[block_key].add(idx)

    # Block by location (country + city) for companies with complete location data
    location_mask = (
        df["main_country_code"].notna()
        & (df["main_country_code"] != "UNKNOWN")
        & df["main_city"].notna()
        & (df["main_city"] != "UNKNOWN")
    )
    location_df = df[location_mask]

    for (country, city), group in location_df.groupby(
        ["main_country_code", "main_city"]
    ):
        if len(group) < 2:  # Skip single-company locations
            continue

        block_key = f"location_{country}_{city}".replace(" ", "_")
        blocks[block_key] = set(group.index)

    # Block by industry/sector for companies with complete industry data
    industry_mask = (
        df["main_industry"].notna()
        & (df["main_industry"] != "UNKNOWN")
        & df["main_sector"].notna()
        & (df["main_sector"] != "UNKNOWN")
    )
    industry_df = df[industry_mask]

    for (industry, sector), group in industry_df.groupby(
        ["main_industry", "main_sector"]
    ):
        if len(group) < 2:  # Skip single-company industries
            continue

        block_key = f"industry_{industry}_{sector}".replace(" ", "_")
        blocks[block_key] = set(group.index)

    # Filter out blocks with only one company
    blocks = {k: v for k, v in blocks.items() if len(v) > 1}

    # Filter out blocks that are too large (more than 200 companies)
    blocks = {k: v for k, v in blocks.items() if len(v) <= 200}

    return blocks


def analyze_blocks(blocks: Dict[str, Set[int]], df: pd.DataFrame) -> None:
    """Analyze and print statistics about the blocks."""
    print("\nBlock Analysis:")
    print(f"Total number of blocks: {len(blocks)}")

    block_sizes = [len(indices) for indices in blocks.values()]
    print(f"Average block size: {np.mean(block_sizes):.2f}")
    print(f"Max block size: {max(block_sizes)}")
    print(f"Min block size: {min(block_sizes)}")

    # Count records in blocks
    records_in_blocks = set().union(*blocks.values())
    print(f"Total unique records in blocks: {len(records_in_blocks)}")
    print(
        f"Percentage of records in blocks: {(len(records_in_blocks) / len(df)) * 100:.2f}%"
    )

    # Count blocks by type
    block_types = {}
    for key in blocks.keys():
        block_type = key.split("_")[0]
        block_types[block_type] = block_types.get(block_type, 0) + 1

    print("\nBlocks by type:")
    for block_type, count in sorted(
        block_types.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{block_type}: {count} blocks")

    # Analyze block sizes distribution
    size_ranges = [(2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float("inf"))]
    print("\nBlock size distribution:")
    for start, end in size_ranges:
        count = sum(1 for size in block_sizes if start <= size <= end)
        print(f"{start}-{int(end) if end != float('inf') else '∞'}: {count} blocks")


def compute_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity between two company names using multiple metrics.
    Returns a weighted combination of similarity scores.
    """
    if not name1 or not name2:
        return 0.0

    # For very short names (1-2 words), use Jaro-Winkler
    words1 = name1.split()
    words2 = name2.split()
    if len(words1) <= 2 and len(words2) <= 2:
        return jaro_winkler_similarity(name1, name2)

    # For longer names, use a combination of metrics
    # 1. TF-IDF cosine similarity for overall text similarity
    vectorizer = TfidfVectorizer(
        min_df=1,
        analyzer="char_wb",
        ngram_range=(2, 3),
        lowercase=True,
        strip_accents="unicode",
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([name1, name2])
        cosine_sim = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
    except:
        cosine_sim = 0.0

    # 2. Jaro-Winkler for character-level differences
    jw_sim = jaro_winkler_similarity(name1, name2)

    # 3. Word overlap ratio
    words1_set = set(words1)
    words2_set = set(words2)
    overlap = len(words1_set.intersection(words2_set))
    word_overlap = overlap / max(len(words1_set), len(words2_set))

    # Combine metrics with weights based on name length
    if len(words1) > 3 or len(words2) > 3:
        # For longer names, give more weight to TF-IDF
        return 0.5 * cosine_sim + 0.3 * jw_sim + 0.2 * word_overlap
    else:
        # For shorter names, give equal weight to all metrics
        return 0.4 * cosine_sim + 0.3 * jw_sim + 0.3 * word_overlap


def find_matches(df, threshold=0.88):
    """Find matching companies using blocking and multiple similarity metrics"""
    # Create blocks
    blocks = create_blocks(df)

    # Find matches within blocks
    matches = []
    processed_pairs = set()  # Keep track of processed pairs to avoid duplicates

    # Sort blocks by size to process smaller blocks first
    sorted_blocks = sorted(blocks.items(), key=lambda x: len(x[1]))

    # Early stopping if we find too many matches
    max_matches = 2000  # Increased from 1000 to 2000
    if len(matches) >= max_matches:
        return matches

    for block_key, indices in sorted_blocks:
        if len(indices) < 2:
            continue

        # Convert to list for easier indexing
        indices_list = list(indices)

        # For large blocks, only compare with nearby indices
        max_comparisons = 25  # Increased from 10 to 25
        for i in range(len(indices_list)):
            idx1 = indices_list[i]
            company1 = df.loc[idx1, "company_name"]

            # Skip if company1 is None
            if pd.isna(company1):
                continue

            # Get all name variations for company1
            name1_variations = df.loc[idx1, "name_variations"].split(" | ")

            # Compare with next max_comparisons companies
            for j in range(i + 1, min(i + max_comparisons + 1, len(indices_list))):
                idx2 = indices_list[j]
                company2 = df.loc[idx2, "company_name"]

                # Skip if company2 is None
                if pd.isna(company2):
                    continue

                # Skip if pair already processed
                pair_key = tuple(sorted([str(company1), str(company2)]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Get all name variations for company2
                name2_variations = df.loc[idx2, "name_variations"].split(" | ")

                # Quick check for exact matches in variations
                if any(
                    v1 == v2
                    for v1 in name1_variations
                    for v2 in name2_variations
                    if v1 and v2
                ):
                    matches.append(
                        {
                            "company1": company1,
                            "company2": company2,
                            "similarity": 1.0,
                            "block_key": block_key,
                            "company1_processed": df.loc[
                                idx1, "company_name_processed"
                            ],
                            "company2_processed": df.loc[
                                idx2, "company_name_processed"
                            ],
                        }
                    )
                    continue

                # Compare each variation
                max_similarity = 0
                for name1 in name1_variations:
                    for name2 in name2_variations:
                        if not name1 or not name2:
                            continue
                        similarity = compute_similarity(name1, name2)
                        max_similarity = max(max_similarity, similarity)
                        # Early stopping if we find a good match
                        if max_similarity >= threshold:
                            break
                    if max_similarity >= threshold:
                        break

                if max_similarity >= threshold:
                    # Additional checks for high-quality matches
                    if max_similarity >= 0.95:
                        # For high similarity matches, check if they're in the same country
                        country1 = df.loc[idx1, "main_country_code"]
                        country2 = df.loc[idx2, "main_country_code"]
                        if (
                            country1 != "UNKNOWN"
                            and country2 != "UNKNOWN"
                            and country1 != country2
                        ):
                            # If countries don't match, require higher similarity
                            if max_similarity < 0.98:
                                continue

                    matches.append(
                        {
                            "company1": company1,
                            "company2": company2,
                            "similarity": max_similarity,
                            "block_key": block_key,
                            "company1_processed": df.loc[
                                idx1, "company_name_processed"
                            ],
                            "company2_processed": df.loc[
                                idx2, "company_name_processed"
                            ],
                        }
                    )

                # Early stopping if we find too many matches
                if len(matches) >= max_matches:
                    return matches

    # Sort matches by similarity score
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def create_unique_companies_df(df: pd.DataFrame, matches: List[Dict]) -> pd.DataFrame:
    """
    Create a DataFrame containing unique companies and their details from the matches.
    """
    # Create a mapping of processed names to original names more efficiently
    processed_to_original = {}
    for _, row in df.iterrows():
        processed_name = row["company_name_processed"]
        if not processed_name:  # Skip empty processed names
            continue
        if processed_name in processed_to_original:
            # If we already have this processed name, keep the shorter original name
            if len(row["company_name"]) < len(processed_to_original[processed_name]):
                processed_to_original[processed_name] = row["company_name"]
        else:
            processed_to_original[processed_name] = row["company_name"]

    # Create a set of unique companies from matches using processed names
    unique_processed_names = set()
    for match in matches:
        unique_processed_names.add(match["company1_processed"])
        unique_processed_names.add(match["company2_processed"])

    # Create a DataFrame with unique companies more efficiently
    unique_companies_df = df[
        df["company_name_processed"].isin(unique_processed_names)
    ].copy()

    # Keep only one row per processed name by using the mapping
    unique_companies_df = unique_companies_df[
        unique_companies_df.apply(
            lambda x: x["company_name"]
            == processed_to_original.get(x["company_name_processed"]),
            axis=1,
        )
    ]

    # Add additional columns for analysis
    unique_companies_df["has_match"] = True
    unique_companies_df["match_count"] = 0

    # Count number of matches for each company using processed names more efficiently
    match_counts = defaultdict(int)
    for match in matches:
        processed1 = match["company1_processed"]
        processed2 = match["company2_processed"]

        # If processed names match, count as one match
        if processed1 == processed2:
            match_counts[processed1] += 1
        else:
            match_counts[processed1] += 1
            match_counts[processed2] += 1

    # Update match counts in the DataFrame
    unique_companies_df["match_count"] = unique_companies_df[
        "company_name_processed"
    ].map(match_counts)

    # Sort by match count and company name
    unique_companies_df = unique_companies_df.sort_values(
        ["match_count", "company_name"], ascending=[False, True]
    )

    # Add alternative names more efficiently
    alt_names_dict = defaultdict(list)
    for _, row in df.iterrows():
        processed_name = row["company_name_processed"]
        if processed_name in unique_processed_names:
            original_name = processed_to_original[processed_name]
            if row["company_name"] != original_name:
                alt_names_dict[processed_name].append(row["company_name"])

    # Update alternative names in the DataFrame
    unique_companies_df["alternative_names"] = unique_companies_df[
        "company_name_processed"
    ].map(lambda x: " | ".join(alt_names_dict[x]) if x in alt_names_dict else "")

    # Final check to ensure no duplicates
    unique_companies_df = unique_companies_df.drop_duplicates(
        subset=["company_name_processed"]
    )

    return unique_companies_df


def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet("./veridion_entity_resolution_challenge.snappy.parquet")
    print(f"Loaded {len(df)} companies")

    # Preprocess data
    print("\nPreprocessing data...")
    df, original_values = preprocess_dataframe(df)

    # Create blocks
    print("\nCreating blocks...")
    blocks = create_blocks(df)

    # Analyze blocks
    print("\nAnalyzing blocks...")
    analyze_blocks(blocks, df)

    # Find matches
    print("\nFinding potential matches...")
    matches = find_matches(df)

    # Print matches
    print("\nPotential matches:")
    for match in matches:
        print(
            f"\nMatch (similarity: {match['similarity']:.3f}):\n"
            f"  {match['company1']}\n"
            f"  {match['company2']}"
        )

    # Save matches to CSV
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv("matches.csv", index=False)
    print(f"\nSaved {len(matches)} matches to matches.csv")

    # Create DataFrame of unique companies
    print("\nCreating DataFrame of unique companies...")
    unique_companies_df = create_unique_companies_df(df, matches)

    # Save to CSV
    output_file = "unique_companies.csv"
    print(f"\nSaving unique companies to {output_file}...")
    unique_companies_df.to_csv(output_file, index=False)
    print(f"Saved {len(unique_companies_df)} unique companies to {output_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total companies in dataset: {len(df)}")
    print(f"Companies with matches: {len(unique_companies_df)}")
    print(f"Total number of matches found: {len(matches)}")
    print(
        f"Average matches per company: {unique_companies_df['match_count'].mean():.2f}"
    )
    print(f"Maximum matches for a company: {unique_companies_df['match_count'].max()}")


if __name__ == "__main__":
    main()
