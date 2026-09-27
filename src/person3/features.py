import re
import pandas as pd

from preprocessing import preprocess_dataframe


def jaccard_similarity(a, b):
    """Token-set Jaccard similarity."""
    if not a or not b:
        return 0.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def character_similarity(a, b):
    """Simple character overlap similarity."""
    if not a or not b:
        return 0.0

    a = re.sub(r"\s+", "", a)
    b = re.sub(r"\s+", "", b)

    if not a or not b:
        return 0.0

    shorter = min(len(a), len(b))

    common = 0
    for char in set(a):
        common += min(a.count(char), b.count(char))

    return common / max(len(a), len(b))


def exact_match(a, b):
    if not a or not b:
        return 0
    return int(a == b)


def country_match(a, b):
    if not a or not b:
        return 0
    return int(a == b)


def number_overlap(a, b):
    """Compare numeric parts of addresses."""
    if not a or not b:
        return 0.0

    a_numbers = set(a.split())
    b_numbers = set(b.split())

    if not a_numbers or not b_numbers:
        return 0.0

    return len(a_numbers & b_numbers) / len(a_numbers | b_numbers)


def create_pair_features(source1, source2):
    """
    Create matching features for pairs of business records.

    source1 and source2 must each contain one business record.
    """

    s1 = preprocess_dataframe(source1.copy())
    s2 = preprocess_dataframe(source2.copy())

    a = s1.iloc[0]
    b = s2.iloc[0]

    features = {
        "name_exact": exact_match(
            a["name_clean"],
            b["name_clean"]
        ),

        "name_token_jaccard": jaccard_similarity(
            a["name_tokens"],
            b["name_tokens"]
        ),

        "name_character_similarity": character_similarity(
            a["name_clean"],
            b["name_clean"]
        ),

        "address_exact": exact_match(
            a["address_clean"],
            b["address_clean"]
        ),

        "address_token_jaccard": jaccard_similarity(
            a["address_tokens"],
            b["address_tokens"]
        ),

        "address_number_overlap": number_overlap(
            a["address_numbers"],
            b["address_numbers"]
        ),

        "country_match": country_match(
            a["country_clean"],
            b["country_clean"]
        ),
    }

    return features
