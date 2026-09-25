import re
import pandas as pd


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def normalize_business_name(text):
    """
    Normalize business names for entity matching.

    Examples:
        ABC Technologies Pvt. Ltd.
        -> abc technologies pvt ltd

        ABC & Sons Corporation
        -> abc and sons corp
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Normalize "&"
    text = text.replace("&", " and ")

    # Normalize common business/legal terms
    replacements = {
        "private limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "privatelimited": "pvt ltd",
        "limited": "ltd",
        "corporation": "corp",
        "incorporated": "inc",
        "company": "co",
    }

    # Replace longer phrases first
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove punctuation, keep letters/numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(text):
    """
    Normalize business addresses.

    Numbers are intentionally preserved because they may contain:
      - building numbers
      - house numbers
      - PIN codes
      - street numbers
      - highway numbers
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Normalize "&"
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Common address variations
    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt",
        "building": "bldg",
        "floor": "fl",
        "near": "nr",
    }

    words = text.split()

    normalized_words = []

    for word in words:
        normalized_words.append(
            replacements.get(word, word)
        )

    text = " ".join(normalized_words)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(text):
    """
    Normalize country labels.

    IMPORTANT:
    Do NOT restrict this to US/India.
    This must also work for unseen countries such as France.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# NAME TOKENS
# ============================================================

def get_name_tokens(text):
    """
    Convert a normalized business name into unique tokens.

    Example:
        "abc technologies abc"
        -> "abc technologies"
    """

    if not text:
        return ""

    tokens = sorted(set(text.split()))

    return " ".join(tokens)


# ============================================================
# ADDRESS TOKENS
# ============================================================

def get_address_tokens(text):
    """
    Convert normalized address into unique tokens.
    """

    if not text:
        return ""

    tokens = sorted(set(text.split()))

    return " ".join(tokens)


# ============================================================
# EXTRACT NUMBERS
# ============================================================

def extract_numbers(text):
    """
    Extract all numeric parts.

    Example:
        "12 mg road bangalore 560001"
        -> "12 560001"
    """

    if pd.isna(text):
        return ""

    numbers = re.findall(
        r"\d+",
        str(text)
    )

    return " ".join(numbers)


# ============================================================
# COMPLETE DATAFRAME PREPROCESSING
# ============================================================

def preprocess_dataframe(df):
    """
    Add preprocessing columns to a dataframe.

    Original columns are preserved.
    """

    df = df.copy()

    # Handle missing values found in Source 2 and Source 3
    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
    )

    # Main normalized fields
    df["name_clean"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )

    df["country_clean"] = df["country"].map(
        normalize_country
    )

    # Token fields
    df["name_tokens"] = df["name_clean"].map(
        get_name_tokens
    )

    df["address_tokens"] = df["address_clean"].map(
        get_address_tokens
    )

    # Numeric information from addresses
    df["address_numbers"] = df["address_clean"].map(
        extract_numbers
    )

    return df


# ============================================================
# LIGHTWEIGHT PREPROCESSING
# ============================================================

def preprocess_for_blocking(df):
    """
    Lightweight preprocessing for the candidate-generation
    stage.

    This creates only the main fields required for blocking.
    """

    df = df.copy()

    df["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    df["business_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
    )

    df["country"] = (
        df["country"]
        .fillna("")
        .astype(str)
    )

    df["name_clean"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_clean"] = df["business_address"].map(
        normalize_address
    )

    df["country_clean"] = df["country"].map(
        normalize_country
    )

    return df
```
