```python
import os
import sqlite3
import pandas as pd

from preprocessing import preprocess_for_blocking


# ============================================================
# FILES
# ============================================================

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

DB_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_index.db"
)

# Process large files in chunks
CHUNK_SIZE = 100_000


# ============================================================
# BLOCKING KEY FUNCTIONS
# ============================================================

def create_blocking_keys(df):
    """
    Create multiple blocking keys.

    A candidate is generated if Source 1 and Source 2/3
    share at least one useful blocking key.

    We intentionally use several independent keys.
    """

    # --------------------------------------------------------
    # Clean name without spaces
    #
    # Example:
    # "ABC Technologies Pvt Ltd"
    # ->
    # "abctechnologiespvtltd"
    # --------------------------------------------------------

    name_no_space = (
        df["name_clean"]
        .str.replace(" ", "", regex=False)
    )

    # --------------------------------------------------------
    # First token
    # --------------------------------------------------------

    first_token = (
        df["name_clean"]
        .str.split()
        .str[0]
        .fillna("")
    )

    # --------------------------------------------------------
    # Second token
    # --------------------------------------------------------

    second_token = (
        df["name_clean"]
        .str.split()
        .str[1]
        .fillna("")
    )

    # --------------------------------------------------------
    # Address number
    # --------------------------------------------------------

    first_address_number = (
        df["address_numbers"]
        .str.split()
        .str[0]
        .fillna("")
    )

    # --------------------------------------------------------
    # KEY 1
    #
    # Country + first 6 characters of business name
    #
    # Example:
    # india|reliance
    # --------------------------------------------------------

    df["block_name_prefix"] = (
        df["country_clean"]
        + "|"
        + name_no_space.str[:6]
    )

    # --------------------------------------------------------
    # KEY 2
    #
    # Country + first 4 characters of first token
    # --------------------------------------------------------

    df["block_first_token"] = (
        df["country_clean"]
        + "|"
        + first_token.str[:4]
    )

    # --------------------------------------------------------
    # KEY 3
    #
    # Country + first 4 chars of first + second token
    #
    # This is more selective than first token alone.
    # --------------------------------------------------------

    df["block_two_tokens"] = (
        df["country_clean"]
        + "|"
        + first_token.str[:4]
        + "|"
        + second_token.str[:4]
    )

    # --------------------------------------------------------
    # KEY 4
    #
    # Country + address number + first 4 chars of name
    #
    # Useful when names are common.
    # --------------------------------------------------------

    df["block_address_name"] = (
        df["country_clean"]
        + "|"
        + first_address_number
        + "|"
        + first_token.str[:4]
    )

    return df


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Delete previous database if it exists.
    # This prevents old candidates from contaminating a new run.
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    connection = sqlite3.connect(
        DB_FILE
    )

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Candidate index table
    #
    # One row means:
    #
    # blocking key -> candidate entity
    #
    # source tells us whether it came from S2 or S3.
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE candidates (
            block_key TEXT NOT NULL,
            entity_id TEXT NOT NULL
        )
        """
    )

    connection.commit()

    return connection


# ============================================================
# INSERT ONE SOURCE INTO DATABASE
# ============================================================

def index_source(
    connection,
    filename,
    source_name
):

    print()
    print("=" * 70)
    print("INDEXING", source_name)
    print("=" * 70)

    total_rows = 0

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str
    ):

        chunk = preprocess_for_blocking(
            chunk
        )

        chunk = create_blocking_keys(
            chunk
        )

        rows = []

        # ----------------------------------------------------
        # Add each blocking key.
        # ----------------------------------------------------

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            keys = [
                row.block_name_prefix,
                row.block_first_token,
                row.block_two_tokens,
                row.block_address_name,
            ]

            for key in keys:

                # Ignore empty/weak keys.
                if not key:
                    continue

                # Avoid keys where there is no actual name.
                if key.endswith("|"):
                    continue

                rows.append(
                    (
                        key,
                        entity_id
                    )
                )

        # ----------------------------------------------------
        # Insert chunk.
        # ----------------------------------------------------

        connection.executemany(
            """
            INSERT INTO candidates
            (
                block_key,
                entity_id
            )
            VALUES (?, ?)
            """,
            rows
        )

        connection.commit()

        total_rows += len(chunk)

        print(
            f"{source_name}: "
            f"{total_rows:,} records processed"
        )

    print(
        f"Finished {source_name}: "
        f"{total_rows:,} records"
    )


# ============================================================
# CREATE DATABASE INDEX
# ============================================================

def create_database_index(
    connection
):

    print()
    print(
        "Creating SQLite index..."
    )

    connection.execute(
        """
        CREATE INDEX idx_block_key
        ON candidates(block_key)
        """
    )

    connection.commit()

    print(
        "SQLite index created."
    )


# ============================================================
# GENERATE CANDIDATES FOR ONE S1 RECORD
# ============================================================

def get_candidates(
    cursor,
    row
):

    # --------------------------------------------------------
    # Recreate the same blocking keys used for S2/S3.
    # --------------------------------------------------------

    name_no_space = (
        row.name_clean
        .replace(" ", "")
    )

    name_parts = (
        row.name_clean.split()
        if row.name_clean
        else []
    )

    first_token = (
        name_parts[0]
        if len(name_parts) >= 1
        else ""
    )

    second_token = (
        name_parts[1]
        if len(name_parts) >= 2
        else ""
    )

    address_numbers = (
        row.address_numbers.split()
        if row.address_numbers
        else []
    )

    first_address_number = (
        address_numbers[0]
        if address_numbers
        else ""
    )

    country = row.country_clean

    keys = []

    # KEY 1
    if country and name_no_space:
        keys.append(
            country
            + "|"
            + name_no_space[:6]
        )

    # KEY 2
    if country and first_token:
        keys.append(
            country
            + "|"
            + first_token[:4]
        )

    # KEY 3
    if country and first_token and second_token:
        keys.append(
            country
            + "|"
            + first_token[:4]
            + "|"
            + second_token[:4]
        )

    # KEY 4
    if (
        country
        and first_address_number
        and first_token
    ):
        keys.append(
            country
            + "|"
            + first_address_number
            + "|"
            + first_token[:4]
        )

    candidates = set()

    # --------------------------------------------------------
    # Query SQLite for each blocking key.
    # --------------------------------------------------------

    for key in keys:

        cursor.execute(
            """
            SELECT entity_id
            FROM candidates
            WHERE block_key = ?
            """,
            (key,)
        )

        results = cursor.fetchall()

        for result in results:

            candidate_id = result[0]

            # Only S2/S3 can be candidates.
            if candidate_id.startswith(
                ("S2-", "S3-")
            ):
                candidates.add(
                    candidate_id
                )

    return candidates


# ============================================================
# GENERATE FINAL CANDIDATE FILE
# ============================================================

def generate_candidate_file(
    connection
):

    print()
    print("=" * 70)
    print("GENERATING CANDIDATE PAIRS")
    print("=" * 70)

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Make sure output directory exists.
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write output.
    #
    # IMPORTANT:
    # Every Source 1 ID must appear exactly once.
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as output:

        output.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

        processed = 0

        # ----------------------------------------------------
        # Read S1 in chunks.
        # ----------------------------------------------------

        for chunk in pd.read_csv(
            S1_FILE,
            sep="\t",
            chunksize=CHUNK_SIZE,
            dtype=str
        ):

            chunk = preprocess_for_blocking(
                chunk
            )

            chunk = create_blocking_keys(
                chunk
            )

            for row in chunk.itertuples(
                index=False
            ):

                candidates = get_candidates(
                    cursor,
                    row
                )

                # Sort for reproducibility.
                candidate_list = sorted(
                    candidates
                )

                candidate_string = ",".join(
                    candidate_list
                )

                output.write(
                    row.entity_id
                    + "\t"
                    + candidate_string
                    + "\n"
                )

                processed += 1

                if processed % 10_000 == 0:

                    print(
                        f"S1 processed: "
                        f"{processed:,}"
                    )

    print()
    print(
        "Candidate generation completed."
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print(
        "Source 1 records processed:",
        f"{processed:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUSINESS ENTITY CANDIDATE GENERATION")
    print("=" * 70)

    print()
    print("Source 1:", S1_FILE)
    print("Source 2:", S2_FILE)
    print("Source 3:", S3_FILE)
    print("Output:", OUTPUT_FILE)

    # --------------------------------------------------------
    # Create database
    # --------------------------------------------------------

    connection = create_database()

    try:

        # ----------------------------------------------------
        # Index Source 2
        # ----------------------------------------------------

        index_source(
            connection,
            S2_FILE,
            "Source 2"
        )

        # ----------------------------------------------------
        # Index Source 3
        # ----------------------------------------------------

        index_source(
            connection,
            S3_FILE,
            "Source 3"
        )

        # ----------------------------------------------------
        # Create SQLite index
        # ----------------------------------------------------

        create_database_index(
            connection
        )

        # ----------------------------------------------------
        # Generate candidate file
        # ----------------------------------------------------

        generate_candidate_file(
            connection
        )

    finally:

        connection.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
```
