import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_dataframe
from person3.features import create_pair_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "output"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

MODEL_FILE = OUTPUT_DIR / "person3_model.joblib"


# ============================================================
# SETTINGS
# ============================================================

# Number of S1 records used for training.
# This keeps the first training run manageable.
MAX_S1 = 100_000

# Negative examples per S1 record.
NEGATIVES_PER_S1 = 2


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading training data...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str
    )

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str
    )

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str
    )

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str
    )

    s1 = s1.fillna("")
    s2 = s2.fillna("")
    s3 = s3.fillna("")
    gt = gt.fillna("")

    print("S1:", len(s1))
    print("S2:", len(s2))
    print("S3:", len(s3))
    print("Ground truth:", len(gt))

    return s1, s2, s3, gt


# ============================================================
# CREATE LOOKUP TABLE
# ============================================================

def create_lookup(source2, source3):

    print("Creating entity lookup...")

    combined = pd.concat(
        [source2, source3],
        ignore_index=True
    )

    lookup = {}

    for row in combined.itertuples(index=False):

        lookup[row.entity_id] = {
            "entity_id": row.entity_id,
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country
        }

    print("Lookup entities:", len(lookup))

    return lookup


# ============================================================
# RANDOM NEGATIVE SAMPLING
# ============================================================

def random_negative_ids(
    all_ids,
    positive_ids,
    count
):

    candidates = []

    while len(candidates) < count:

        idx = np.random.randint(
            0,
            len(all_ids)
        )

        candidate = all_ids[idx]

        if candidate not in positive_ids:
            candidates.append(candidate)

    return candidates


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(
    s1_record,
    candidate_record
):

    a = pd.DataFrame([s1_record])
    b = pd.DataFrame([candidate_record])

    return create_pair_features(
        a,
        b
    )


# ============================================================
# BUILD TRAINING DATA
# ============================================================

def build_training_data(
    s1,
    lookup,
    ground_truth
):

    print()
    print("Building training examples...")
    print("Maximum S1 records:", MAX_S1)

    # --------------------------------------------------------
    # Ground truth lookup
    # --------------------------------------------------------

    truth = {}

    for row in ground_truth.itertuples(index=False):

        ids = [
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        ]

        truth[row.source1_entity_id] = set(ids)

    # --------------------------------------------------------
    # All candidate IDs
    # --------------------------------------------------------

    all_ids = list(lookup.keys())

    X = []
    y = []

    # --------------------------------------------------------
    # Sample S1 records
    # --------------------------------------------------------

    sample = s1.head(MAX_S1)

    for counter, row in enumerate(
        sample.itertuples(index=False),
        start=1
    ):

        s1_id = row.entity_id

        positive_ids = truth.get(
            s1_id,
            set()
        )

        # --------------------------------------------
        # Positive examples
        # --------------------------------------------

        for candidate_id in positive_ids:

            candidate = lookup.get(candidate_id)

            if candidate is None:
                continue

            features = calculate_features(
                row._asdict(),
                candidate
            )

            X.append(features)
            y.append(1)

        # --------------------------------------------
        # Negative examples
        # --------------------------------------------

        negative_ids = random_negative_ids(
            all_ids,
            positive_ids,
            NEGATIVES_PER_S1
        )

        for candidate_id in negative_ids:

            candidate = lookup[candidate_id]

            features = calculate_features(
                row._asdict(),
                candidate
            )

            X.append(features)
            y.append(0)

        if counter % 1_000 == 0:

            print(
                f"S1 training records processed: "
                f"{counter:,}"
            )

    X = pd.DataFrame(X)
    y = pd.Series(y)

    print()
    print("Training examples:", len(X))
    print("Positive:", int((y == 1).sum()))
    print("Negative:", int((y == 0).sum()))

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):

    print()
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    print("Model training completed.")

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    s1, s2, s3, gt = load_data()

    lookup = create_lookup(
        s2,
        s3
    )

    X, y = build_training_data(
        s1,
        lookup,
        gt
    )

    model = train_model(
        X,
        y
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()
    print("========================================")
    print("PERSON 3 MODEL READY")
    print("========================================")
    print("Model:", MODEL_FILE)
    print("Features:", list(X.columns))


if __name__ == "__main__":
    main()
