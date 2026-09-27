# Bussiness_Entity_Hackathon
## Overview

This project addresses business entity resolution across multiple data sources.

The goal is to identify records in Source 2 and Source 3 that correspond to entities in Source 1.

## Team Pipeline

Our pipeline consists of:

1. Data loading and analysis
2. Business name and address preprocessing
3. Candidate generation using blocking
4. Pair-level feature engineering
5. Machine-learning based matching
6. Candidate and matching result generation

## Input Data

The solution expects the following dataset structure:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
Output

The final outputs are:

output/
├── candidate_pairs.tsv
└── matching_results.tsv

candidate_pairs.tsv contains the candidate Source 2/Source 3 entities considered for each Source 1 entity.

matching_results.tsv contains the final matching output for each Source 1 entity.

Source Code

The main implementation is located in:

src/
├── data_loader.py
├── data_analysis.py
├── preprocessing.py
├── candidate.blocking.fast.py
├── person3/
│   ├── features.py
│   └── train_model.py
└── generate_test_candidates.py
Environment

Python 3 is required.

Install dependencies with:

pip install -r requirements.txt
Notes

The dataset contains millions of records, so candidate generation and processing are performed in chunks and use blocking techniques to reduce the number of comparisons.

External business data lookup was not used.
