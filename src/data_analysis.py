import pandas as pd


def load_data():

    s1 = pd.read_csv(
        "dataset/train/train_source1.tsv",
        sep="\t"
    )

    s2 = pd.read_csv(
        "dataset/train/train_source2.tsv",
        sep="\t"
    )

    s3 = pd.read_csv(
        "dataset/train/train_source3.tsv",
        sep="\t"
    )

    gt = pd.read_csv(
        "dataset/train/train_ground_truth.tsv",
        sep="\t"
    )

    return s1, s2, s3, gt


s1, s2, s3, gt = load_data()


print("SOURCE 1 INFO")
print(s1.info())

print("\nSOURCE 2 INFO")
print(s2.info())

print("\nSOURCE 3 INFO")
print(s3.info())


print("\nMissing values")

print("\nSource1")
print(s1.isnull().sum())

print("\nSource2")
print(s2.isnull().sum())

print("\nSource3")
print(s3.isnull().sum())


print("\nCountry distribution")

print("Source1")
print(s1["country"].value_counts())

print("\nSource2")
print(s2["country"].value_counts())

print("\nSource3")
print(s3["country"].value_counts())