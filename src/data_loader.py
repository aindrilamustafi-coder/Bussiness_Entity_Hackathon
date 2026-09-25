import pandas as pd
import os


BASE_PATH = "dataset"


def load_train_data():

    source1 = pd.read_csv(
        os.path.join(BASE_PATH, "train", "train_source1.tsv"),
        sep="\t"
    )

    source2 = pd.read_csv(
        os.path.join(BASE_PATH, "train", "train_source2.tsv"),
        sep="\t"
    )

    source3 = pd.read_csv(
        os.path.join(BASE_PATH, "train", "train_source3.tsv"),
        sep="\t"
    )

    ground_truth = pd.read_csv(
        os.path.join(BASE_PATH, "train", "train_ground_truth.tsv"),
        sep="\t"
    )

    return source1, source2, source3, ground_truth



if __name__ == "__main__":

    s1, s2, s3, gt = load_train_data()


    print("SOURCE 1")
    print(s1.head())

    print("\nSOURCE 2")
    print(s2.head())

    print("\nSOURCE 3")
    print(s3.head())

    print("\nGROUND TRUTH")
    print(gt.head())


    print("\nSizes:")
    print("Source1:", s1.shape)
    print("Source2:", s2.shape)
    print("Source3:", s3.shape)
    print("Ground Truth:", gt.shape)