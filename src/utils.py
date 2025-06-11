import pandas as pd

def encode_labels(df):
    """Convert 'Biased' and 'Unbiased' to 1 and 0."""
    df["label_encoded"] = df["label"].map({"Biased": 1, "Unbiased": 0})
    return df
