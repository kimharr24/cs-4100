import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from preprocess import get_preprocessed_sentences


def get_train_test_split(
    test_size: int = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(
        "data.csv",
        names=["sentiment", "tweet_id", "data", "query", "user", "text"],
        encoding="ISO-8859-1",
    )
    train_df = pd.DataFrame(
        {"text": df["text"].to_numpy(), "sentiment": df["sentiment"].to_numpy()}
    )
    # Pre-process the sentences to remove punctuation, stopwords, etc.
    processed_sentences = get_preprocessed_sentences(list(train_df["text"]))
    processed_train_df = pd.DataFrame(
        {"text": processed_sentences, "sentiment": train_df["sentiment"].to_numpy()}
    )

    # Remove rows where text is empty
    num_empty_rows = processed_train_df[processed_train_df["text"] == ""].shape[0]
    print(f"Removing {num_empty_rows} rows where text is empty...")
    processed_train_df = processed_train_df[processed_train_df["text"] != ""]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        processed_train_df["text"].to_numpy(),
        processed_train_df["sentiment"].to_numpy(),
        test_size=test_size,
        random_state=42,
    )

    return X_train_text, X_test_text, y_train, y_test
