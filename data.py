import pandas as pd
import torch
from typing import Tuple
from preprocess import Preprocessor


class TwitterDataset(torch.utils.data.Dataset):
    """Custom dataset for Twitter sentiment analysis."""

    def __init__(self, should_lemmatize: bool = True) -> None:
        self.processor = Preprocessor(should_lemmatize=should_lemmatize)
        df = pd.read_csv(
            "data.csv",
            names=["sentiment", "tweet_id", "data", "query", "user", "text"],
            encoding="ISO-8859-1",
        )
        train_df = pd.DataFrame(
            {"text": df["text"].to_numpy(), "sentiment": df["sentiment"].to_numpy()}
        )
        # Pre-process the sentences to remove punctuation, stopwords, etc.
        processed_sentences = self.processor.get_preprocessed_sentences(list(train_df["text"]))
        processed_train_df = pd.DataFrame(
            {"text": processed_sentences, "sentiment": train_df["sentiment"].to_numpy()}
        )

        # Remove rows where text is empty
        num_empty_rows = processed_train_df[processed_train_df["text"] == ""].shape[0]
        print(f"Removing {num_empty_rows} rows where text is empty...")
        self.processed_train_df = processed_train_df[processed_train_df["text"] != ""]

    def __len__(self) -> int:
        return len(self.processed_train_df)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (
            self.processed_train_df["text"].values[idx],
            self["sentiment"].values[idx],
        )
