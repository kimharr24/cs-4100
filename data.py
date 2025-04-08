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
        df = df.iloc[:int(len(df) * 0.07), :][["text", "sentiment"]]
    
        processed_sentences, non_empty_sentence_indices = self.processor.get_preprocessed_sentences(list(df["text"]))
        self.processed_sentences = processed_sentences
        self.labels = df["sentiment"].values[non_empty_sentence_indices]

    def __len__(self) -> int:
        return len(self.processed_sentences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return (
            self.processed_sentences[idx],
            self.labels[idx],
        )
