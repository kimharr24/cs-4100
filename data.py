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
        df = df.sample(frac=0.07, random_state=42)[["text", "sentiment"]]

        print(df["sentiment"].describe())
    
        processed_sentences, non_empty_sentence_indices = self.processor.get_preprocessed_sentences(list(df["text"]))
        self.processed_sentences = processed_sentences
        self.labels = df["sentiment"].values[non_empty_sentence_indices]
        self.labels[self.labels == 4] = 1

    def __len__(self) -> int:
        """"Returns the size of the dataset."""
        return len(self.processed_sentences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Returns a single item from the dataset.

        Returns:
            - processed_sentences: The processed sentence at the given index of shape (embedding_dim, max_word_count).
            - labels: The label at the given index of shape (1, ).
        """
        return (
            self.processed_sentences[idx].permute(1, 0),
            torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0),
        )
