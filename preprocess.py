import re
import os
import gc
import nltk
from tqdm import tqdm
import time
import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List
from embedding_models.googleNewsWord2Vec import get_google_news_embedding_model
from embedding_models.twitterWord2Vec import get_twitter_embedding_model


class Lemmatizer:
    """Removes stopwords and performs lemmatization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, tweet: str) -> str:
        """Performs lemmatization on a tweet."""
        words = nltk.word_tokenize(tweet)
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(filtered_words)


class Preprocessor:
    """Preprocesses tweets for sentiment analysis."""

    def __init__(
        self,
        should_lemmatize: bool = True,
        max_word_count: int = 20,
        embed_model: str = "twitter",
    ):
        """
        Constructor for the pre-processor object.

        Args:
            - should_lemmatize: Whether to perform lemmatization on the tweets.
            - max_word_count: Maximum number of words in a sentence.
            - embed_model: The embedding model to use (twitter or googlenews).
        """
        assert embed_model in ["twitter", "googlenews"]
        self.should_lemmatize = should_lemmatize
        self.lemmatizer = Lemmatizer() if self.should_lemmatize else None
        self.max_word_count = max_word_count

        model, embedding_dim, vocab = (
            get_twitter_embedding_model()
            if embed_model == "twitter"
            else get_google_news_embedding_model()
        )

        self.embedding_model = model
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.stopwords = set(stopwords.words("english"))

    def _delete_user_mentions(self, tweet: str) -> str:
        """Removes user mentions from a tweet. (e.g. @someuser hello -> hello)"""
        return re.sub(r"@\w+", "", tweet).strip()

    def _lowercase(self, tweet: str) -> str:
        """Converts a tweet to lowercase."""
        return tweet.lower()

    def _remove_numbers(self, tweet: str) -> str:
        """Removes numbers from a tweet."""
        return re.sub(r"\d", "", tweet)

    def _remove_urls(self, tweet: str) -> str:
        """Given a tweet, removes urls of the form https://... or http://..."""
        return re.sub(r"http[s]?://\S+", "", tweet).strip()

    def _remove_punctuation(self, tweet: str) -> str:
        """Removes punctuation from a tweet."""
        punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        return "".join(char for char in tweet if char not in punctuation)

    def _remove_website_urls(self, tweet: str) -> str:
        """Removes website URLs of the form www.website.com."""
        pattern = r"\bwww\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
        cleaned_text = re.sub(pattern, "", tweet)
        return cleaned_text

    def get_preprocessed_sentences(self, sentences: List[str]) -> List[str]:
        """
        Given a list of sentences, performs the following pre-processing steps:

        1. Removes user mentions
        2. Removes any URLs
        3. Removes punctuation
        4. Lowercases all sentences
        5. Removes numbers
        6. Removes stopwords
        7. Performs lemmatization
        """
        print("Performing pre-processing steps...")
        time_start = time.perf_counter()
        sentences = [
            self._lowercase(
                self._remove_numbers(
                    self._remove_punctuation(
                        self._remove_website_urls(
                            self._remove_urls(self._delete_user_mentions(sentence))
                        )
                    )
                )
            )
            for sentence in tqdm(sentences)
        ]
        time_end = time.perf_counter()
        print(f"Time to complete: {time_end - time_start} seconds.")

        print("Removing stopwords...")
        filtered_sentences = []
        for sentence in tqdm(sentences):
            words = sentence.split()
            filtered_words = [word for word in words if word not in self.stopwords]
            filtered_sentences.append(" ".join(filtered_words))
        sentences = filtered_sentences

        if self.should_lemmatize:
            time_start = time.perf_counter()
            print("Performing lemmatization...")
            lemmatizer = Lemmatizer()
            sentences = [lemmatizer.lemmatize(sentence) for sentence in tqdm(sentences)]
            time_end = time.perf_counter()
            print(f"Time to complete: {time_end - time_start} seconds.")
        else:
            print("Skipping lemmatization.")

        time_start = time.perf_counter()
        print("Creating embeddings...")
        all_embeddings = []
        non_empty_sentence_indices = []
        for sentence_index, sentence in tqdm(
            enumerate(sentences), total=len(sentences)
        ):
            if sentence != "":
                words = sentence.split()
                embeddings = torch.zeros((self.max_word_count, self.embedding_dim))
                for i, word in enumerate(words):
                    if i >= self.max_word_count:
                        break
                    # If the word is in the embedding model vocabulary, get its vector
                    if word in self.vocab:
                        embeddings[i] = torch.tensor(
                            self.embedding_model[word], dtype=torch.float16
                        )
                    # If the word is not in the vocabulary, fill with zeros
                    else:
                        embeddings[i] = torch.zeros(self.embedding_dim)
                # If the number of words is less than max_word_count, fill the rest with zeros as padding
                if len(words) < self.max_word_count:
                    for i in range(len(words), self.max_word_count):
                        embeddings[i] = torch.zeros(self.embedding_dim)
                all_embeddings.append(embeddings)
                non_empty_sentence_indices.append(sentence_index)

        time_end = time.perf_counter()
        print(f"Time to complete: {time_end - time_start} seconds.")

        del sentences
        gc.collect()

        return all_embeddings, non_empty_sentence_indices
