import re
import os
import nltk
from tqdm import tqdm
import time
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List
import gensim.downloader as api


class Lemmatizer:
    """Removes stopwords and performs lemmatization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english"))

    def lemmatize(self, tweet: str) -> str:
        """Removes stopwords and performs lemmatization on a tweet."""
        words = nltk.word_tokenize(tweet)
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stopwords
        ]
        return " ".join(filtered_words)


class Preprocessor:
    """Preprocesses tweets for sentiment analysis."""

    def __init__(
        self,
        should_lemmatize: bool = True,
        cache_dir: str = "cache/",
    ):
        self.lemmatizer = Lemmatizer()
        self.embedding_model_name = "word2vec-google-news-300"
        self.embedding_model = api.load(self.embedding_model_name)
        self.should_lemmatize = should_lemmatize
        self.cache_dir = cache_dir

    def _get_cache_name(self) -> str:
        """Returns the name of the cache file."""
        cache_name = ""
        if self.should_lemmatize:
            cache_name += "lemmatized"
        return self.cache_dir + self.embedding_model_name + "_" + cache_name + ".pkl"

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
        if not os.path.exists(self._get_cache_name()):
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

            if self.should_lemmatize:
                time_start = time.perf_counter()
                print("Removing stopwords and performing lemmatization...")
                lemmatizer = Lemmatizer()
                sentences = [
                    lemmatizer.lemmatize(sentence) for sentence in tqdm(sentences)
                ]
                time_end = time.perf_counter()
                print(f"Time to complete: {time_end - time_start} seconds.")
            else:
                print("Skipping lemmatization.")

            with open(self._get_cache_name(), "wb") as f:
                pickle.dump(sentences, f)
        else:
            print("Loading preprocessed sentences from cache")
            with open(self._get_cache_name(), "rb") as f:
                sentences = pickle.load(f)

        return sentences
