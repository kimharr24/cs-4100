import re
import os
import nltk
import tqdm
import time
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List

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

def delete_user_mentions(tweet: str) -> str:
    """Removes user mentions from a tweet. (e.g. @someuser hello -> hello)"""
    return re.sub(r"@\w+", "", tweet).strip()


def lowercase(tweet: str) -> str:
    """Converts a tweet to lowercase."""
    return tweet.lower()


def remove_numbers(tweet: str) -> str:
    """Removes numbers from a tweet."""
    return re.sub(r"\d", "", tweet)


def remove_urls(tweet: str) -> str:
    """Given a tweet, removes urls of the form https://... or http://..."""
    return re.sub(r"http[s]?://\S+", "", tweet).strip()


def remove_punctuation(tweet: str) -> str:
    """Removes punctuation from a tweet."""
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    return "".join(char for char in tweet if char not in punctuation)


def remove_website_urls(tweet: str) -> str:
    """Removes website URLs of the form www.website.com."""
    pattern = r"\bwww\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
    cleaned_text = re.sub(pattern, "", tweet)
    return cleaned_text


def get_preprocessed_sentences(sentences: List[str]) -> None:
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
    if not os.path.exists("preprocessed_sentences.pkl"):
        time_start = time.perf_counter()
        sentences = [
            lowercase(
                remove_numbers(
                    remove_punctuation(
                        remove_website_urls(remove_urls(delete_user_mentions(sentence)))
                    )
                )
            )
            for sentence in tqdm(sentences)
        ]
        time_end = time.perf_counter()
        print(f"Time to complete: {time_end - time_start} seconds.")

        time_start = time.perf_counter()
        print("Removing stopwords and performing lemmatization...")
        lemmatizer = Lemmatizer()
        sentences = [lemmatizer.lemmatize(sentence) for sentence in tqdm(sentences)]
        time_end = time.perf_counter()
        print(f"Time to complete: {time_end - time_start} seconds.")

        with open("preprocessed_sentences.pkl", "wb") as f:
            pickle.dump(sentences, f)
    else:
        print("Loading preprocessed sentences from cache")
        with open("preprocessed_sentences.pkl", "rb") as f:
            sentences = pickle.load(f)

    return sentences
