import gensim.downloader as api

def get_google_news_embedding_model():
    """
    Load the Word2Vec model and return it along with the embedding dimension and vocabulary.

    Returns:
        - model: The loaded Word2Vec model.
        - embedding_dim: The dimensionality of the word vectors.
        - vocab: The vocabulary of the model.
    """
    print("Loading the Google News Word2Vec model, this can take some time...")
    model = api.load("word2vec-google-news-300")

    embedding_dim = model.vector_size
    vocab = model.key_to_index
    return model, embedding_dim, vocab