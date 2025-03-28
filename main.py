import pandas as pd
from preprocess import get_preprocessed_sentences

def main():
    df = pd.read_csv(
        "data.csv",
        names=["sentiment", "tweet_id", "data", "query", "user", "text"],
        encoding="ISO-8859-1",
    )
    train_df = pd.DataFrame({
        "text": df['text'].to_numpy(),
        "sentiment": df['sentiment'].to_numpy()
    })
    print(train_df.head())



if __name__ == "__main__":
    main()
