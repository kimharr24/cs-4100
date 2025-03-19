import pandas as pd


def main():
    df = pd.read_csv(
        "data.csv",
        names=["sentiment", "tweet_id", "data", "query", "user", "text"],
        encoding="ISO-8859-1",
    )
    print(df.head())


if __name__ == "__main__":
    main()
