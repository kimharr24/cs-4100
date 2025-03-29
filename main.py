import pandas as pd
import argparse
from data import get_train_test_split

def main():
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.20)
    print(X_train[:15])

if __name__ == "__main__":
    main()
