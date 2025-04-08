import pandas as pd
from torch.utils.data import random_split, DataLoader
import argparse
from data import TwitterDataset
from cnn import CNN


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    num_epochs=100,
    checkpoint_path="checkpoints/",
):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            texts, labels = batch
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
    dataset = TwitterDataset()
    train_dataset, test_dataset = random_split(
        dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":
    main()
