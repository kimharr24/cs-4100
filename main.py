import pandas as pd
from torch.utils.data import random_split, DataLoader
import argparse
import torch
from data import TwitterDataset
from cnn import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for training")


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    args,
    checkpoint_path="checkpoints/",
):
    print(f"Training model for {args.epochs} epochs")
    model.train()
    for epoch in range(args.epochs):
        loss, steps = 0, 0
        for batch in train_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)

            loss = criterion(outputs, labels)
            loss += loss.item()
            steps += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {(loss / steps):.4f}")

    if args.save_model:
        name = f"model_epochs_{args.epochs}_batch_{args.batch_size}_lr_{args.learning_rate}_sample_{args.sample_percentage}_lemmatize_{args.lemmatize}_max_words_{args.max_word_count}.pth"
        torch.save(
            model.state_dict(),
            f"{checkpoint_path}{name}",
        )
        print(f"Model saved to {checkpoint_path}{name}")


def main():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train the model"
    )
    args.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    args.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    args.add_argument(
        "--sample_percentage",
        type=float,
        default=0.07,
        help="Percentage of data to sample",
    )
    args.add_argument(
        "--lemmatize", action="store_true", help="Whether to lemmatize the text data"
    )
    args.add_argument(
        "--max_word_count",
        type=int,
        default=20,
        help="Maximum number of words in a sentence",
    )
    args.add_argument(
        "--save_model",
        action="store_true",
        help="Whether to save the model after training",
    )

    args = args.parse_args()

    dataset = TwitterDataset(
        should_lemmatize=args.lemmatize,
        sample_percentage=args.sample_percentage,
        max_word_count=args.max_word_count,
    )
    train_dataset, test_dataset = random_split(
        dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CNN(embedding_dim=300).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        args=args,
        checkpoint_path="checkpoints/",
    )


if __name__ == "__main__":
    main()
