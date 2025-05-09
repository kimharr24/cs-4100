import pandas as pd
from torch.utils.data import random_split, DataLoader
import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report
from data import TwitterDataset
from models.cnn import CNN, CNNFromScratch
from models.mlp import MLP
from preprocess import Preprocessor


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
    """Train the model on the training dataset."""
    print(f"Training model for {args.epochs} epochs")
    model.train()
    for epoch in range(args.epochs):
        total_loss, steps = 0, 0
        for batch in train_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            steps += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {(total_loss / steps):.4f}")

    if args.save_model:
        name = f"{args.model}_epochs_{args.epochs}_batch_{args.batch_size}_lr_{args.learning_rate}_sample_{args.sample_percentage}_lemmatize_{args.lemmatize}_max_words_{args.max_word_count}.pth"
        torch.save(
            model.state_dict(),
            f"{checkpoint_path}{name}",
        )
        print(f"Model saved to {checkpoint_path}{name}")


def test_model(model, test_loader, args):
    """Test the model on the test dataset."""
    print("Testing model...")
    model.eval()

    all_labels, all_predictions = [], []
    with torch.no_grad():
        for batch in test_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)

            if args.should_reverse:
                texts = torch.flip(texts, [2])

            outputs = model(texts)
            predicted = torch.sigmoid(outputs).round()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Negative Sentiment", "Positive Sentiment"],
        )
    )


def evaluate_samples(model, sentences, args) -> None:
    """Evaluate the model on a list of sentences."""
    model.eval()
    with torch.no_grad():
        processed_sentences, non_empty_indices = Preprocessor(
            should_lemmatize=args.lemmatize,
            max_word_count=args.max_word_count,
            embed_model=args.embed_model,
        ).get_preprocessed_sentences(sentences)

        assert len(non_empty_indices) == len(
            processed_sentences
        ), "Should not have empty sentences."

        outputs = model(
            torch.stack(processed_sentences, dim=0).permute(0, 2, 1).to(device)
        )
        predictions = torch.sigmoid(outputs).round().cpu().numpy()

        for idx, sentence in enumerate(sentences):
            print(
                f"Sentence: {sentence} | Predicted Sentiment: {'positive' if predictions[idx] == 1 else 'negative'}"
            )


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
    args.add_argument(
        "--embed_model",
        type=str,
        default="twitter",
        help="Embedding model to use (googlenews or twitter)",
    )
    args.add_argument(
        "--model",
        type=str,
        default="cnn",
        help="Model to use (cnn or mlp)",
    )
    args.add_argument(
        "--evaluate_model",
        action="store_true",
        help="Whether to evaluate the model on example sentences.",
    )
    args.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint",
    )
    args.add_argument(
        "--should_reverse",
        action="store_true",
        help="Whether to reverse the order of the words in the sentence",
    )
    args.add_argument(
        "--from_scratch",
        action="store_true",
        help="Whether to train the CNN model from scratch",
        default=False,
    )

    args = args.parse_args()
    embedding_dim = 400 if args.embed_model == "twitter" else 300

    if args.from_scratch:
        model = CNNFromScratch(embedding_dim=embedding_dim).to(device)
    elif args.model == "cnn":
        model = CNN(embedding_dim=embedding_dim).to(device)
    elif args.model == "mlp":
        model = MLP(embedding_dim=embedding_dim, max_word_count=args.max_word_count).to(
            device
        )
    else:
        raise ValueError(
            f"Model {args.model} not supported. Choose between 'cnn' and 'mlp'."
        )

    if args.evaluate_model:
        assert args.checkpoint_path, "Checkpoint path is required for evaluation."
        model.load_state_dict(torch.load(args.checkpoint_path))

        print(f"Loaded model from {args.checkpoint_path}")
        sentences = [
            "I love this product!",
            "This is the worst :(",
            "I am not happy",
            "Initially, I said I am happy but actually I am not",
        ]
        evaluate_samples(model, sentences, args)
    else:
        dataset = TwitterDataset(
            should_lemmatize=args.lemmatize,
            sample_percentage=args.sample_percentage,
            max_word_count=args.max_word_count,
            embed_model=args.embed_model,
        )

        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            dataset,
            [int(0.8 * len(dataset)), int(0.2 * len(dataset))],
            generator=generator,
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

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
        test_model(model, test_loader, args)


if __name__ == "__main__":
    main()
