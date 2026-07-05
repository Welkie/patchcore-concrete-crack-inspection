import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import csv
from tqdm import tqdm

from patchcore.datasets.concrete import ConcreteDataset, DatasetSplit

class SupervisedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        return image, label

def build_supervised_datasets(data_path, num_positives, resize=256, imagesize=224):
    # Load basic splits
    train_concrete = ConcreteDataset(data_path, split=DatasetSplit.TRAIN, resize=resize, imagesize=imagesize)
    val_concrete = ConcreteDataset(data_path, split=DatasetSplit.VAL, resize=resize, imagesize=imagesize)
    test_concrete = ConcreteDataset(data_path, split=DatasetSplit.TEST, resize=resize, imagesize=imagesize)

    # Extract images and labels
    train_samples = []
    # Train negatives (1600)
    for i in range(len(train_concrete)):
        item = train_concrete[i]
        train_samples.append((item["image"], 0))

    # Validation negatives & positives
    val_negatives = []
    val_positives = []
    for i in range(len(val_concrete)):
        item = val_concrete[i]
        if item["is_anomaly"] == 1:
            val_positives.append(item["image"])
        else:
            val_negatives.append(item["image"])

    # Add selected positives to training set for supervised data efficiency
    # num_positives is the number of positive samples we allow the supervised model to train on
    for img in val_positives[:num_positives]:
        train_samples.append((img, 1))

    # Remaining positives and all negatives from val form the validation set
    val_samples = []
    for img in val_negatives:
        val_samples.append((img, 0))
    for img in val_positives[num_positives:]:
        val_samples.append((img, 1))

    # If no validation positives are left, we just use a small subset of training positives for validation
    # to avoid empty validation sets when num_positives is large
    if len(val_positives[num_positives:]) == 0 and num_positives > 0:
        # borrow a few for validation, or just skip val validation
        pass

    # Test samples
    test_samples = []
    for i in range(len(test_concrete)):
        item = test_concrete[i]
        test_samples.append((item["image"], item["is_anomaly"]))

    return (
        SupervisedDataset(train_samples),
        SupervisedDataset(val_samples) if len(val_samples) > 0 else None,
        SupervisedDataset(test_samples)
    )

def train_model(model_name, train_loader, val_loader, device, epochs=10, lr=1e-4):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_scores)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": acc,
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    parser = argparse.ArgumentParser(description="Train supervised CNN baseline for crack detection.")
    parser.argument_name = "train_supervised"
    parser.add_argument("--data_path", type=str, required=True, help="Path to concrete dataset folder.")
    parser.add_argument("--model_name", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--num_positives", type=int, default=200, help="Number of positive training images to use.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use.")
    parser.add_argument("--save_path", type=str, default="results/supervised", help="Directory to save results.")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building datasets...")
    train_dataset, val_dataset, test_dataset = build_supervised_datasets(
        args.data_path, args.num_positives
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training {args.model_name} with {args.num_positives} positive samples...")
    model = train_model(args.model_name, train_loader, None, device, epochs=args.epochs, lr=args.lr)

    print("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")

    # Save to CSV
    os.makedirs(args.save_path, exist_ok=True)
    result_file = os.path.join(args.save_path, "results.csv")
    file_exists = os.path.exists(result_file)
    
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Num Positives", "Epochs", "Batch Size", "LR", "Accuracy", "AUROC", "F1", "Precision", "Recall"])
        writer.writerow([
            args.model_name, args.num_positives, args.epochs, args.batch_size, args.lr,
            f"{metrics['accuracy']:.4f}", f"{metrics['auroc']:.4f}", f"{metrics['f1']:.4f}",
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}"
        ])
    print(f"Results appended to {result_file}")

if __name__ == "__main__":
    main()
