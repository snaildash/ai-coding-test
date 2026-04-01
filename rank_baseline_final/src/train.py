import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RankingDataset
from features import get_dense_feature_dim
from model import RankModel


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        dense_feat = batch["dense_feat"].to(device)
        query_intent = batch["query_intent"].to(device)
        doc_type = batch["doc_type"].to(device)
        label = batch["label"].to(device)

        pred = model(dense_feat, query_intent, doc_type)
        loss = F.mse_loss(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = RankingDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cpu")
    model = RankModel(dense_dim=get_dense_feature_dim()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"[Epoch {epoch+1}] train_loss={loss:.4f}")


if __name__ == "__main__":
    main()
