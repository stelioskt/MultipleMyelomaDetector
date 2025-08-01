import os
import glob
import logging
import csv
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split

# === User-defined parameters === 
GRAPH_DIR = "Dataset/Graphs"        # folder containing .pt graph files
LOG_DIR = "GCN_logs"        # where logs, CSV, and checkpoints go
EPOCHS = 100
BATCH_SIZE = 4
HIDDEN_DIM = 32
LEARNING_RATE = 1e-3
TEST_RATIO = 0.2        # fraction of graphs to reserve for testing
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# =================== Model definition ===================

#TODO: Architecture to be defined

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# =================== Training & Evaluation routines ===================
#TODO: Loss function to be defined, optimizer to be defined, scheduler to be defined
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes
    return total_loss / total_nodes

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, preds = [], []
    total_loss = 0.0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(logits, batch.y)
        total_loss += loss.item() * batch.num_nodes
        total_nodes += batch.num_nodes

        ys.append(batch.y.cpu())
        preds.append(logits.argmax(dim=1).cpu())

    ys = torch.cat(ys).numpy()
    preds = torch.cat(preds).numpy()
    accuracy = (ys == preds).mean()
    precision = precision_score(ys, preds, zero_division=0)
    recall    = recall_score(ys, preds, zero_division=0)
    f1        = f1_score(ys, preds, zero_division=0)
    avg_loss  = total_loss / total_nodes if total_nodes > 0 else 0.0

    return avg_loss, precision, recall, f1, accuracy

# =================== Main execution ===================

def main():
    # Setup logging
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting training")

    # Load graphs
    graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, "*.pt")))
    dataset = [torch.load(f, weights_only=False) for f in graph_files]
    logging.info(f"Loaded {len(dataset)} graph(s) from {GRAPH_DIR}")

    # Split train/test by graph
    n_test  = int(len(dataset) * TEST_RATIO)
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    logging.info(f"Train graphs: {n_train}, Test graphs: {n_test}")

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model, optimizer, loss with class weights
    in_ch = dataset[0].num_node_features
    model = GCN(in_ch, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Compute class weights from all training nodes
    all_labels = torch.cat([d.y for d in train_ds]).long()
    counts = torch.bincount(all_labels)
    weights = (counts.sum() / (2 * counts.float())).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    #TODO: Include segmentation metrics

    # Prepare metrics CSV
    metrics_csv = os.path.join(LOG_DIR, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"])

    best_f1 = 0.0
    best_model_path = os.path.join(LOG_DIR, "best_model.pt")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, prec, rec, f1, accuracy = eval_epoch(model, test_loader, DEVICE)

        logging.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Acc: {accuracy:.3f} P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f}"
        )

        # Append metrics
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, test_loss, prec, rec, f1])

        # Checkpoint best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  New best model saved (F1={best_f1:.3f})")

    logging.info("Training complete.")
    logging.info(f"Best model: {best_model_path} (F1={best_f1:.3f})")

if __name__ == "__main__":
    main()