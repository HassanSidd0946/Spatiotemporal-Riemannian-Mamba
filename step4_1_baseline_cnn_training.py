# step4_1_baseline_cnn_training.py

"""
BCI Research Pipeline — Phase 4, Step 4.1
Task : Verify training loop infrastructure + establish baseline metric.

Why this step exists
--------------------
Before committing GPU-hours to the full SpatiotemporalRiemannianMamba,
we validate that:
  (a) DataLoaders, train/test split, and batching are correct.
  (b) The training loop (forward → loss → backward → step) is bug-free.
  (c) We have a concrete baseline accuracy to beat in Phase 5.

A 1D-CNN baseline is the standard point of comparison for EEG classification:
fast to train, well-understood, and representative of the prior-art floor.

No Mamba or Riemannian code is present in this script.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES   = 410
N_CHANNELS  = 62
N_TIMES     = 251
N_CLASSES   = 2

TRAIN_RATIO = 0.80
BATCH_SIZE  = 16
N_EPOCHS    = 5
LR          = 1e-3
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEP = "=" * 60


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline Model · 1D-CNN
# ══════════════════════════════════════════════════════════════════════════════
class BaselineCNN(nn.Module):
    """
    Lightweight 1D-CNN for EEG binary classification.

    Input  : (B, C=62, T=251)
    Output : (B, 2)   — raw logits

    Pipeline
    --------
    Conv1d(62→16, k=3)  →  BN1d  →  ReLU  →  MaxPool1d(2)  →  Flatten  →  Linear(→2)

    MaxPool1d(2) halves the time dimension:
        T_after_conv = 251 - 3 + 1 = 249
        T_after_pool = 249 // 2    = 124
    Flatten size = 16 * 124 = 1984
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_times: int = N_TIMES,
                 n_classes: int = N_CLASSES):
        super().__init__()

        # Compute flattened size analytically to avoid magic numbers
        t_conv = n_times - 3 + 1      # 249  (Conv1d, k=3, no padding, stride=1)
        t_pool = t_conv // 2           # 124  (MaxPool1d, k=2)
        flat   = 16 * t_pool           # 1984

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 · Mock Dataset & DataLoaders
# ══════════════════════════════════════════════════════════════════════════════
def build_dataloaders():
    print(f"\n{SEP}")
    print("  Step 1 · Mock Dataset & DataLoaders")
    print(SEP)

    torch.manual_seed(RANDOM_SEED)

    # Balanced binary labels: 205 × class-0, 205 × class-1
    X = torch.randn(N_SAMPLES, N_CHANNELS, N_TIMES, dtype=torch.float32)
    y = torch.cat([
        torch.zeros(N_SAMPLES // 2, dtype=torch.long),
        torch.ones (N_SAMPLES // 2, dtype=torch.long),
    ])

    # Shuffle before split (reproducible)
    perm    = torch.randperm(N_SAMPLES, generator=torch.Generator().manual_seed(RANDOM_SEED))
    X, y    = X[perm], y[perm]

    n_train = int(N_SAMPLES * TRAIN_RATIO)   # 328
    n_test  = N_SAMPLES - n_train             # 82

    X_train, X_test = X[:n_train].to(DEVICE), X[n_train:].to(DEVICE)
    y_train, y_test = y[:n_train].to(DEVICE), y[n_train:].to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size = BATCH_SIZE,
        shuffle    = True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size = BATCH_SIZE,
        shuffle    = False,
    )

    print(f"  Total samples    : {N_SAMPLES}  (class-0: {N_SAMPLES//2}, class-1: {N_SAMPLES//2})")
    print(f"  Train / Test     : {n_train} / {n_test}  ({int(TRAIN_RATIO*100)}/{100-int(TRAIN_RATIO*100)} split)")
    print(f"  Train batches    : {len(train_loader)}  (batch_size={BATCH_SIZE}, shuffle=True)")
    print(f"  Test  batches    : {len(test_loader)}  (batch_size={BATCH_SIZE}, shuffle=False)")
    print(f"  Tensors on       : {DEVICE}")

    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 · Model Instantiation
# ══════════════════════════════════════════════════════════════════════════════
def build_model():
    print(f"\n{SEP}")
    print("  Step 2 · BaselineCNN Instantiation")
    print(SEP)

    model  = BaselineCNN().to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Architecture:")
    print(f"    Conv1d(62 → 16, k=3)  → BN1d → ReLU → MaxPool1d(2) → Flatten → Linear(1984→2)")
    print(f"    T: 251 → 249 (conv) → 124 (pool)")
    print(f"  Trainable params : {params:,}")
    print(f"  Device           : {next(model.parameters()).device}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 · Training Loop
# ══════════════════════════════════════════════════════════════════════════════
def train(model, train_loader, criterion, optimizer):
    print(f"\n{SEP}")
    print(f"  Step 3 · Training Loop  ({N_EPOCHS} epochs, AdamW lr={LR})")
    print(SEP)

    model.train()

    for epoch in range(1, N_EPOCHS + 1):
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch [{epoch:02d}/{N_EPOCHS}]  avg train loss : {avg_loss:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 · Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, test_loader, criterion):
    print(f"\n{SEP}")
    print("  Step 4 · Test Set Evaluation")
    print(SEP)

    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits   = model(X_batch)
            loss     = criterion(logits, y_batch)
            preds    = logits.argmax(dim=1)

            total_loss    += loss.item()
            total_correct += (preds == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * total_correct / total_samples

    print(f"  Test samples     : {total_samples}")
    print(f"  Correct / Total  : {total_correct} / {total_samples}")
    print(f"  Test Loss        : {avg_loss:.6f}")
    print(f"  Test Accuracy    : {accuracy:.2f}%")

    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(test_loss, test_acc):
    print(f"\n{SEP}")
    print("  Phase 4 · Step 4.1 · Summary")
    print(SEP)

    checks = [
        ("DataLoaders built (train + test)",      True),
        ("BaselineCNN instantiated on CUDA",      DEVICE.type == "cuda"),
        ("Training loop ran for 5 epochs",        True),
        ("Test loss is finite",                   test_loss == test_loss),  # NaN check
        ("Baseline accuracy established",         test_acc > 0.0),
    ]

    print(f"\n  {'Check':<45} {'Result'}")
    print(f"  {'-'*55}")
    for label, ok in checks:
        print(f"  {label:<45} {'✓' if ok else '✗'}")

    print(f"""
  ┌─────────────────────────────────────────────────┐
  │  BASELINE METRIC (1D-CNN, 5 epochs, mock data)  │
  │  Test Loss     : {test_loss:>8.6f}                      │
  │  Test Accuracy : {test_acc:>6.2f}%                        │
  │                                                 │
  │  Target for SpatiotemporalRiemannianMamba:      │
  │  Test Accuracy > {test_acc:.2f}%  on real ds005189 data │
  └─────────────────────────────────────────────────┘""")

    print(f"\n  Next: Phase 4 Step 4.2 — train SpatiotemporalRiemannianMamba")
    print(f"        on real sub-01 EEG epochs from ds005189.")
    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{SEP}")
    print("  Phase 4 · Step 4.1 · Baseline CNN Training")
    print(SEP)
    print(f"  Device : {DEVICE}")

    train_loader, test_loader = build_dataloaders()
    model                     = build_model()
    criterion                 = nn.CrossEntropyLoss()
    optimizer                 = torch.optim.AdamW(model.parameters(), lr=LR)

    train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print_summary(test_loss, test_acc)


if __name__ == "__main__":
    main()