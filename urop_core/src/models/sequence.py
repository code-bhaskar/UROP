"""PyTorch LSTM and GRU sequence models with strictly segmented windows & two-phase Transfer Learning support."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.metrics import regression_metrics


class RecurrentBatteryNet(nn.Module):
    """Encoder-Head recurrent architecture (LSTM or GRU) designed for cross-chemistry transfer learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn_type = rnn_type.upper()
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.encoder = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.encoder(h)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class TrainedSequenceModel:
    model_name: str
    model: RecurrentBatteryNet
    scaler: StandardScaler | None
    seq_len: int
    config: dict
    train_time_s: float
    infer_time_ms: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history: dict[str, list[float]]
    clip_bounds: tuple[float, float]

    def predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_seq, dtype=torch.float32)
            preds = self.model(tensor_x).cpu().numpy()
        return np.clip(preds, self.clip_bounds[0], self.clip_bounds[1])


def build_segmented_sequences(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str],
    seq_len: int,
    transform_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct causal sliding windows strictly inside contiguous segments (never crossing cell/cycle boundaries)."""
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for _, grp in df.group_by(group_cols, maintain_order=True):
        if grp.height < seq_len:
            continue
        X_raw = grp.select(feature_cols).to_numpy().astype(np.float64)
        X_proc = transform_fn(X_raw) if transform_fn is not None else X_raw
        y_arr = grp[target_col].to_numpy().astype(np.float64)

        n_win = len(X_proc) - seq_len + 1
        # Fast vectorized strided windowing
        idx = np.arange(seq_len)[None, :] + np.arange(n_win)[:, None]
        X_list.append(X_proc[idx])
        y_list.append(y_arr[seq_len - 1 :])

    if not X_list:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_seq = np.concatenate(X_list, axis=0).astype(np.float32)
    y_seq = np.concatenate(y_list, axis=0).astype(np.float32)
    return X_seq, y_seq


def train_sequence_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray,
    rnn_type: str = "LSTM",
    hidden_dim: int = 48,
    num_layers: int = 2,
    epochs: int = 18,
    batch_size: int = 128,
    lr: float = 2e-3,
    clip_bounds: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
    scaler: StandardScaler | None = None,
) -> TrainedSequenceModel:
    """Train a PyTorch LSTM or GRU with AdamW, cosine LR schedule, and validation early stopping."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    input_dim = X_train_seq.shape[2]
    seq_len = X_train_seq.shape[1]
    net = RecurrentBatteryNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=0.1,
    )

    effective_bs = min(batch_size, max(16, len(X_train_seq) // 8))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32),
            torch.tensor(y_train_seq, dtype=torch.float32),
        ),
        batch_size=effective_bs,
        shuffle=True,
    )
    Xva_t = torch.tensor(X_val_seq, dtype=torch.float32)
    yva_t = torch.tensor(y_val_seq, dtype=torch.float32)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    history = {"train_loss": [], "val_loss": []}

    t0 = time.perf_counter()
    for _ in range(epochs):
        net.train()
        tr_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = net(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(float(loss.item()))
        scheduler.step()

        net.eval()
        with torch.no_grad():
            v_pred = net(Xva_t)
            v_loss = float(loss_fn(v_pred, yva_t).item())

        history["train_loss"].append(round(float(np.mean(tr_losses)), 6))
        history["val_loss"].append(round(v_loss, 6))

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = copy.deepcopy(net.state_dict())

    t_train = time.perf_counter() - t0
    net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        val_preds = np.clip(net(Xva_t).cpu().numpy(), *clip_bounds)
        ti0 = time.perf_counter()
        test_preds = np.clip(net(torch.tensor(X_test_seq, dtype=torch.float32)).cpu().numpy(), *clip_bounds)
        t_inf_ms = (time.perf_counter() - ti0) * 1000.0

    return TrainedSequenceModel(
        model_name=rnn_type.upper(),
        model=net,
        scaler=scaler,
        seq_len=seq_len,
        config={
            "rnn_type": rnn_type.upper(),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "epochs": epochs,
            "lr": lr,
        },
        train_time_s=round(t_train, 4),
        infer_time_ms=round(t_inf_ms, 3),
        val_metrics=regression_metrics(y_val_seq, val_preds),
        test_metrics=regression_metrics(y_test_seq, test_preds),
        history=history,
        clip_bounds=clip_bounds,
    )


def finetune_sequence_model(
    source_model: TrainedSequenceModel,
    X_tgt_train: np.ndarray,
    y_tgt_train: np.ndarray,
    X_tgt_val: np.ndarray,
    y_tgt_val: np.ndarray,
    head_epochs: int = 8,
    full_epochs: int = 12,
    lr_head: float = 3e-3,
    lr_full: float = 6e-4,
    seed: int = 42,
) -> RecurrentBatteryNet:
    """Two-phase Neural Transfer Learning: Phase 1 adapts head (frozen encoder), Phase 2 unfreezes with low LR."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    net = copy.deepcopy(source_model.model)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tgt_train, dtype=torch.float32),
            torch.tensor(y_tgt_train, dtype=torch.float32),
        ),
        batch_size=min(64, max(8, len(X_tgt_train))),
        shuffle=True,
    )
    Xva_t = torch.tensor(X_tgt_val, dtype=torch.float32)
    yva_t = torch.tensor(y_tgt_val, dtype=torch.float32)

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())

    # Phase 1: Freeze recurrent encoder, adapt input projection + regression head
    for p in net.encoder.parameters():
        p.requires_grad = False
    opt1 = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr_head,
        weight_decay=1e-4,
    )
    for _ in range(head_epochs):
        net.train()
        for bx, by in loader:
            opt1.zero_grad()
            loss = loss_fn(net(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt1.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xva_t), yva_t).item())
            if vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(net.state_dict())

    # Phase 2: Unfreeze all layers with differential learning rates (lower LR on encoder)
    for p in net.parameters():
        p.requires_grad = True
    opt2 = torch.optim.AdamW(
        [
            {"params": net.encoder.parameters(), "lr": lr_full * 0.5},
            {"params": net.input_proj.parameters(), "lr": lr_full},
            {"params": net.head.parameters(), "lr": lr_full * 1.5},
        ],
        weight_decay=1e-4,
    )
    for _ in range(full_epochs):
        net.train()
        for bx, by in loader:
            opt2.zero_grad()
            loss = loss_fn(net(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt2.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xva_t), yva_t).item())
            if vl < best_val:
                best_val = vl
                best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    net.eval()
    return net
