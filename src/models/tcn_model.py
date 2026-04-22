import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.models.utils import expanding_window_slices


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        # x: (batch, lookback, 1) -> (batch, 1, lookback)
        x = x.transpose(1, 2)
        feat = self.tcn(x)[:, :, -1]
        return self.head(feat).squeeze(-1)


def train_tcn(df, feature_cols, config):
    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]
    lookback = config["model"].get("lookback", 32)
    epochs = config["model"].get("tcn_epochs", 15)
    batch_size = config["model"].get("tcn_batch_size", 64)
    lr = config["model"].get("tcn_lr", 1e-3)
    kernel_size = config["model"].get("tcn_kernel_size", 3)
    dropout = config["model"].get("tcn_dropout", 0.1)
    channels = config["model"].get("tcn_channels", [16, 16, 16])

    torch.manual_seed(42)
    np.random.seed(42)

    series = df["log_return"].values.astype(np.float32)
    future = df[target_col].values.astype(np.float32)

    X_all, y_all = [], []
    for i in range(len(series) - lookback):
        X_all.append(series[i : i + lookback])
        y_all.append(future[i + lookback - 1])

    X_all = np.array(X_all, dtype=np.float32).reshape(-1, lookback, 1)
    y_all = np.array(y_all, dtype=np.float32)

    splits = expanding_window_slices(len(X_all), n_splits, train_fraction=0.6)
    if not splits:
        raise ValueError("Not enough data for walk-forward split.")

    fold_das, all_preds, all_y, fold_sizes = [], [], [], []
    for train_end, test_end in splits:
        X_train, X_test = X_all[:train_end], X_all[train_end:test_end]
        y_train, y_test = y_all[:train_end], y_all[train_end:test_end]
        fold_sizes.append(len(y_test))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        x_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        x_test_t = torch.from_numpy(X_test)

        model = TCNRegressor(channels=channels, kernel_size=kernel_size, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for start in range(0, len(x_train_t), batch_size):
                xb = x_train_t[start : start + batch_size]
                yb = y_train_t[start : start + batch_size]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(x_test_t).cpu().numpy().astype(np.float32)

        preds_dir = (preds > 0).astype(int)
        y_true = (y_test > 0).astype(int)
        da = (preds_dir == y_true).mean()

        fold_das.append(float(da))
        all_preds.extend(preds_dir.tolist())
        all_y.extend(y_true.tolist())

    mean_da = float(np.mean(fold_das)) if fold_das else None
    return {
        "predictions": np.array(all_preds),
        "y_test": np.array(all_y),
        "fold_das": fold_das,
        "mean_fold_da": mean_da,
        "fold_sizes": fold_sizes,
        "n_test_obs": int(sum(fold_sizes)),
    }
