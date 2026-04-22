import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.models.utils import expanding_window_slices


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        lookback: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(1, patch_len), stride=(1, stride))
        n_patches = max(1, (lookback - patch_len) // stride + 1)

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, 1) -> (B, 1, 1, L)
        x = x.transpose(1, 2).unsqueeze(2)
        patches = self.unfold(x).transpose(1, 2)
        z = self.patch_proj(patches) + self.pos_emb[:, : patches.size(1), :]
        z = self.encoder(z)
        return self.head(z.mean(dim=1)).squeeze(-1)


def train_patchtst(df, feature_cols, config):
    target_col = "fwd_return"
    n_splits = config["model"]["n_splits"]

    lookback = config["model"].get("lookback", 32)
    epochs = config["model"].get("patchtst_epochs", 15)
    batch_size = config["model"].get("patchtst_batch_size", 64)
    lr = config["model"].get("patchtst_lr", 1e-3)
    patch_len = config["model"].get("patchtst_patch_len", 8)
    stride = config["model"].get("patchtst_stride", 4)
    d_model = config["model"].get("patchtst_d_model", 32)
    n_heads = config["model"].get("patchtst_n_heads", 4)
    n_layers = config["model"].get("patchtst_n_layers", 2)
    dropout = config["model"].get("patchtst_dropout", 0.1)

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

        model = PatchTSTRegressor(
            lookback=lookback,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
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
