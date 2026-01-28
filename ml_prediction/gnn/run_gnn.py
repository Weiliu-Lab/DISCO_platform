import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read as ase_read
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if (v is not None and str(v).strip()) else default


# ============================
# Runtime configuration
# ============================
SEED = _env_int("SEED", 42)

TRAIN_RATIO = _env_float("TRAIN_RATIO", 0.8)
VAL_RATIO = _env_float("VAL_RATIO", 0.1)
TEST_RATIO = _env_float("TEST_RATIO", 0.1)

BATCH_SIZE = _env_int("BATCH_SIZE", 32)
CUTOFF = _env_float("CUTOFF", 7.0)

N_TRIALS = _env_int("N_TRIALS", 80)
TRIAL_EPOCHS = _env_int("TRIAL_EPOCHS", 80)
FINAL_EPOCHS = _env_int("FINAL_EPOCHS", 250)

STUDY_NAME = _env_str("STUDY_NAME", "dimenet++")
STORAGE_URL = _env_str("STORAGE_URL", "sqlite:///GNN.db")

STRUCT_DIR = Path(_env_str("STRUCT_DIR", "structures"))
TARGETS_CSV = Path(_env_str("TARGETS_CSV", "targets.csv"))

DATA_PT = Path(_env_str("DATA_PT", "data.pt"))
TRAIN_PT = Path(_env_str("TRAIN_PT", "train.pt"))
VAL_PT = Path(_env_str("VAL_PT", "val.pt"))
TEST_PT = Path(_env_str("TEST_PT", "test.pt"))

CKPT_PATH = Path(_env_str("CKPT_PATH", "dimenet++.pth"))

TRAIN_PRED_CSV = Path(_env_str("TRAIN_PRED_CSV", "gnn_train_predictions.csv"))
TEST_PRED_CSV = Path(_env_str("TEST_PRED_CSV", "gnn_test_predictions.csv"))
METRICS_JSON = Path(_env_str("METRICS_JSON", "gnn_metrics.json"))

LEGACY_TEST_PRED_CSV = Path(_env_str("LEGACY_TEST_PRED_CSV", "gnn_predictions.csv"))

PARITY_SVG = Path(_env_str("PARITY_SVG", "train_test_parity.svg"))


def set_seed(seed: int = 42):
    try:
        from utils import set_seed as _set_seed  # type: ignore

        return _set_seed(seed)
    except Exception:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def convert_to_graph(atoms, y: float, idx: str):
    from utils import convert_to_graph as _convert_to_graph  # type: ignore

    return _convert_to_graph(atoms, y=y, idx=idx)


def load_targets(path: Path) -> dict:
    df = pd.read_csv(path)
    if "id" in df.columns:
        id_col = "id"
    elif "filename" in df.columns:
        id_col = "filename"
    else:
        id_col = df.columns[0]

    if "target" in df.columns:
        y_col = "target"
    else:
        y_col = df.columns[1]

    m: dict[str, float] = {}
    for _, r in df.iterrows():
        k = str(r[id_col]).strip()
        try:
            m[k] = float(r[y_col])
        except Exception:
            continue
    return m


def build_dataset(struct_dir: Path, targets: dict) -> list:
    files = sorted([p for p in struct_dir.glob("*") if p.is_file()])
    dataset = []
    skipped = 0

    for p in files:
        sid = p.stem
        if sid not in targets:
            skipped += 1
            continue
        y = float(targets[sid])
        try:
            atoms = ase_read(str(p))
        except Exception as exc:
            print(f"[WARN] Failed to parse {p.name}: {exc}")
            continue
        dataset.append(convert_to_graph(atoms, y=y, idx=sid))

    print(f"[DATA] structures={len(files)} matched={len(dataset)} skipped_no_target={skipped}")
    return dataset


def split_dataset(dataset: list, seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    n_total = len(dataset)
    if n_total <= 0:
        raise RuntimeError("Dataset is empty after matching targets.")

    ratios_sum = float(train_ratio + val_ratio + test_ratio)
    if ratios_sum <= 0:
        raise ValueError("Sum of ratios must be > 0")

    train_ratio = train_ratio / ratios_sum
    val_ratio = val_ratio / ratios_sum
    test_ratio = test_ratio / ratios_sum

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()

    train_n = int(train_ratio * n_total)
    val_n = int(val_ratio * n_total)
    test_n = n_total - train_n - val_n
    if test_n < 0:
        val_n = max(0, min(val_n, n_total - train_n))
        test_n = n_total - train_n - val_n

    idx_train = perm[:train_n]
    idx_val = perm[train_n : train_n + val_n]
    idx_test = perm[train_n + val_n :]

    train_list = [dataset[i] for i in idx_train]
    val_list = [dataset[i] for i in idx_val]
    test_list = [dataset[i] for i in idx_test]
    return train_list, val_list, test_list


def _model_from_params(params: dict, device: torch.device) -> DimeNetPlusPlus:
    return DimeNetPlusPlus(
        hidden_channels=int(params["hidden_channels"]),
        num_blocks=int(params["num_blocks"]),
        int_emb_size=int(params["int_emb_size"]),
        basis_emb_size=int(params["basis_emb_size"]),
        out_emb_channels=int(params["out_emb_channels"]),
        num_spherical=int(params["num_spherical"]),
        num_radial=int(params["num_radial"]),
        cutoff=float(CUTOFF),
        out_channels=1,
    ).to(device)


def _fit_one(model: DimeNetPlusPlus, train_loader: DataLoader, val_loader: DataLoader, *, epochs: int, lr: float) -> tuple[float, dict]:
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.L1Loss(reduction="sum")

    best_val = float("inf")
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = criterion(out, batch.y.view(-1))
            train_loss += float(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.z, batch.pos, batch.batch).view(-1)
                loss = criterion(out, batch.y.view(-1))
                val_loss += float(loss.item())

        val_mae = val_loss / max(1, len(val_loader.dataset))
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == int(epochs):
            train_mae = train_loss / max(1, len(train_loader.dataset))
            print(f"Epoch {epoch:03d} | Train MAE {train_mae:.6f} | Val MAE {val_mae:.6f}")

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return float(best_val), best_state


@torch.no_grad()
def _predict(model: DimeNetPlusPlus, dataset: list, batch_size: int) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    preds_all: list[float] = []
    ys_all: list[float] = []
    ids_all: list[str] = []
    abs_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.z, batch.pos, batch.batch).view(-1)
        y = batch.y.view(-1)

        preds_all.extend(pred.detach().cpu().tolist())
        ys_all.extend(y.detach().cpu().tolist())
        abs_sum += float(torch.nn.functional.l1_loss(pred, y, reduction="sum").item())

        bidx = getattr(batch, "idx", None)
        if bidx is None:
            ids_all.extend([""] * int(pred.shape[0]))
        else:
            try:
                ids_all.extend([str(x) for x in bidx])
            except Exception:
                ids_all.extend([""] * int(pred.shape[0]))

    mae = abs_sum / max(1, len(dataset))
    return ids_all, np.asarray(ys_all, dtype=float), np.asarray(preds_all, dtype=float), float(mae)


def _save_parity_svg(train_y: np.ndarray, train_pred: np.ndarray, train_mae: float, test_y: np.ndarray, test_pred: np.ndarray, test_mae: float, out_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib not available, skip parity plot: {exc}")
        return False

    all_vals = np.concatenate([train_y, train_pred, test_y, test_pred]).astype(float)
    if all_vals.size == 0:
        return False
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(train_y, train_pred, s=12, alpha=0.6, label=f"Train (MAE={train_mae:.4f})")
    plt.scatter(test_y, test_pred, s=12, alpha=0.6, label=f"Test (MAE={test_mae:.4f})")
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1.5, label="y = x")
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Parity Plot: Train vs Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    return True


def main():
    set_seed(SEED)

    # 1) Build dataset from structures + targets
    targets = load_targets(TARGETS_CSV)
    dataset = build_dataset(STRUCT_DIR, targets)
    torch.save(dataset, str(DATA_PT))

    # 2) Split and persist splits
    train_list, val_list, test_list = split_dataset(dataset, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    print(f"[SPLIT] train={len(train_list)} val={len(val_list)} test={len(test_list)}")
    if len(train_list) == 0 or len(val_list) == 0 or len(test_list) == 0:
        raise RuntimeError("Train/Val/Test split is empty; please adjust ratios or dataset size.")

    torch.save(train_list, str(TRAIN_PT))
    torch.save(val_list, str(VAL_PT))
    torch.save(test_list, str(TEST_PT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ENV] device={device} cuda={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")

    train_loader = DataLoader(train_list, batch_size=int(BATCH_SIZE), shuffle=True)
    val_loader = DataLoader(val_list, batch_size=int(BATCH_SIZE), shuffle=False)

    # 3) Optuna search best hyperparams
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(f"Optuna is required but not available: {exc}")

    optuna.create_study(study_name=STUDY_NAME, storage=STORAGE_URL, direction="minimize", load_if_exists=True)
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_URL)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [128, 256, 512]),
            "num_blocks": trial.suggest_categorical("num_blocks", [4, 5, 6]),
            "int_emb_size": trial.suggest_categorical("int_emb_size", [64, 96, 128]),
            "out_emb_channels": trial.suggest_categorical("out_emb_channels", [128, 192, 256]),
            "basis_emb_size": trial.suggest_categorical("basis_emb_size", [8, 10, 12]),
            "num_spherical": trial.suggest_int("num_spherical", 5, 10),
            "num_radial": trial.suggest_int("num_radial", 5, 10),
        }
        model = _model_from_params(params, device)
        val_mae, _ = _fit_one(model, train_loader, val_loader, epochs=int(TRIAL_EPOCHS), lr=1e-4)
        return float(val_mae)

    print(f"[OPTUNA] study={STUDY_NAME} storage={STORAGE_URL} n_trials={N_TRIALS} trial_epochs={TRIAL_EPOCHS}")
    study.optimize(objective, n_trials=int(N_TRIALS))

    best_params = dict(study.best_trial.params)
    print("[OPTUNA] best_val_mae=", float(study.best_value))
    print("[OPTUNA] best_params=", best_params)

    # 4) Train final model
    best_model = _model_from_params(best_params, device)
    best_val_mae, best_state = _fit_one(best_model, train_loader, val_loader, epochs=int(FINAL_EPOCHS), lr=1e-4)
    torch.save(best_state, str(CKPT_PATH))

    best_model.load_state_dict(torch.load(str(CKPT_PATH), weights_only=False))
    best_model.eval()

    # 5) Predict on train/test and save
    train_ids, train_y, train_pred, train_mae = _predict(best_model, train_list, int(BATCH_SIZE))
    test_ids, test_y, test_pred, test_mae = _predict(best_model, test_list, int(BATCH_SIZE))

    pd.DataFrame({"id": train_ids, "y_true": train_y.tolist(), "y_pred": train_pred.tolist()}).to_csv(TRAIN_PRED_CSV, index=False)
    pd.DataFrame({"id": test_ids, "y_true": test_y.tolist(), "y_pred": test_pred.tolist()}).to_csv(TEST_PRED_CSV, index=False)

    # Backward compatible artifact name
    if LEGACY_TEST_PRED_CSV != TEST_PRED_CSV:
        pd.DataFrame({"id": test_ids, "y_true": test_y.tolist(), "y_pred": test_pred.tolist()}).to_csv(LEGACY_TEST_PRED_CSV, index=False)

    plot_ok = _save_parity_svg(train_y, train_pred, train_mae, test_y, test_pred, test_mae, PARITY_SVG)

    metrics = {
        "model": "dimenet_pp",
        "seed": int(SEED),
        "cutoff": float(CUTOFF),
        "batch_size": int(BATCH_SIZE),
        "ratios": {"train": float(TRAIN_RATIO), "val": float(VAL_RATIO), "test": float(TEST_RATIO)},
        "n_total": int(len(dataset)),
        "n_train": int(len(train_list)),
        "n_val": int(len(val_list)),
        "n_test": int(len(test_list)),
        "optuna": {
            "study_name": STUDY_NAME,
            "storage": STORAGE_URL,
            "n_trials": int(N_TRIALS),
            "trial_epochs": int(TRIAL_EPOCHS),
            "best_value": float(study.best_value),
            "best_params": best_params,
        },
        "final_train": {"epochs": int(FINAL_EPOCHS), "best_val_mae": float(best_val_mae)},
        "parity": {"train_mae": float(train_mae), "test_mae": float(test_mae), "svg": str(PARITY_SVG), "ok": bool(plot_ok)},
        "artifacts": {
            "data_pt": str(DATA_PT),
            "train_pt": str(TRAIN_PT),
            "val_pt": str(VAL_PT),
            "test_pt": str(TEST_PT),
            "ckpt": str(CKPT_PATH),
            "train_pred_csv": str(TRAIN_PRED_CSV),
            "test_pred_csv": str(TEST_PRED_CSV),
            "parity_svg": str(PARITY_SVG),
        },
    }

    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE] wrote", str(METRICS_JSON), str(CKPT_PATH), str(TRAIN_PRED_CSV), str(TEST_PRED_CSV), str(PARITY_SVG))


if __name__ == "__main__":
    main()
