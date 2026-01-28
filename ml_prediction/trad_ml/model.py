import json
import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

import optuna

# This file is meant to be the only script you edit for Traditional ML runs.
# The selected model can be provided by environment variable MODEL_NAME.
DEFAULT_MODEL_NAME = "__DEFAULT_MODEL__"  # will be replaced by the UI


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


RANDOM_SEED = _env_int("RANDOM_SEED", 42)
N_SPLITS = _env_int("KFOLD_SPLITS", 5)
N_TRIALS = _env_int("OPTUNA_TRIALS", 200)


def load_train_csv(path: str = "train_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("train_data.csv must have at least 2 columns: features + target")

    # Prefer explicit 'target' column; fallback to last column.
    target_col = "target" if "target" in df.columns else df.columns[-1]

    filename_col = "filename" if "filename" in df.columns else None

    y = df[target_col]
    drop_cols = [target_col]
    if filename_col:
        drop_cols.append(filename_col)

    X = df.drop(columns=drop_cols)

    # Keep original filename (if any) for output
    filenames = df[filename_col] if filename_col else None
    return X, y, filenames


def make_gpr_kernel(tag: str):
    kernels = {
        "RBF_0.1": RBF(length_scale=0.1),
        "RBF_1": RBF(length_scale=1.0),
        "RBF_10": RBF(length_scale=10.0),
        "Matern_0.5": Matern(length_scale=1.0, nu=0.5),
        "Matern_1.5": Matern(length_scale=1.0, nu=1.5),
        "RQ": RationalQuadratic(length_scale=1.0, alpha=1.0),
        "RBF+White": RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
    }
    if tag not in kernels:
        raise ValueError(f"Unsupported GPR kernel tag: {tag}")
    return kernels[tag]


def make_model(model_name: str, params: dict):
    name = (model_name or "").lower().strip()

    if name == "rf":
        p = {"random_state": RANDOM_SEED, "n_jobs": -1}
        p.update(params)
        return RandomForestRegressor(**p)

    if name == "svr":
        return SVR(**params)

    if name == "krr":
        return KernelRidge(**params)

    if name == "mlp":
        p = {"max_iter": 2000, "random_state": RANDOM_SEED}
        p.update(params)
        return MLPRegressor(**p)

    if name == "gpr":
        # params are JSON-friendly; kernel is a tag string
        kernel_tag = params.get("kernel", "RBF_1")
        alpha = float(params.get("alpha", 1e-10))
        n_restarts = int(params.get("n_restarts_optimizer", 0))
        return GaussianProcessRegressor(
            kernel=make_gpr_kernel(kernel_tag),
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            optimizer="fmin_l_bfgs_b",
            random_state=RANDOM_SEED,
        )

    if name == "xgb":
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError(
                "Model 'xgb' requires xgboost installed in the runtime environment"
            ) from e

        p = {
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
        }
        p.update(params)
        return xgb.XGBRegressor(**p)

    raise ValueError(f"Unsupported model: {name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    name = model_name.lower().strip()

    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        }

    if name == "svr":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1e0, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            "kernel": kernel,
        }
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
        if kernel in ("rbf", "poly"):
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0])
        return params

    if name == "krr":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        params = {
            "kernel": kernel,
            "alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True),
        }
        if kernel in {"rbf", "poly"}:
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 3)
            params["coef0"] = trial.suggest_float("coef0", 0.0, 5.0)
        return params

    if name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(64, 32), (128, 64), (256, 128), (128, 64, 32)],
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": "adam",
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
        }

    if name == "gpr":
        return {
            "kernel": trial.suggest_categorical(
                "kernel",
                [
                    "RBF_0.1",
                    "RBF_1",
                    "RBF_10",
                    "Matern_0.5",
                    "Matern_1.5",
                    "RQ",
                    "RBF+White",
                ],
            ),
            "alpha": trial.suggest_float("alpha", 1e-12, 1e-4, log=True),
            "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 5),
        }

    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_categorical("subsample", [0.5, 0.6, 0.8, 1.0]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.6, 0.8, 1.0]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"No optuna space defined for model: {name}")


def run_optuna(X, y, model_name: str, n_trials: int):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        params = suggest_params(trial, model_name)
        model = make_model(model_name, params)
        scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        return float(-scores.mean())

    study = optuna.create_study(
        direction="minimize",
        study_name=f"TradML_{model_name}",
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study.best_params, float(study.best_value)


def main():
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).lower().strip()
    X, y, filenames = load_train_csv("train_data.csv")

    best_params = {}
    optuna_best_cv_mae = None

    if N_TRIALS and N_TRIALS > 0:
        best_params, optuna_best_cv_mae = run_optuna(X, y, model_name, N_TRIALS)

    model = make_model(model_name, best_params)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    y_pred_oof = cross_val_predict(model, X, y, cv=kf, n_jobs=1, method="predict")

    metrics = {
        "model": model_name,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "cv_splits": int(N_SPLITS),
        "optuna_trials": int(N_TRIALS),
        "oof_mae": float(mean_absolute_error(y, y_pred_oof)),
        "oof_r2": float(r2_score(y, y_pred_oof)),
        "oof_rmse": float(np.sqrt(mean_squared_error(y, y_pred_oof))),
        "optuna_best_cv_mae": float(optuna_best_cv_mae) if optuna_best_cv_mae is not None else None,
        "best_params": best_params,
    }

    out = pd.DataFrame({"y_true": y, "y_pred": y_pred_oof})
    if filenames is not None:
        out.insert(0, "filename", filenames)
    out.to_csv("results.csv", index=False)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Fit final model on full data
    model.fit(X, y)
    joblib.dump(model, "best_model.joblib")

    print("Done. Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
