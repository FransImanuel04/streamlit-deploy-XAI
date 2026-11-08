"""Melatih (atau memuat) model terbaik lalu mengekspor artefak untuk aplikasi Streamlit.

Contoh:
  python train_export.py --data AI_SocialMedia_Student_Dataset.csv                 # latih ulang MLP seperti pada skripsi
  python train_export.py --data AI_SocialMedia_Student_Dataset.csv --model linear  # Linear Regression
  python train_export.py --data ... --reuse models/model_terbaik_MLP.joblib        # pakai model dari notebook (versi library harus sama)
  python train_export.py --data ... --fast                                         # tanpa tuning (hanya untuk uji cepat)
"""
import argparse, datetime, json, os, platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import TARGET, NUM, CAT, FEATURES, EDU_ORDER, make_codec, build_explainer, explain_rows

RS = 42


def make_pre(scale):
    num = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]), CAT)])


def get_model(name):
    if name == "linear":
        return "Linear Regression", Pipeline([("pre", make_pre(False)), ("model", LinearRegression())]), None
    if name == "rf":
        p = {"model__n_estimators": [200, 300, 500], "model__max_depth": [None, 6, 10, 16],
             "model__min_samples_leaf": [1, 5, 10, 20, 40], "model__max_features": [0.5, 0.75, 1.0]}
        return "Random Forest", Pipeline([("pre", make_pre(False)), ("model", RandomForestRegressor(random_state=RS, n_jobs=-1))]), p
    if name == "xgb":
        from xgboost import XGBRegressor
        p = {"model__n_estimators": [200, 400, 600], "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
             "model__max_depth": [2, 3, 4, 6], "model__subsample": [0.7, 0.85, 1.0],
             "model__colsample_bytree": [0.7, 0.85, 1.0], "model__min_child_weight": [1, 5, 10], "model__reg_lambda": [1, 5, 10]}
        return "XGBoost", Pipeline([("pre", make_pre(False)), ("model", XGBRegressor(random_state=RS, n_jobs=-1, tree_method="hist"))]), p
    p = {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
         "model__learning_rate_init": [1e-3, 5e-3, 1e-2], "model__activation": ["relu", "tanh"]}
    return "MLP", Pipeline([("pre", make_pre(True)),
                            ("model", MLPRegressor(max_iter=600, early_stopping=True, n_iter_no_change=15, random_state=RS))]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="AI_SocialMedia_Student_Dataset.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default="mlp", choices=["mlp", "linear", "rf", "xgb"])
    ap.add_argument("--reuse", default=None, help="jalur model .joblib dari notebook (opsional)")
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--fast", action="store_true", help="lewati tuning")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data)[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    strata = pd.qcut(y, q=3, labels=False)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=strata)
    print(f"Data: {len(df)} baris | latih {len(X_tr)} | uji {len(X_te)}")

    if a.reuse:
        model = joblib.load(a.reuse); name = "Model dari notebook (reuse)"
    else:
        name, pipe, params = get_model(a.model)
        if params is None or a.fast:
            model = pipe.fit(X_tr, y_tr)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=RS)
            rs = RandomizedSearchCV(pipe, params, n_iter=a.n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                                    random_state=RS, n_jobs=-1, refit=True).fit(X_tr, y_tr)
            model = rs.best_estimator_
            print("Hyperparameter terbaik:", rs.best_params_)
    print("Model:", name)

    pred = model.predict(X_te); res = y_te.values - pred
    metrics = dict(MAE=float(mean_absolute_error(y_te, pred)), RMSE=float(np.sqrt(mean_squared_error(y_te, pred))),
                   R2=float(r2_score(y_te, pred)),
                   baseline_RMSE=float(np.sqrt(np.mean((y_te.values - y_tr.mean()) ** 2))))
    print("Metrik uji:", {k: round(v, 4) for k, v in metrics.items()})

    cats = {"Gender": sorted(df["Gender"].unique().tolist()),
            "Education_Level": [e for e in EDU_ORDER if e in set(df["Education_Level"])] or sorted(df["Education_Level"].unique().tolist())}
    c1, c2 = y_tr.quantile([1 / 3, 2 / 3]).tolist()
    meta = {
        "model_name": name, "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "n_total": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "categories": cats,
        "numeric_ranges": {f: {"min": float(df[f].min()), "max": float(df[f].max()), "mean": float(df[f].mean())} for f in NUM},
        "edu_age_ranges": {e: [int(df.loc[df.Education_Level == e, "Age"].min()), int(df.loc[df.Education_Level == e, "Age"].max())] for e in cats["Education_Level"]},
        "target": {"mean": float(y_tr.mean()), "std": float(y_tr.std()), "min": float(y.min()), "max": float(y.max()),
                   "tertile_cuts": [float(c1), float(c2)], "quantiles": [float(v) for v in np.quantile(y_tr, np.linspace(0, 1, 101))]},
        "metrics": metrics,
        "residual_quantiles": {k: float(np.quantile(res, q)) for k, q in [("q05", .05), ("q10", .10), ("q90", .90), ("q95", .95)]},
    }

    bg = X_tr.sample(100, random_state=RS); bg.to_csv(f"{a.out}/background.csv", index=False)
    gmap, emap = make_codec(meta)
    expl = build_explainer(model, bg, gmap, emap)
    sample = X_te.sample(min(300, len(X_te)), random_state=RS)
    sv, base = explain_rows(expl, sample, gmap, emap)
    meta["shap_global"] = {f: float(v) for f, v in sv.abs().mean().sort_values(ascending=False).items()}
    meta["shap_base"] = base; meta["shap_n"] = int(len(sample))

    joblib.dump(model, f"{a.out}/model.joblib")
    with open(f"{a.out}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Artefak tersimpan di:", a.out)
    print("Kepentingan SHAP global:", {k: round(v, 3) for k, v in meta["shap_global"].items()})


if __name__ == "__main__":
    main()
