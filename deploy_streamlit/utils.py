"""Fungsi bersama untuk train_export.py dan app.py."""
import json
import joblib
import numpy as np
import pandas as pd

TARGET = "Mental_Health_Score"
NUM = ["Daily_AI_Tool_Usage_Hours", "Daily_Social_Media_Hours", "Sleep_Hours", "Physical_Activity_Hours", "Age"]
CAT = ["Gender", "Education_Level"]
FEATURES = NUM + CAT
EDU_ORDER = ["High School", "College", "University"]


def load_artifacts(folder="artifacts"):
    model = joblib.load(f"{folder}/model.joblib")
    with open(f"{folder}/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    background = pd.read_csv(f"{folder}/background.csv")
    return model, meta, background


def make_codec(meta):
    """Pemetaan kategori -> kode numerik (hanya untuk keperluan SHAP)."""
    gmap = {g: i for i, g in enumerate(meta["categories"]["Gender"])}
    emap = {e: i for i, e in enumerate(meta["categories"]["Education_Level"])}
    return gmap, emap


def to_view(frame, gmap, emap):
    v = frame[FEATURES].copy()
    v["Gender"] = v["Gender"].map(gmap)
    v["Education_Level"] = v["Education_Level"].map(emap)
    return v.astype(float).reset_index(drop=True)


def from_view(arr, gmap, emap):
    inv_g = {v: k for k, v in gmap.items()}
    inv_e = {v: k for k, v in emap.items()}
    d = pd.DataFrame(arr, columns=FEATURES)
    d["Gender"] = d["Gender"].round().astype(int).map(inv_g)
    d["Education_Level"] = d["Education_Level"].round().astype(int).map(inv_e)
    return d


def build_explainer(model, background_raw, gmap, emap, seed=42):
    """SHAP model-agnostik (PermutationExplainer) pada 7 variabel penelitian."""
    import shap
    bg = to_view(background_raw, gmap, emap)
    fn = lambda arr: model.predict(from_view(arr, gmap, emap))
    return shap.PermutationExplainer(fn, shap.maskers.Independent(bg, max_samples=100), seed=seed)


def explain_rows(explainer, X_raw, gmap, emap, max_evals=200):
    view = to_view(X_raw, gmap, emap)
    ex = explainer(view.values, max_evals=max_evals)
    return pd.DataFrame(ex.values, columns=FEATURES), float(np.mean(ex.base_values))
