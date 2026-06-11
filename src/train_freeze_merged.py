"""xG Stüdyo — freeze modelini TAM veride (75k) eğitir + adil lift ölçümü."""
import json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

RS, TS = 42, 0.20
BOOL = ["under_pressure", "shot_first_time", "shot_one_on_one"]
CAT = ["body_part", "technique", "shot_type", "play_pattern"]
NUM_BASE = ["distance", "angle"]
NUM_FRZ = ["distance", "angle", "opp_in_cone", "gk_dist_goal", "gk_dist_ball"]

df = pd.read_parquet("data/shots_freeze_merged.parquet")
# GK sentinel 99 -> NaN (XGBoost 'eksik' olarak öğrensin)
for c in ["gk_dist_goal", "gk_dist_ball"]:
    df.loc[df[c] == 99.0, c] = np.nan
print(f"Veri: {len(df)} şut | gol oranı {df['is_goal'].mean():.4f} | StatsBomb xG ort {df['xg_statsbomb'].mean():.4f}")

def pipe(num):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
                            remainder="passthrough")
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                        random_state=RS, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)]), num

def make_X(num):
    X = df[num + BOOL + CAT].copy()
    for b in BOOL: X[b] = X[b].astype(int)
    return X

y = df["is_goal"].astype(int)
idx = np.arange(len(df))
_, _, y_tr, y_te, itr, ite = train_test_split(idx, y, idx, test_size=TS, stratify=y, random_state=RS)
sb_test = df.iloc[ite]["xg_statsbomb"].values

def fit_eval(num, name):
    base, _ = pipe(num)
    X = make_X(num)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X.iloc[itr], y_tr)
    p = model.predict_proba(X.iloc[ite])[:, 1]
    auc, brier = roc_auc_score(y_te, p), brier_score_loss(y_te, p)
    print(f"  {name:28s} AUC {auc:.4f} | Brier {brier:.5f} | ort.tahmin {p.mean():.4f}")
    return model, auc, brier

print("\nAynı split, aynı pipeline — adil karşılaştırma:")
m_base, auc_b, br_b = fit_eval(NUM_BASE, "baseline (distance+angle+...)")
m_frz,  auc_f, br_f = fit_eval(NUM_FRZ,  "freeze (+opp_in_cone,gk...)")
auc_sb = roc_auc_score(y_te, sb_test)
print(f"  {'StatsBomb referans':28s} AUC {auc_sb:.4f}")
print(f"\n>>> TAM VERİDE LİFT: +{auc_f-auc_b:.4f} AUC (freeze - baseline), aynı 15070 şutluk test seti")
print(f">>> StatsBomb'a fark: baseline {auc_sb-auc_b:.4f} -> freeze {auc_sb-auc_f:.4f}")

# OOD guard
guard = {"numeric": {}, "categorical": {}}
for c in NUM_FRZ:
    s = df[c].dropna()
    guard["numeric"][c] = {"min": float(s.min()), "max": float(s.max()),
                           "p01": float(s.quantile(.01)), "p99": float(s.quantile(.99))}
for c in CAT:
    guard["categorical"][c] = sorted(df[c].dropna().unique().tolist())

Path("models").mkdir(exist_ok=True)
joblib.dump(dict(model=m_frz, num_features=NUM_FRZ, bool_features=BOOL, cat_features=CAT,
                 ood=guard, metrics=dict(auc=auc_f, brier=br_f, lift_vs_baseline=auc_f-auc_b,
                 baseline_auc=auc_b, statsbomb_auc=auc_sb), n_rows=len(df), goal_rate=float(y.mean())),
            "models/xg_model_freeze.joblib")
print("\nKAYDEDİLDİ: models/xg_model_freeze.joblib (TAM veri, 75k)")
