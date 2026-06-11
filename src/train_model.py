"""
xG Stüdyo — Model eğitimi (kalibreli)
=====================================
shots_features.parquet'ten okur, kalibrasyonlu bir xG modeli eğitir,
StatsBomb xG referansı ile aynı test setinde karşılaştırır,
dağılım-dışı (OOD) koruması metadata'sı ile birlikte joblib'e kaydeder.

Pipeline: ColumnTransformer (OneHotEncoder handle_unknown='ignore')
          -> XGBoost -> isotonic kalibrasyon (CalibratedClassifierCV, cv=5)

Kullanım:
    python src/train_model.py
    python src/train_model.py --data data/shots_features.parquet --out models/xg_model.joblib

Çıktılar:
    models/xg_model.joblib   : kalibreli model + metadata (feature listeleri, OOD aralıkları, metrikler)
    reports/calibration.png  : kalibrasyon eğrisi (bizim model vs StatsBomb)
    reports/metrics.json     : metrikler

Notlar:
- Penaltılar (shot_type='Penalty') StatsBomb'da sabit 0.7835 xG ile etiketli, gol oranı ~%74.
  Varsayılan olarak DAHİL edilirler ve model bunu shot_type feature'ından öğrenir.
  Sadece açık-oyun xG modeli istersen --open-play-only ile penaltı/serbest vuruşu hariç tut.
- Adil ekip karşılaştırması için: aynı RANDOM_STATE + aynı test_size = aynı test seti.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

# --- Sabitler: ekip karşılaştırması adil olsun diye herkes aynısını kullanmalı ---
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature seti
NUM_FEATURES = ["distance", "angle"]
BOOL_FEATURES = ["under_pressure", "shot_first_time", "shot_one_on_one"]
CAT_FEATURES = ["body_part", "technique", "shot_type", "play_pattern"]
TARGET = "is_goal"
REFERENCE = "xg_statsbomb"  # değerlendirme karşılaştırması için, feature DEĞİL

# Not: 'position' (25 değer) kasten dışarıda — zayıf feature + one-hot şişirir.
#      x, y dışarıda — distance & angle bunlardan türetildi (mükerrer bilgi).


def build_pipeline() -> Pipeline:
    """OneHot(handle_unknown='ignore') + XGBoost. Booleanlar int'e çevrilip passthrough."""
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ],
        remainder="passthrough",  # NUM + BOOL olduğu gibi geçer
    )
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def load_data(path: str, open_play_only: bool) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if open_play_only:
        before = len(df)
        df = df[df["shot_type"] == "Open Play"].copy()
        print(f"[open-play-only] {before} -> {len(df)} satır (penaltı/serbest vuruş vb. çıkarıldı)")
    return df


def make_xy(df: pd.DataFrame):
    X = df[NUM_FEATURES + BOOL_FEATURES + CAT_FEATURES].copy()
    for b in BOOL_FEATURES:
        X[b] = X[b].astype(int)
    y = df[TARGET].astype(int)
    return X, y


def evaluate(name: str, y_true, p) -> dict:
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p)),
        "mean_pred": float(np.mean(p)),
    }


def build_ood_guard(df_train: pd.DataFrame) -> dict:
    """
    Dağılım-dışı koruması metadata'sı (sunum slayt 7).
    Eğitim verisindeki gözlemlenen aralıkları/kategorileri saklar; serving sırasında
    bunun dışındaki absürt şutlarda 'güvenilmez' uyarısı verilebilir.
    """
    guard = {"numeric": {}, "categorical": {}}
    for c in NUM_FEATURES:
        guard["numeric"][c] = {
            "min": float(df_train[c].min()),
            "max": float(df_train[c].max()),
            "p01": float(df_train[c].quantile(0.01)),
            "p99": float(df_train[c].quantile(0.99)),
        }
    for c in CAT_FEATURES:
        guard["categorical"][c] = sorted(df_train[c].unique().tolist())
    return guard


def is_out_of_distribution(shot: dict, guard: dict) -> tuple:
    """
    Tek bir şutu OOD aralığına göre kontrol eder.
    Dönüş: (ood: bool, sebepler: list[str])
    Serving (Streamlit) tarafında modelin yanında çağrılır.
    """
    reasons = []
    for c, rng in guard["numeric"].items():
        v = shot.get(c)
        if v is None:
            continue
        if v < rng["p01"] or v > rng["p99"]:
            reasons.append(f"{c}={v} eğitim aralığı dışında [{rng['p01']:.1f}, {rng['p99']:.1f}]")
    for c, levels in guard["categorical"].items():
        v = shot.get(c)
        if v is not None and v not in levels:
            reasons.append(f"{c}='{v}' eğitimde görülmedi")
    return (len(reasons) > 0, reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shots_features.parquet")
    ap.add_argument("--out", default="models/xg_model.joblib")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic")
    ap.add_argument("--open-play-only", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reports).mkdir(parents=True, exist_ok=True)

    # --- Veri ---
    df = load_data(args.data, args.open_play_only)
    X, y = make_xy(df)
    print(f"Veri: {len(df)} şut | gol oranı {y.mean():.4f} | StatsBomb xG ort. {df[REFERENCE].mean():.4f}")

    # --- Bölme (stratified, sabit seed -> tekrar üretilebilir/adil) ---
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    df_train = df.iloc[idx_tr]
    sb_test = df.iloc[idx_te][REFERENCE].values  # StatsBomb, aynı test seti

    # --- Eğitim: kalibreli model ---
    base = build_pipeline()
    model = CalibratedClassifierCV(base, method=args.calibration, cv=5)
    print(f"Eğitiliyor (XGBoost + {args.calibration} kalibrasyon, cv=5)...")
    model.fit(X_tr, y_tr)

    # --- Değerlendirme (aynı test seti) ---
    p_cal = model.predict_proba(X_te)[:, 1]
    base.fit(X_tr, y_tr)  # kalibrasyonsuz karşılaştırma için
    p_raw = base.predict_proba(X_te)[:, 1]

    results = [
        evaluate("xG_studio_calibrated", y_te, p_cal),
        evaluate("xG_studio_raw_xgb", y_te, p_raw),
        evaluate("statsbomb_reference", y_te, sb_test),
    ]

    print("\n=== Sonuçlar (test seti) ===")
    print(f"{'Model':24s} {'ROC-AUC':>8s} {'Brier':>9s} {'LogLoss':>9s} {'Ort.Tahmin':>11s}")
    for r in results:
        print(f"{r['model']:24s} {r['roc_auc']:8.4f} {r['brier']:9.5f} {r['log_loss']:9.5f} {r['mean_pred']:11.4f}")
    print(f"\nGerçek gol oranı (test): {y_te.mean():.4f}  "
          f"(kalibreli ort. tahmin {p_cal.mean():.4f} buna yakınsa kalibrasyon iyi demektir)")

    # --- Kalibrasyon eğrisi (plot) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        for label, p in [("xG Stüdyo (kalibreli)", p_cal), ("StatsBomb xG", sb_test)]:
            frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Mükemmel kalibrasyon")
        ax.set_xlabel("Tahmin edilen xG (ortalama)")
        ax.set_ylabel("Gerçekleşen gol oranı")
        ax.set_title("Kalibrasyon eğrisi")
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        cal_path = str(Path(args.reports) / "calibration.png")
        fig.savefig(cal_path, dpi=130)
        print(f"\nKalibrasyon eğrisi: {cal_path}")
    except ImportError:
        print("\n(matplotlib yok — kalibrasyon eğrisi atlandı)")

    # --- Kaydet: model + metadata ---
    guard = build_ood_guard(df_train)
    artifact = {
        "model": model,
        "feature_config": {
            "numeric": NUM_FEATURES,
            "boolean": BOOL_FEATURES,
            "categorical": CAT_FEATURES,
            "target": TARGET,
        },
        "ood_guard": guard,
        "metrics": {r["model"]: r for r in results},
        "training": {
            "n_rows": int(len(df)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "goal_rate": float(y.mean()),
            "calibration": args.calibration,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "open_play_only": args.open_play_only,
        },
    }
    joblib.dump(artifact, args.out)
    with open(Path(args.reports) / "metrics.json", "w") as f:
        json.dump(artifact["metrics"], f, indent=2, ensure_ascii=False)
    print(f"Model kaydedildi: {args.out}")
    print(f"Metrikler: {Path(args.reports) / 'metrics.json'}")


if __name__ == "__main__":
    main()
