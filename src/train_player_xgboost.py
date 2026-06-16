import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "shots_freeze.parquet"
MODEL_PATH = ROOT / "models" / "xg_model_player_xgboost.joblib"


SHOWCASE_PLAYERS = [
    "Lionel Andrés Messi Cuccittini",
    "Luis Alberto Suárez Díaz",
    "Neymar da Silva Santos Junior",
    "Cristiano Ronaldo dos Santos Aveiro",
    "Kylian Mbappé Lottin",
    "Antoine Griezmann",
    "Zlatan Ibrahimović",
    "Harry Kane",
    "Karim Benzema",
    "Gareth Frank Bale",
]


def main():
    df = pd.read_parquet(DATA_PATH)

    print("Veri shape:", df.shape)

    features = [
        "distance",
        "angle",
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "opp_in_cone",
        "gk_dist_goal",
        "gk_dist_ball",
        "body_part",
        "technique",
        "shot_type",
        "play_pattern",
        "player",
        "position",
    ]

    target = "is_goal"

    X = df[features]
    y = df[target]

    numeric_features = [
        "distance",
        "angle",
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "opp_in_cone",
        "gk_dist_goal",
        "gk_dist_ball",
    ]

    categorical_features = [
        "body_part",
        "technique",
        "shot_type",
        "play_pattern",
        "player",
        "position",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, preds)
    ll = log_loss(y_test, preds)
    brier = brier_score_loss(y_test, preds)

    print("\nModel sonuçları:")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print(f"Brier Score: {brier:.4f}")

    print("\nShowcase oyuncu şut sayıları:")
    print(df[df["player"].isin(SHOWCASE_PLAYERS)]["player"].value_counts())

    artifact = {
        "model": pipe,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "showcase_players": SHOWCASE_PLAYERS,
        "metrics": {
            "roc_auc": auc,
            "log_loss": ll,
            "brier_score": brier,
        },
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
