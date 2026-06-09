"""
build_features.py — ham şut tablosunu (data/raw_shots.parquet) okur,
modele hazır TEMİZ tabloya çevirir: data/shots_features.parquet

Ne yapar:
- location [x, y]'yi ayrı kolonlara açar
- distance ve angle'ı HAM koordinattan hesaplar (feature engineering burada)
- under_pressure / first_time / one_on_one gibi bool'ları NaN -> False yapar
- hedef değişkeni is_goal üretir
- kategorikleri METİN bırakır (encoding herkesin kendi model pipeline'ında, scikit ile)

Çalıştırma:
    python src/build_features.py

Not: distance/angle "saha birimi"dir (StatsBomb 120x80; metreye çok yakın ama birebir metre değil).
"""

import pathlib
import numpy as np
import pandas as pd

RAW = pathlib.Path("data/raw_shots.parquet")
OUT = pathlib.Path("data/shots_features.parquet")

GOAL_WIDTH = 8.0          # iki direk arası (y=36 ile y=44)
GOAL_X, GOAL_Y = 120.0, 40.0


def split_location(loc):
    """StatsBomb location [x, y] -> (x, y). Liste/array/NaN'a dayanıklı."""
    if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return np.nan, np.nan


def shot_angle(x, y):
    """Gol ağzının şut noktasından gördüğü açı (derece). Yakın+merkez = geniş açı."""
    dx = GOAL_X - x
    num = GOAL_WIDTH * dx
    den = dx**2 + (y - GOAL_Y)**2 - (GOAL_WIDTH / 2.0)**2
    ang = np.arctan2(num, den)
    ang = np.where(ang < 0, ang + np.pi, ang)
    return np.degrees(ang)


def main():
    df = pd.read_parquet(RAW)
    print(f"Ham şut: {len(df)}")

    # 1) location -> x, y
    xy = df["location"].apply(split_location)
    df["x"] = xy.apply(lambda t: t[0])
    df["y"] = xy.apply(lambda t: t[1])
    df = df.dropna(subset=["x", "y"]).copy()

    # 2) feature engineering: mesafe + açı (ham koordinattan)
    df["distance"] = np.hypot(GOAL_X - df["x"], GOAL_Y - df["y"]).round(2)
    df["angle"] = shot_angle(df["x"].values, df["y"].values).round(2)

    # 3) bool'lar: StatsBomb sadece True'yu yazar, gerisi NaN -> False
    for col in ["under_pressure", "shot_first_time", "shot_one_on_one"]:
        df[col] = df[col].fillna(False).astype(bool) if col in df else False

    # 4) hedef
    df["is_goal"] = (df["shot_outcome"] == "Goal").astype(int)

    # 5) temiz şema (kategorikler METİN olarak kalıyor)
    rename = {
        "shot_body_part": "body_part",
        "shot_technique": "technique",
        "shot_type": "shot_type",
        "play_pattern": "play_pattern",
        "shot_outcome": "outcome",
        "shot_statsbomb_xg": "xg_statsbomb",
    }
    df = df.rename(columns=rename)

    keep = [
        "match_id", "competition_name", "season_label", "minute",
        "team", "player", "position",
        "x", "y", "distance", "angle",
        "under_pressure", "shot_first_time", "shot_one_on_one",
        "body_part", "technique", "shot_type", "play_pattern",
        "outcome", "xg_statsbomb", "is_goal",
    ]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    df.to_parquet(OUT, index=False)
    print(f"BİTTİ. Temiz tablo: {len(df)} satır, {df.shape[1]} kolon -> {OUT}")
    print(f"Gol oranı: {df['is_goal'].mean():.1%} | Ortalama xG (StatsBomb): {df['xg_statsbomb'].mean():.3f}")
    print("Kolonlar:", list(df.columns))


if __name__ == "__main__":
    main()
