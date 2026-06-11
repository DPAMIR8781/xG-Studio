"""
fetch_shots.py — config/leagues.py'deki tüm lig-sezonlardan şutları çeker,
ham şut tablosunu diske (parquet) cache'ler.

Çalıştırma:
    python src/fetch_shots.py

Notlar:
- Her sezon ayrı parquet'e yazılır (data/raw/). İkinci çalıştırmada zaten
  inen sezonlar atlanır → kod kaldığı yerden devam eder, çökse de baştan başlamaz.
- Sonuçta hepsi data/raw_shots.parquet olarak birleştirilir.
- Feature hesabı (mesafe/açı) BURADA YOK; o build_features.py'de.
"""

import sys, time, pathlib
import pandas as pd
from statsbombpy import sb

# --- statsbombpy "credentials not supplied" uyarısını sustur (açık veri kullanıyoruz) ---
import warnings
try:
    from statsbombpy.api_client import NoAuthWarning
    warnings.simplefilter("ignore", NoAuthWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config.leagues import LEAGUES

RAW_DIR = pathlib.Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path("data/raw_shots.parquet")

# StatsBomb şut event'inden almak istediğimiz ham kolonlar
KEEP = [
    "match_id", "minute", "second", "team", "player", "position",
    "location", "under_pressure", "play_pattern",
    "shot_outcome", "shot_body_part", "shot_technique", "shot_type",
    "shot_statsbomb_xg", "shot_first_time", "shot_one_on_one",
]


def fetch_season(lg):
    """Tek bir lig-sezonun tüm şutlarını DataFrame olarak döndürür."""
    cid, sid = lg["competition_id"], lg["season_id"]
    cache = RAW_DIR / f"shots_{cid}_{sid}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)   # zaten inmiş → cache'ten oku

    matches = sb.matches(competition_id=cid, season_id=sid)
    frames = []
    for mid in matches["match_id"]:
        try:
            ev = sb.events(mid)
        except Exception as e:
            print(f"    ! maç {mid} atlandı: {e}")
            continue
        shots = ev[ev["type"] == "Shot"].copy()
        if len(shots):
            cols = [c for c in KEEP if c in shots.columns]
            shots = shots[cols]
            shots["competition_name"] = lg["name"].rsplit(" ", 1)[0]
            shots["season_label"] = lg["name"]
            frames.append(shots)
        time.sleep(0.4)   # rate-limit'e takılmamak için nazik bekleme

    season_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    season_df.to_parquet(cache, index=False)   # sezonu cache'le
    return season_df


def main():
    all_frames = []
    for i, lg in enumerate(LEAGUES, 1):
        print(f"[{i}/{len(LEAGUES)}] {lg['name']} ...", flush=True)
        df = fetch_season(lg)
        print(f"    -> {len(df)} şut", flush=True)
        if len(df):
            all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    full.to_parquet(OUT, index=False)
    print(f"\nBİTTİ. Toplam {len(full)} şut → {OUT}")
    print(f"Lig-sezon sayısı: {len(LEAGUES)} | Gol oranı: {(full['shot_outcome']=='Goal').mean():.1%}")


if __name__ == "__main__":
    main()
