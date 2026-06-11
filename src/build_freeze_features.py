"""
xG Stüdyo — freeze_frame feature extraction
============================================
StatsBomb Open Data'daki shot.freeze_frame alanından geometrik feature türetir.
Bunlar StatsBomb'un kendi xG modelinin üstünlüğünün kaynağı: şut anında sahadaki
oyuncu konumları. Mesafe+açıya bunları eklemek (La Liga 5 sezon, ~4.300 şut,
6×5-fold CV) baseline'ı +0.0115 AUC iyileştirdi — gürültü değil (±0.0014).

Üretilen feature'lar (önem sırasına yakın):
  opp_in_cone   : top-kale üçgeni (şut konisi) içindeki rakip sayısı   [GÜÇLÜ]
  gk_dist_ball  : kalecinin topa mesafesi                              [GÜÇLÜ]
  gk_dist_goal  : kalecinin kale merkezine mesafesi (çizgide mi, çıkmış mı) [ORTA]

Denenip ELENENLER (bu örneklemde sinyal taşımadı, gürültü ekledi):
  dist_near_opp (en yakın rakibe mesafe), n_opp_3m (3m içindeki rakip sayısı).
  Tutmak istersen ENABLE_WEAK=True yap.

Kullanım — durum kartındaki fetch akışına ek olarak:
    from build_freeze_features import build_season
    df = build_season(competition_id=11, season_id=4)   # tek sezon
    # veya tüm config/leagues.py listesi için döngüyle çağır, concat et.

NOT: Bu modül fetch_shots.py'ın YERİNE değil, ONA EK. fetch_shots.py zaten şutları
çekiyor; tek değişiklik: events satırındaki 'shot_freeze_frame' kolonunu
freeze_features() ile 3 kolona çevirip mevcut feature tablosuna eklemek.
Sonra build_features.py çıktısına bu 3 kolon eklenir, model script'i num feature
listesine ['opp_in_cone','gk_dist_ball','gk_dist_goal'] eklenir — gerisi aynı.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# StatsBomb saha geometrisi (120x80). Kale: x=120, direkler y=36 ve y=44.
GOAL = np.array([120.0, 40.0])
POST1 = np.array([120.0, 36.0])
POST2 = np.array([120.0, 44.0])
GOAL_WIDTH = 8.0
ENABLE_WEAK = False  # elenen zayıf feature'ları da üretmek istersen True


def subtended_angle(x: float, y: float) -> float:
    """Şut noktasından kalenin gerdiği açı (derece). Yakın+geniş = büyük açı."""
    a = np.hypot(120 - x, 36 - y)
    b = np.hypot(120 - x, 44 - y)
    cosv = np.clip((a * a + b * b - GOAL_WIDTH ** 2) / (2 * a * b + 1e-9), -1, 1)
    return float(np.degrees(np.arccos(cosv)))


def _in_triangle(p, a, b, c) -> bool:
    """p noktası, (a,b,c) üçgeninin içinde mi? (işaret testi)"""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def freeze_features(location, freeze_frame) -> dict:
    """
    Tek bir şutun freeze_frame'inden feature dict döndürür.
    freeze_frame: [{'location':[x,y], 'teammate':bool, 'position':{'name':..}}, ...]
    Eksikse NaN/sentinel (99.0) döner; model NaN'i handle eder ya da impute edilir.
    """
    out = {"opp_in_cone": np.nan, "gk_dist_goal": 99.0, "gk_dist_ball": 99.0}
    if ENABLE_WEAK:
        out.update({"dist_near_opp": 99.0, "n_opp_3m": np.nan})
    if not isinstance(freeze_frame, list) or not isinstance(location, list):
        return out

    ball = np.array(location[:2])
    cone = 0
    near_dists = []
    n3 = 0
    for e in freeze_frame:
        if e.get("teammate", True):  # sadece rakipler
            continue
        pl = np.array(e["location"][:2])
        if e.get("position", {}).get("name") == "Goalkeeper":
            out["gk_dist_goal"] = float(np.hypot(*(pl - GOAL)))
            out["gk_dist_ball"] = float(np.hypot(*(pl - ball)))
            continue
        if _in_triangle(pl, ball, POST1, POST2):
            cone += 1
        d = float(np.hypot(*(pl - ball)))
        near_dists.append(d)
        if d <= 3.0:
            n3 += 1
    out["opp_in_cone"] = cone
    if ENABLE_WEAK:
        out["dist_near_opp"] = min(near_dists) if near_dists else 99.0
        out["n_opp_3m"] = n3
    return out


def build_season(competition_id: int, season_id: int, cache_dir: str = "data/freeze_cache") -> pd.DataFrame:
    """
    Bir lig-sezonun şutlarını freeze feature'larıyla çeker. Sezon başına cache'ler
    (kesilirse kaldığı yerden). statsbombpy gerektirir (açık veri, login gerekmez).
    """
    from statsbombpy import sb

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"c{competition_id}_s{season_id}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    rows = []
    for mid in matches["match_id"]:
        ev = sb.events(mid)
        shots = ev[ev["type"] == "Shot"]
        for _, r in shots.iterrows():
            loc = r.get("location")
            if not isinstance(loc, list):
                continue
            x, y = loc[0], loc[1]
            rec = dict(
                competition_id=competition_id,
                season_id=season_id,
                match_id=mid,
                minute=int(r.get("minute", 0)),
                player=r.get("player"),
                x=x, y=y,
                distance=float(np.hypot(120 - x, 40 - y)),
                angle=subtended_angle(x, y),
                under_pressure=bool(r.get("under_pressure", False)) is True,
                shot_first_time=bool(r.get("shot_first_time", False)) is True,
                shot_one_on_one=bool(r.get("shot_one_on_one", False)) is True,
                body_part=r.get("shot_body_part"),
                technique=r.get("shot_technique"),
                shot_type=r.get("shot_type"),
                play_pattern=r.get("play_pattern"),
                xg_statsbomb=r.get("shot_statsbomb_xg"),
                is_goal=int(r.get("shot_outcome") == "Goal"),
            )
            rec.update(freeze_features(loc, r.get("shot_freeze_frame")))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(cache)
    return df


def build_many(comp_season_pairs, cache_dir: str = "data/freeze_cache") -> pd.DataFrame:
    """[(comp_id, season_id), ...] listesinden hepsini çekip birleştirir."""
    parts = []
    for cid, sid in comp_season_pairs:
        d = build_season(cid, sid, cache_dir)
        parts.append(d)
        print(f"comp {cid} season {sid}: {len(d)} şut")
    out = pd.concat(parts, ignore_index=True)
    print(f"TOPLAM: {len(out)} şut | freeze eksik oranı {out['opp_in_cone'].isna().mean():.3f}")
    return out


if __name__ == "__main__":
    # Örnek: prototipte kullanılan 5 La Liga sezonu. Tüm kapsam için config/leagues.py'den besle.
    LA_LIGA = [(11, 4), (11, 42), (11, 90), (11, 1), (11, 2)]
    df = build_many(LA_LIGA)
    df.to_parquet("data/shots_freeze.parquet")
    print("Kaydedildi: data/shots_freeze.parquet")
