"""
Oyuncu Karşılaştırma — BONUS (3.) model
=======================================
Aynı şut senaryosunda elit golcülerin xG tahminlerini karşılaştırır.

DİKKAT — bu model EĞLENCE amaçlı bir bonus. Ana modelimizin (baseline/freeze)
aksine `player` (oyuncu kimliği) ve `position` feature'larını KULLANIR.
Test AUC'si ana freeze modeliyle aynı (~0.819) ama oyuncu kimliği eklenince
küçük örneklemli oyunculara OVERFIT eder: az şutu olan bir oyuncu (örn. Bale,
124 şut) şans eseri yüksek dönüşüm oranı yüzünden listenin başına çıkabilir.
Bu, ana modelimizde `player`'ı neden KASTEN attığımızın canlı kanıtıdır.

Bu sayfa ana demoyu (streamlit_app.py) hiç etkilemez — bağımsız çalışır.
Model: models/xg_model_player_xgboost.joblib
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xg_model_player_xgboost.joblib"
PEN_SPOT = (108.0, 40.0)

# Deck paleti (ana demo ile aynı)
GOLD = "#F2A91E"; BG = "#0C3326"; PANEL = "#11402F"
MINT = "#9CC4B3"; CREAM = "#F3F7F4"; LINE = "#1d5740"; RED = "#d8584a"

st.set_page_config(page_title="Oyuncu Karşılaştırma · xG Stüdyo",
                   layout="wide", page_icon="⚽", initial_sidebar_state="expanded")

# Görünen kısa ad + şut sayısı (shots_freeze.parquet'ten; bağlam/güvenilirlik için)
PLAYER_META = {
    "Lionel Andrés Messi Cuccittini":      ("Messi",     2632, "Right Wing"),
    "Luis Alberto Suárez Díaz":            ("Suárez",     632, "Center Forward"),
    "Neymar da Silva Santos Junior":       ("Neymar",     466, "Left Wing"),
    "Cristiano Ronaldo dos Santos Aveiro": ("Ronaldo",    413, "Center Forward"),
    "Kylian Mbappé Lottin":                ("Mbappé",     306, "Left Wing"),
    "Antoine Griezmann":                   ("Griezmann",  284, "Center Forward"),
    "Zlatan Ibrahimović":                  ("Zlatan",     231, "Center Forward"),
    "Harry Kane":                          ("Kane",       224, "Center Forward"),
    "Karim Benzema":                       ("Benzema",    168, "Center Forward"),
    "Gareth Frank Bale":                   ("Bale",       124, "Right Wing"),
}
LOW_N = 200  # bu sayının altı = "az veri" uyarısı


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
    .stApp {{ background:{BG}; }}
    #MainMenu, [data-testid="stToolbar"], footer {{ visibility:hidden; height:0; }}
    header[data-testid="stHeader"] {{ background:transparent; }}
    section[data-testid="stSidebar"] {{ transform:none !important; visibility:visible !important;
        min-width:300px !important; margin-left:0 !important; background:{PANEL}; border-right:1px solid {LINE}; }}
    [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{ display:none !important; }}
    .block-container {{ max-width:1040px; margin:0 auto; padding-top:1.1rem; }}
    html, body, [class*="css"] {{ font-family:'Inter',-apple-system,sans-serif; }}
    h1,h2,h3 {{ font-family:'Poppins',sans-serif; }}

    .xg-h1 {{ font-family:'Poppins'; font-size:34px; font-weight:700; color:{CREAM};
              text-align:center; letter-spacing:-.5px; margin:.2rem 0 .1rem; }}
    .xg-h1 .accent {{ color:{GOLD}; }}
    .xg-sub {{ text-align:center; color:{MINT}; font-size:14px; max-width:720px;
               margin:0 auto 1rem; line-height:1.5; }}
    .pitch-hint {{ text-align:center; color:{MINT}; font-size:13px; margin:.3rem 0 .5rem; }}

    /* bar listesi */
    .bar-row {{ display:flex; align-items:center; gap:10px; margin:7px 0; }}
    .bar-name {{ width:140px; color:{CREAM}; font-size:14px; font-weight:500; text-align:right;
                 white-space:nowrap; }}
    .bar-name .n {{ color:{MINT}; font-size:11px; font-weight:400; }}
    .bar-name .warn {{ color:{RED}; font-size:11px; }}
    .bar-track {{ flex:1; background:{BG}; border:1px solid {LINE}; border-radius:999px; height:24px;
                  overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px;
                 background:linear-gradient(90deg,#a06b12,{GOLD}); }}
    .bar-fill.lown {{ background:linear-gradient(90deg,#7a3a33,{RED}); opacity:.85; }}
    .bar-val {{ width:54px; color:{CREAM}; font-family:'Poppins'; font-weight:600; font-size:15px; }}

    .winner {{ font-family:'Poppins'; font-weight:700; font-size:40px; color:{GOLD}; line-height:1; }}
    .winner-sub {{ color:{MINT}; font-size:13px; margin-bottom:.6rem; }}

    .warnbox {{ background:{BG}; border-left:3px solid {RED}; border-radius:8px; padding:12px 14px;
                color:#f0d9d4; font-size:13px; line-height:1.5; margin-top:1rem; }}
    .warnbox b {{ color:{CREAM}; }}
    .xg-foot {{ color:{MINT}; font-size:12px; line-height:1.55; border-top:1px solid {LINE};
                padding-top:.9rem; margin-top:1rem; }}
    div[data-testid="column"] iframe {{ display:block; margin:0 auto; }}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def subtended_angle(x, y):
    a = np.hypot(120 - x, 36 - y); b = np.hypot(120 - x, 44 - y)
    return float(np.degrees(np.arccos(np.clip((a * a + b * b - 64) / (2 * a * b + 1e-9), -1, 1))))


def build_row(player, position, dist, ang, head, pres, first_time, one_on_one, cone, gk_off, penalty=False):
    if penalty:
        return pd.DataFrame([dict(distance=12.0, angle=36.9, under_pressure=0, shot_first_time=0,
            shot_one_on_one=0, opp_in_cone=0, gk_dist_goal=np.nan, gk_dist_ball=np.nan,
            body_part="Right Foot", technique="Normal", shot_type="Penalty", play_pattern="Other",
            player=player, position=position)])
    return pd.DataFrame([dict(distance=dist, angle=ang, under_pressure=int(pres),
        shot_first_time=int(first_time), shot_one_on_one=int(one_on_one), opp_in_cone=int(cone),
        gk_dist_goal=7.5 if gk_off else 2.9, gk_dist_ball=max(dist - 7, 1) if gk_off else max(dist, 1),
        body_part="Head" if head else "Right Foot", technique="Normal", shot_type="Open Play",
        play_pattern="Regular Play", player=player, position=position)])


def draw_pitch(shot_x, shot_y, defenders=0, gk_off=False, penalty=False):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.set_facecolor("#0f4433"); fig.patch.set_facecolor(BG)
    ax.plot([0, 80], [120, 120], color="#3f7d66")
    ax.plot([0, 0], [85, 120], color="#3f7d66")
    ax.plot([80, 80], [85, 120], color="#3f7d66")
    ax.plot([0, 80], [85, 85], color="#3f7d66")
    ax.plot([18, 18, 62, 62], [120, 102, 102, 120], color="#3f7d66")
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], color="#3f7d66")
    ax.plot([36, 44], [120, 120], color="#eaf3ee", lw=4)
    ax.plot([shot_y, 40], [shot_x, 120], "--", color=GOLD, lw=1.5, alpha=0.7)
    if not penalty:
        for i in range(defenders):
            t = 0.5 if defenders == 1 else 0.42 + i * 0.20
            off = 0 if defenders == 1 else (-6 if i == 0 else 6)
            ax.plot(shot_y + off + (40 - shot_y) * t, shot_x + (120 - shot_x) * t,
                    "o", color=RED, ms=9, mec="white")
        gt = 0.28 if gk_off else 0.07
        ax.plot(40 + (shot_y - 40) * gt, 120 + (shot_x - 120) * gt, "o", color="#3b82d6", ms=10, mec="white")
    ax.plot(shot_y, shot_x, "o", color=GOLD, ms=13, mec="white")
    ax.set_xlim(-2, 82); ax.set_ylim(83, 122); ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


# ---------------- UI ----------------
inject_css()
artifact = load_artifact()
model = artifact["model"]
players = artifact["showcase_players"]
metrics = artifact["metrics"]

st.markdown("<div class='xg-h1'>Oyuncu <span class='accent'>Karşılaştırma</span> "
            "<span style='font-size:15px;color:#9cc4b3'>· bonus model</span></div>",
            unsafe_allow_html=True)
st.markdown("<div class='xg-sub'>Aynı şutu farklı golcüler nasıl değerlendirir? Bu bonus model "
            "oyuncu kimliğini de görür. Eğlenceli — ama aşağıdaki uyarıyı oku.</div>",
            unsafe_allow_html=True)

if "shot" not in st.session_state:
    st.session_state.shot = [104.0, 40.0]
if "penalty" not in st.session_state:
    st.session_state.penalty = False

with st.sidebar:
    st.subheader("Şut senaryosu")
    selected = st.multiselect("Karşılaştırılacak oyuncular", players, default=players,
                              format_func=lambda p: PLAYER_META[p][0])
    pen = st.session_state.penalty
    head = st.toggle("Kafa şutu", value=False, disabled=pen)
    pres = st.toggle("Baskı altında", value=False, disabled=pen)
    first_time = st.toggle("İlk dokunuşta şut", value=False, disabled=pen)
    one_on_one = st.toggle("Kaleciyle karşı karşıya", value=False, disabled=pen)
    st.markdown("**Freeze özellikleri**")
    cone = st.segmented_control("Önündeki savunmacı", options=[0, 1, 2],
                                default=0, selection_mode="single", disabled=pen)
    if cone is None:
        cone = 0
    gk_off = st.toggle("Kaleci çizgiden çıkmış", value=False, disabled=pen)
    st.divider()
    if st.button("⚽ Penaltı noktası", use_container_width=True):
        st.session_state.shot = [PEN_SPOT[0], PEN_SPOT[1]]
        st.session_state.penalty = True
        st.rerun()
    if st.session_state.penalty and st.button("↩ Açık oyuna dön", use_container_width=True):
        st.session_state.penalty = False
        st.rerun()

# saha tıklaması
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK = True
except Exception:
    HAS_CLICK = False

shot_x, shot_y = st.session_state.shot
penalty = st.session_state.penalty

if HAS_CLICK:
    st.markdown("<div class='pitch-hint'>Sahaya tıklayarak şutu yerleştir "
                "(tıklayınca penaltı modu kapanır)</div>", unsafe_allow_html=True)
    fig = draw_pitch(shot_x, shot_y, defenders=cone, gk_off=gk_off, penalty=penalty)
    (ROOT / "app").mkdir(exist_ok=True)
    fig.savefig(ROOT / "app" / "_pitch_player.png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    _l, _m, _r = st.columns([1, 3, 1])
    with _m:
        coords = streamlit_image_coordinates(str(ROOT / "app" / "_pitch_player.png"), width=500)
    if coords:
        py = coords["x"] / coords["width"] * 80
        px = 120 - coords["y"] / coords["height"] * 36
        st.session_state.shot = [float(np.clip(px, 85, 119)), float(np.clip(py, 2, 78))]
        st.session_state.penalty = False
        st.rerun()
else:
    c1, c2 = st.columns(2)
    px = c1.slider("Derinlik (kaleye yakınlık)", 85.0, 119.0, float(shot_x))
    py = c2.slider("Yanal konum", 2.0, 78.0, float(shot_y))
    if (px, py) != (shot_x, shot_y):
        st.session_state.penalty = False
    st.session_state.shot = [px, py]; shot_x, shot_y = px, py
    _l, _m, _r = st.columns([1, 3, 1])
    with _m:
        st.pyplot(draw_pitch(shot_x, shot_y, defenders=cone, gk_off=gk_off, penalty=st.session_state.penalty))

penalty = st.session_state.penalty
dist = float(np.hypot(120 - shot_x, 40 - shot_y))
ang = subtended_angle(shot_x, shot_y)

if not selected:
    st.warning("En az bir oyuncu seç.")
    st.stop()

# tahminler
rows = []
for p in selected:
    short, n, pos = PLAYER_META[p]
    xg = float(model.predict_proba(build_row(p, pos, dist, ang, head, pres,
                                             first_time, one_on_one, cone, gk_off, penalty))[:, 1][0])
    rows.append((xg, short, n))
rows.sort(reverse=True)
maxxg = max(r[0] for r in rows) or 1e-9

# kazanan + bağlam
best_xg, best_name, best_n = rows[0]
hdr = f"{dist:.0f} m · {ang:.0f}° açı" + (" · penaltı" if penalty else "")
st.markdown(f"<div class='winner'>{best_name} · {best_xg:.2f}</div>"
            f"<div class='winner-sub'>en yüksek tahmin · {hdr}</div>", unsafe_allow_html=True)

# sıralı yatay bar
bars = ""
for xg, name, n in rows:
    pct = max(xg / maxxg * 100, 4)
    low = n < LOW_N
    ntag = (f"<span class='warn'>n={n} · az veri</span>" if low else f"<span class='n'>n={n}</span>")
    bars += (f"<div class='bar-row'>"
             f"<div class='bar-name'>{name} {ntag}</div>"
             f"<div class='bar-track'><div class='bar-fill{' lown' if low else ''}' "
             f"style='width:{pct:.0f}%'></div></div>"
             f"<div class='bar-val'>{xg:.2f}</div></div>")
st.markdown(bars, unsafe_allow_html=True)

# dürüst uyarı kutusu
st.markdown(
    "<div class='warnbox'>⚠️ <b>Bu bir bonus / eğlence modeli.</b> Ana modelimiz oyuncu kimliğini "
    "<b>kasten dışlar</b> — çünkü xG pozisyonun kalitesini ölçmeli, kimin vurduğunu değil. Bu model "
    "ise <b>player</b> feature'ını ekliyor ve az şutu olan oyunculara (kırmızı barlar, ör. Bale 124 şut) "
    "<b>overfit</b> ediyor: küçük örneklemde şans eseri yüksek dönüşüm, gerçek bitiricilik gibi görünüyor. "
    "Test AUC'si ana modelle aynı (~0.819) ama oyuncu kimliği ayrıma neredeyse hiç katkı vermiyor — "
    "yani sıralamalar <b>finishing yeteneği değil, örneklem gürültüsü</b>. İşte ana modelde "
    "<b>player'ı neden attığımızın canlı kanıtı.</b></div>",
    unsafe_allow_html=True)

st.markdown(
    f"<div class='xg-foot'>Player XGBoost (bonus) · ROC-AUC {metrics['roc_auc']:.4f} · "
    f"LogLoss {metrics['log_loss']:.4f} · Brier {metrics['brier_score']:.4f}. "
    f"Oyuncu kimliği + freeze özellikleriyle eğitildi (kalibrasyonsuz). Kırmızı bar = az veri "
    f"(n&lt;{LOW_N}), tahmini güvenilmez.</div>",
    unsafe_allow_html=True)
