import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xg_model_player_xgboost.joblib"
PEN_SPOT = (108.0, 40.0)

st.set_page_config(page_title="Player Comparison xG Studio", layout="wide")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def subtended_angle(x, y):
    a = np.hypot(120 - x, 36 - y)
    b = np.hypot(120 - x, 44 - y)
    return float(np.degrees(np.arccos(np.clip((a*a + b*b - 64) / (2*a*b + 1e-9), -1, 1))))


def draw_pitch(shot_x, shot_y, defenders=0, gk_off=False, penalty=False):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.set_facecolor("#0f4433")
    fig.patch.set_facecolor("#0d3d2e")

    ax.plot([0, 80], [120, 120], color="#3f7d66")
    ax.plot([18, 18, 62, 62], [120, 102, 102, 120], color="#3f7d66")
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], color="#3f7d66")
    ax.plot([36, 44], [120, 120], color="white", lw=4)

    ax.plot([shot_y, 40], [shot_x, 120], "--", color="#f2a623", lw=1.5)

    if not penalty:
        for i in range(defenders):
            t = 0.5 if defenders == 1 else 0.42 + i * 0.20
            off = 0 if defenders == 1 else (-6 if i == 0 else 6)
            ax.plot(
                shot_y + off + (40 - shot_y) * t,
                shot_x + (120 - shot_x) * t,
                "o",
                color="#d8584a",
                ms=9,
                mec="white",
            )

        gt = 0.28 if gk_off else 0.07
        ax.plot(
            40 + (shot_y - 40) * gt,
            120 + (shot_x - 120) * gt,
            "o",
            color="#3b82d6",
            ms=10,
            mec="white",
        )

    ax.plot(shot_y, shot_x, "o", color="#f2a623", ms=13, mec="white")

    if penalty:
        ax.text(shot_y + 2, shot_x, "Penalty", color="white", fontsize=10, va="center")

    ax.set_xlim(0, 80)
    ax.set_ylim(84, 122)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def build_row(player, position, dist, ang, head, pres, first_time, one_on_one, cone, gk_off, penalty=False):
    if penalty:
        return pd.DataFrame([{
            "distance": 12.0,
            "angle": 36.9,
            "under_pressure": 0,
            "shot_first_time": 0,
            "shot_one_on_one": 0,
            "opp_in_cone": 0,
            "gk_dist_goal": np.nan,
            "gk_dist_ball": np.nan,
            "body_part": "Right Foot",
            "technique": "Normal",
            "shot_type": "Penalty",
            "play_pattern": "Other",
            "player": player,
            "position": position,
        }])

    return pd.DataFrame([{
        "distance": dist,
        "angle": ang,
        "under_pressure": int(pres),
        "shot_first_time": int(first_time),
        "shot_one_on_one": int(one_on_one),
        "opp_in_cone": int(cone),
        "gk_dist_goal": 7.5 if gk_off else 2.9,
        "gk_dist_ball": max(dist - 7, 1) if gk_off else max(dist, 1),
        "body_part": "Head" if head else "Right Foot",
        "technique": "Normal",
        "shot_type": "Open Play",
        "play_pattern": "Regular Play",
        "player": player,
        "position": position,
    }])


artifact = load_artifact()
model = artifact["model"]
players = artifact["showcase_players"]
metrics = artifact["metrics"]

POSITION_MAP = {
    "Lionel Andrés Messi Cuccittini": "Right Wing",
    "Luis Alberto Suárez Díaz": "Center Forward",
    "Neymar da Silva Santos Junior": "Left Wing",
    "Cristiano Ronaldo dos Santos Aveiro": "Center Forward",
    "Kylian Mbappé Lottin": "Left Wing",
    "Antoine Griezmann": "Center Forward",
    "Zlatan Ibrahimović": "Center Forward",
    "Harry Kane": "Center Forward",
    "Karim Benzema": "Center Forward",
    "Gareth Frank Bale": "Right Wing",
}


st.title("Player Comparison xG Studio")
st.caption("Aynı şut senaryosunda elit oyuncuların xGBoost tahminlerini karşılaştırır.")

if "shot" not in st.session_state:
    st.session_state.shot = [104.0, 40.0]

if "penalty" not in st.session_state:
    st.session_state.penalty = False

with st.sidebar:
    st.subheader("Şut Senaryosu")

    selected_players = st.multiselect(
        "Karşılaştırılacak oyuncular",
        players,
        default=players[:6],
    )

    head = st.toggle("Kafa şutu", value=False, disabled=st.session_state.penalty)
    pres = st.toggle("Baskı altında", value=False, disabled=st.session_state.penalty)
    first_time = st.toggle("İlk dokunuşta şut", value=False, disabled=st.session_state.penalty)
    one_on_one = st.toggle("Kaleciyle karşı karşıya", value=False, disabled=st.session_state.penalty)

    st.markdown("### Freeze özellikleri")
    cone = st.select_slider(
        "Önündeki savunmacı",
        options=[0, 1, 2],
        value=0,
        disabled=st.session_state.penalty,
    )
    gk_off = st.toggle("Kaleci çizgiden çıkmış", value=False, disabled=st.session_state.penalty)

    if st.button("⚽ Penaltı noktası", use_container_width=True):
        st.session_state.shot = [PEN_SPOT[0], PEN_SPOT[1]]
        st.session_state.penalty = True
        st.rerun()

    if st.button("Açık oyun moduna dön", use_container_width=True):
        st.session_state.penalty = False
        st.rerun()


old_shot = st.session_state.shot

c1, c2 = st.columns(2)
with c1:
    shot_x = st.slider("Derinlik / Kaleye yakınlık", 85.0, 119.0, float(old_shot[0]))
with c2:
    shot_y = st.slider("Yanal konum", 2.0, 78.0, float(old_shot[1]))

if [shot_x, shot_y] != old_shot:
    st.session_state.penalty = False

st.session_state.shot = [shot_x, shot_y]
penalty = st.session_state.penalty

dist = float(np.hypot(120 - shot_x, 40 - shot_y))
ang = subtended_angle(shot_x, shot_y)

if not selected_players:
    st.warning("En az bir oyuncu seç.")
    st.stop()

rows = []
for p in selected_players:
    position = POSITION_MAP.get(p, "Center Forward")
    row = build_row(
        player=p,
        position=position,
        dist=dist,
        ang=ang,
        head=head,
        pres=pres,
        first_time=first_time,
        one_on_one=one_on_one,
        cone=cone,
        gk_off=gk_off,
        penalty=penalty,
    )
    xg = float(model.predict_proba(row)[:, 1][0])
    rows.append({"Oyuncu": p, "Pozisyon": position, "xG": xg})

results = pd.DataFrame(rows).sort_values("xG", ascending=False)

left, right = st.columns([1.15, 1])

with left:
    st.pyplot(draw_pitch(shot_x, shot_y, defenders=cone, gk_off=gk_off, penalty=penalty))

with right:
    st.subheader("Karşılaştırma Sonucu")

    best = results.iloc[0]
    st.markdown(
        f"""
        <div style="font-size:44px;font-weight:700;color:#6ee7a8;line-height:1">
            {best["xG"]:.2f}
        </div>
        <div style="color:#9cc4b3;font-size:15px">
            En yüksek tahmin: {best["Oyuncu"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"Mesafe: **{12.0 if penalty else dist:.1f} m**")
    st.write(f"Açı: **{36.9 if penalty else ang:.1f}°**")

    if penalty:
        st.info("Penaltı modu aktif: shot_type='Penalty', play_pattern='Other' olarak modele gönderiliyor.")

    st.dataframe(
        results.style.format({"xG": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.caption(
    f"Player XGBoost v2 ROC-AUC: {metrics['roc_auc']:.4f} | "
    f"Log Loss: {metrics['log_loss']:.4f} | "
    f"Brier Score: {metrics['brier_score']:.4f}. "
    "Bu bonus model oyuncu kimliği + freeze özellikleriyle eğitilmiştir."
)
