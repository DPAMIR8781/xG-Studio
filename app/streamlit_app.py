"""
xG Stüdyo — Streamlit canlı şut tahmincisi (baseline + freeze, tek arayüz)
==========================================================================
Tek sahaya tıkla → baseline ve freeze modelin xG'sini yan yana gör.
Baseline savunmayı görmez; freeze'in gelişmiş kontrolleri (savunmacı, kaleci)
xG'yi anında değiştirir. Varsayılan = açık şut (Seçenek-2 mimarisi).

AI yorum katmanı: senaryo başına KÜÇÜK VARYANT HAVUZU cache'lenir (slayt 8).
- Aynı senaryo tekrar tıklanınca havuzdaki 3 varyant arasında döner (çeşitlilik).
- Havuz dolunca yeni LLM çağrısı yok → kota minimal.
- GOOGLE_API_KEY tanımlıysa Gemini Flash'tan varyant üretir; yoksa yerel
  varyant havuzuna düşer (demo internetsiz de çeşitli görünür).

Çalıştırma:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...   # opsiyonel; yoksa yerel havuz kullanılır
    streamlit run app/streamlit_app.py

Modeller: models/xg_model.joblib (baseline) + models/xg_model_freeze.joblib (freeze)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PEN_SPOT = (108.0, 40.0)          # StatsBomb penaltı noktası
MAX_VARIANTS = 3                  # senaryo başına AI yorum havuzu boyutu
# Sabit (UI'da olmayan) feature varsayılanları
FIXED = dict(technique="Normal", play_pattern="Regular Play",
             shot_first_time=0, shot_one_on_one=0)

st.set_page_config(page_title="xG Stüdyo", layout="wide")


@st.cache_resource
def load_models():
    return (joblib.load(ROOT / "models" / "xg_model.joblib"),
            joblib.load(ROOT / "models" / "xg_model_freeze.joblib"))


def subtended_angle(x, y):
    a = np.hypot(120 - x, 36 - y); b = np.hypot(120 - x, 44 - y)
    return float(np.degrees(np.arccos(np.clip((a * a + b * b - 64) / (2 * a * b + 1e-9), -1, 1))))


def baseline_row(dist, ang, head, pres, shot_type):
    return pd.DataFrame([dict(distance=dist, angle=ang, under_pressure=int(pres),
                              body_part="Head" if head else "Right Foot",
                              shot_type=shot_type, **FIXED)])


def freeze_row(dist, ang, head, pres, cone, gk_off, shot_type):
    return pd.DataFrame([dict(
        distance=dist, angle=ang, opp_in_cone=int(cone),
        gk_dist_goal=7.5 if gk_off else 2.9,
        gk_dist_ball=max(dist - 7, 1) if gk_off else max(dist, 1),
        under_pressure=int(pres), body_part="Head" if head else "Right Foot",
        shot_type=shot_type, **FIXED)])


# ---------------- AI yorum katmanı (havuzlu cache) ----------------
_POOLS = {
    "vlow": ["Neredeyse imkânsız — buradan goller çok seyrek.",
             "Çok düşük ihtimal; bu mesafe ve açıdan gol istisna.",
             "Zorlu pozisyon, model bu şuta pek şans tanımıyor."],
    "low":  ["Düşük olasılık; çoğu zaman kaleci bunu toplar.",
             "Zor şut — ara sıra girer ama beklenti düşük.",
             "Model temkinli: bu pozisyondan goller az."],
    "mid":  ["Orta seviye şans; iyi vuruşla girebilir.",
             "Fena değil — kaleci ve açı belirleyici olacak.",
             "Ortalama bir şut; isabet kadar şans da gerek."],
    "good": ["İyi fırsat; bu pozisyon golcüyü sevindirir.",
             "Tehlikeli şut — beklenti belirgin biçimde yüksek.",
             "Model bunu beğeniyor: gol ihtimali ciddi."],
    "great":["Büyük fırsat — buradan gol beklenir.",
             "Neredeyse net; golcü bunu çoğu zaman atar.",
             "Model net konuşuyor: yüksek gol olasılığı."],
}
_CONE = ["Önündeki {n} savunmacı şansı kırıyor.", "{n} savunmacı çizgiyi kapatmış.",
         "{n} savunmacı bloke tehdidinde."]
_GKOFF = ["Kaleci çizgiden çıkmış, açı değişiyor.", "Kaleci ileride — boşluk farklı."]


def _tier(xg):
    return "vlow" if xg < 0.05 else "low" if xg < 0.12 else "mid" if xg < 0.25 else "good" if xg < 0.45 else "great"


def _local_variant(xg, ctx, i):
    """Çevrimdışı yedek: i. varyantı deterministik üretir."""
    pool = _POOLS[_tier(xg)]
    s = pool[i % len(pool)]
    if ctx.get("cone", 0) > 0:
        s += " " + _CONE[i % len(_CONE)].format(n="2+" if ctx["cone"] >= 2 else ctx["cone"])
    if ctx.get("gk_off"):
        s += " " + _GKOFF[i % len(_GKOFF)]
    if ctx.get("penalty"):
        s = "Penaltı — istatistiksel olarak en net fırsatlardan biri."
    return s


def _gemini_variant(xg, ctx):
    """Gemini Flash'tan tek cümlelik yorum. API yoksa/başarısızsa None döner."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")   # kotaya göre değiştir
        prompt = (
            "Sen bir futbol analiz asistanısın. Aşağıdaki şut için TEK kısa Türkçe cümle yaz "
            "(en fazla 18 kelime), spiker/analist tonunda, doğal ve özgün. Sayıyı tekrar etme.\n"
            f"- Model: {ctx['model']}\n- xG: {xg:.2f}\n- Mesafe: {ctx['dist']:.0f} m, açı: {ctx['ang']:.0f} derece\n"
            f"- Vücut: {'kafa' if ctx['head'] else 'ayak'}, baskı: {'var' if ctx['pres'] else 'yok'}\n"
            f"- Önündeki savunmacı: {ctx.get('cone', 0)}, kaleci: {'çıkmış' if ctx.get('gk_off') else 'çizgide'}\n"
            f"- Penaltı: {'evet' if ctx.get('penalty') else 'hayır'}\n"
        )
        r = model.generate_content(prompt, generation_config={"temperature": 0.95, "max_output_tokens": 60})
        return r.text.strip().replace("\n", " ")
    except Exception:
        return None


def ai_comment(xg, ctx, side):
    """Havuzlu cache: senaryo anahtarı başına MAX_VARIANTS yorum tutar, aralarında döner."""
    cache = st.session_state.setdefault("ai_cache", {})
    ticks = st.session_state.setdefault("ai_ticks", {})
    key = (ctx["model"], round(ctx["dist"] / 3), round(ctx["ang"] / 10),
           ctx["head"], ctx["pres"], ctx.get("cone", 0), ctx.get("gk_off", False), ctx.get("penalty", False))
    pool = cache.setdefault(key, [])
    if len(pool) < MAX_VARIANTS:
        c = _gemini_variant(xg, ctx) or _local_variant(xg, ctx, len(pool))
        if c not in pool:
            pool.append(c)
    i = ticks.get(side, 0); ticks[side] = i + 1
    return pool[i % len(pool)]


def draw_pitch(shot_x, shot_y, defenders=0, gk_off=None, penalty=False):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.set_facecolor("#0f4433"); fig.patch.set_facecolor("#0d3d2e")
    ax.plot([0, 80], [120, 120], color="#3f7d66")
    ax.plot([18, 18, 62, 62], [120, 102, 102, 120], color="#3f7d66")
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], color="#3f7d66")
    ax.plot([36, 44], [120, 120], color="#eaf3ee", lw=4)
    ax.plot(40, 108, "o", color="#3f7d66", ms=3)
    ax.plot([shot_y, 40], [shot_x, 120], "--", color="#f2a623", lw=1, alpha=0.7)
    if not penalty:
        for i in range(defenders):
            t = 0.5 if defenders == 1 else (0.4 + i * 0.24)
            off = 0 if defenders == 1 else (-7 if i == 0 else 7)
            ax.plot(shot_y + off + (40 - shot_y) * t, shot_x + (120 - shot_x) * t,
                    "o", color="#d8584a", ms=9, mec="white")
        if gk_off is not None:
            gt = 0.28 if gk_off else 0.07
            ax.plot(40 + (shot_y - 40) * gt, 120 + (shot_x - 120) * gt,
                    "o", color="#3b82d6", ms=10, mec="white")
    ax.plot(shot_y, shot_x, "o", color="#f2a623", ms=12, mec="white")
    ax.set_xlim(0, 80); ax.set_ylim(84, 122); ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def xg_color(v):
    return "#6ee7a8" if v >= 0.4 else "#f2d24a" if v >= 0.2 else "#f2a623" if v >= 0.1 else "#e88"


def big_xg(v, sub):
    st.markdown(f"<div style='font-size:46px;font-weight:600;color:{xg_color(v)};line-height:1'>{v:.2f}</div>"
                f"<div style='color:#9cc4b3;font-size:13px'>{sub}</div>", unsafe_allow_html=True)


# ---------------- UI ----------------
st.title("xG Stüdyo — canlı şut tahmincisi")
st.caption("Sahaya tıkla, şutu kur. Baseline model savunmayı görmez; freeze modeli savunmacı ve "
           "kaleci konumuna tepki verir. Aynı şut iki modele de gider.")

base_model, frz_model = load_models()
if "shot" not in st.session_state:
    st.session_state.shot = list(PEN_SPOT[::-1])  # (x,y) -> şu an penaltı yayı civarı degil; asagida ayarlanir
    st.session_state.shot = [104.0, 40.0]
if "penalty" not in st.session_state:
    st.session_state.penalty = False

with st.sidebar:
    st.subheader("Şut kurulumu")
    head = st.toggle("Kafa şutu", value=False)
    pres = st.toggle("Baskı altında", value=False)
    st.markdown("**Freeze — gelişmiş kontroller**")
    cone = st.select_slider("Önündeki savunmacı", options=[0, 1, 2], value=0)
    gk_off = st.toggle("Kaleci çizgiden çıkmış", value=False)
    st.caption("Varsayılan (0 savunmacı + kaleci çizgide) = açık şut → tek-tık deneyimi.")
    st.divider()
    if st.button("⚽ Penaltı noktası", use_container_width=True):
        st.session_state.shot = [PEN_SPOT[0], PEN_SPOT[1]]
        st.session_state.penalty = True
        st.rerun()

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK = True
except Exception:
    HAS_CLICK = False

shot_x, shot_y = st.session_state.shot
penalty = st.session_state.penalty

if HAS_CLICK:
    st.write("Sahaya tıklayarak şutu yerleştir (tıklayınca penaltı modu kapanır):")
    fig = draw_pitch(shot_x, shot_y, defenders=cone, gk_off=gk_off, penalty=penalty)
    (ROOT / "app").mkdir(exist_ok=True)
    fig.savefig(ROOT / "app" / "_pitch.png", dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    coords = streamlit_image_coordinates(str(ROOT / "app" / "_pitch.png"), width=360)
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
    st.pyplot(draw_pitch(shot_x, shot_y, defenders=cone, gk_off=gk_off, penalty=st.session_state.penalty))

penalty = st.session_state.penalty
dist = float(np.hypot(120 - shot_x, 40 - shot_y))
ang = subtended_angle(shot_x, shot_y)
sub = f"{dist:.0f} m · {ang:.0f}° açı" + (" · penaltı" if penalty else "")

if penalty:
    # Penaltı: kaleci çizgide ortada (freeze frame'de GK çoğu kez yok -> NaN),
    # savunmacı yok, play_pattern='Other' (StatsBomb penaltı deseni). shot_type sinyali baskın.
    brow = pd.DataFrame([dict(distance=12.0, angle=36.9, under_pressure=0, shot_first_time=0,
                              shot_one_on_one=0, body_part="Right Foot", technique="Normal",
                              shot_type="Penalty", play_pattern="Other")])
    frow = pd.DataFrame([dict(distance=12.0, angle=36.9, opp_in_cone=0, gk_dist_goal=np.nan,
                              gk_dist_ball=np.nan, under_pressure=0, shot_first_time=0,
                              shot_one_on_one=0, body_part="Right Foot", technique="Normal",
                              shot_type="Penalty", play_pattern="Other")])
    xb = float(base_model["model"].predict_proba(brow)[:, 1][0])
    xf = float(frz_model["model"].predict_proba(frow)[:, 1][0])
else:
    xb = float(base_model["model"].predict_proba(baseline_row(dist, ang, head, pres, "Open Play"))[:, 1][0])
    xf = float(frz_model["model"].predict_proba(freeze_row(dist, ang, head, pres, cone, gk_off, "Open Play"))[:, 1][0])

ctx_b = dict(model="baseline", dist=dist, ang=ang, head=head, pres=pres, penalty=penalty)
ctx_f = dict(model="freeze", dist=dist, ang=ang, head=head, pres=pres,
             cone=cone, gk_off=gk_off, penalty=penalty)

col_b, col_f = st.columns(2)
with col_b:
    st.subheader("Baseline model")
    st.caption("dokunulmamış ilk model · 75k şut")
    big_xg(xb, sub)
    st.info(ai_comment(xb, ctx_b, "b"))
    st.caption("Savunmacı kontrolünü değiştirsen de bu değer sabit — model savunmayı görmez.")
with col_f:
    st.subheader("Freeze model")
    st.caption("freeze-eğitimli · gelişmiş kontroller")
    big_xg(xf, sub + f" · baseline farkı {'+' if xf - xb >= 0 else ''}{xf - xb:.2f}")
    st.info(ai_comment(xf, ctx_f, "f"))

st.divider()
st.caption("Her iki model 75k tam veride eğitildi, kalibrasyonları hizalı (ort. tahmin ≈ gol oranı %11.2). "
           "Freeze, baseline'a göre +0.0166 AUC kazandırıyor (0.802 → 0.818) ve StatsBomb'un kendi modelini "
           "(0.819) neredeyse yakalıyor. Mesaj: baseline savunmaya kör, freeze tepki veriyor. · AI yorumları "
           "senaryo başına havuzlanır ve cache'lenir (kota minimal).")
