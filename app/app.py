"""
xG Stüdyo — giriş / navigasyon
==============================
İki sayfayı açık adlarla birleştirir (st.navigation). Streamlit'in dosya-adından
türettiği çirkin menü etiketleri ("streamlit app") yerine düzgün adlar gösterir.

Çalıştırma:
    streamlit run app/app.py
"""
import streamlit as st

pages = [
    st.Page("streamlit_app.py", title="Canlı Tahmin", icon="⚽", default=True),
    st.Page("player_comparison.py", title="Oyuncu Karşılaştırma", icon="👤"),
]

st.navigation(pages).run()
