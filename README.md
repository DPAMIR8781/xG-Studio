# xG Stüdyo — Canlı Şut Kalitesi (xG) Tahmin Motoru

Workintech Data Scientist & AI kapanış projesi. StatsBomb açık verisiyle eğitilmiş bir
xG (Expected Goals) modeli + canlı Streamlit arayüzü + AI yorum katmanı.

## Ne yapıyor
Kullanıcı saha üzerinde bir şut konumlandırır; model gol olma olasılığını (xG) anında
hesaplar. **İki model yan yana:**
- **baseline** — mesafe, açı, vücut, baskı, oyun durumu (75k şut)
- **freeze** — üstüne şut anı oyuncu konumları (savunmacılar, kaleci): opp_in_cone,
  gk_dist_ball, gk_dist_goal

### Sonuçlar (aynı test seti)
| Model | ROC-AUC |
|---|---|
| baseline | 0.8017 |
| **freeze** | **0.8183** (+0.0166) |
| StatsBomb referans | 0.8191 |

Freeze, StatsBomb'un kendi xG modeline 0.0008 AUC kadar yaklaşıyor.

## Klasör yapısı
```
app/      streamlit_app.py (canlı arayüz) + requirements.txt
src/      build_freeze_features.py, train_model.py, train_freeze_merged.py
models/   xg_model.joblib (baseline) + xg_model_freeze.joblib (freeze)
data/     shots_features.parquet (baseline) + shots_freeze.parquet (freeze)
docs/     index.html — standalone interaktif demo (GitHub Pages)
```

## Çalıştırma
```bash
pip install -r app/requirements.txt
# (opsiyonel) AI yorum katmanı için:
export GOOGLE_API_KEY=...        # yoksa yerel yorum havuzu kullanılır
streamlit run app/streamlit_app.py
```
Model dosyaları models/ altında hazır geldiği için ekstra eğitim/çekim gerekmez.

### Demo (kurulumsuz)
docs/index.html dosyasını tarayıcıda çift tıkla aç — sunucu/kurulum gerektirmez.
GitHub Pages açıksa: https://ubtuna.github.io/xG-Studio/

## Yeniden üretmek (opsiyonel)
```bash
python src/train_freeze_merged.py   # freeze modeli tam veride eğitir
```

## Veri
StatsBomb Open Data (CC BY-SA 4.0, atıf zorunlu). statsbombpy ile çekilir, login gerekmez.
