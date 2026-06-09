# xG Stüdyo ⚽

Canlı şut kalitesi (xG) tahmin motoru — Workintech Data Scientist & AI kapanış projesi.

StatsBomb açık verisiyle eğitilmiş bir Expected Goals (xG) modeli, canlı bir Streamlit
arayüzü ve Gemini Flash destekli AI yorum katmanı. Local'de eğitim, GCP'de serving.

## Ekip
- Uğur Batuhan Tuna (proje koordinasyonu / entegrasyon)
- Doruk Pamir
- Mert Kaan Topuz

## Ne yapıyor?
Kullanıcı saha üzerinde bir şut konumlandırır (yer, ayak/kafa, savunma baskısı), model
o şutun gol olma olasılığını (xG) anında hesaplar, AI katmanı sonucu tek cümleyle
insan diline çevirir.

## Stack
- **Veri:** StatsBomb Open Data (`statsbombpy`)
- **İşleme/SQL:** pandas + BigQuery (analiz/aggregation katmanı)
- **Model:** scikit-learn, XGBoost / LightGBM + olasılık kalibrasyonu
- **Arayüz:** Streamlit
- **AI katmanı:** Gemini Flash (cache'li)
- **Deploy:** GCP VM
- **Ortam:** Python 3.x, pyenv

## Klasör yapısı
```
data/        # ham + işlenmiş şut tabloları (büyük dosyalar git'e gitmez)
notebooks/   # EDA, feature engineering, model deneyleri
src/         # pipeline kodu (veri çekme, temizleme, model eğitimi)
models/      # joblib ile kaydedilmiş model
app/         # Streamlit uygulaması
config/      # seçilen lig/sezon listesi (competition_id / season_id)
```

## Kurulum
```bash
git clone https://github.com/<kullanici>/xg-studio.git
cd xg-studio
python -m venv venv && source venv/bin/activate   # ya da pyenv
pip install -r requirements.txt
```

## Çalıştırma (ilerledikçe doldurulacak)
```bash
# 1) Veriyi çek
python src/fetch_shots.py
# 2) Modeli eğit
python src/train_model.py
# 3) Arayüzü başlat
streamlit run app/main.py
```

## Veri kaynağı & lisans
Data provided by [StatsBomb Open Data](https://github.com/statsbomb/open-data),
CC BY-SA 4.0. StatsBomb'a atıf zorunludur.
