# xG Stüdyo — Durum Notu (freeze TAM veride bitti + Streamlit demo)

## Bu session'da tamamlananlar

### Freeze veri katmanı TAM veride çıkarıldı (artık prototip değil)
- 54 lig-sezonun tamamı, 2.989 maç, 75.349 şut için freeze_frame feature'ları çekildi
  (statsbombpy, açık veri): opp_in_cone, gk_dist_goal, gk_dist_ball.
- Çıktı: data/shots_freeze.parquet (merged, 75.349 satır, baseline ile birebir).
- Freeze eksik oranı sadece %1.23 (o şutlarda freeze frame yok → NaN, XGBoost doğal işler).

### İki bug yakalandı ve çözüldü
1. Disk doldu: statsbombpy /tmp'ye maç başına ~570MB HTTP cache yazıyor. Çıkarımı
   append-only batch'lere böldüm (data/ff_parts/), her tur /tmp/*.sqlite temizliği.
2. Bool bug: extraction'da bool(NaN) is True yüzünden under_pressure/shot_first_time/
   shot_one_on_one HEP True olmuştu. Çözüm: yeniden fetch YOK — baseline parquet'in
   (doğru bool'lar) freeze kolonlarıyla join'i → data/shots_freeze_merged.parquet.

### Freeze modeli TAM veride yeniden eğitildi
- Script: src/train_freeze_merged.py (train_model.py pipeline'ı + 3 freeze feature).
  OneHot → XGBoost → isotonic, aynı split (RS=42, test=0.2, stratified).
- Sonuçlar (aynı 15.070 şutluk test seti):
  - baseline 0.8017 AUC · freeze 0.8183 AUC · StatsBomb 0.8191 AUC
  - LİFT: +0.0166 AUC. StatsBomb'a fark 0.0174 → 0.0008 (açığın ~%95'i kapandı).
  - Kalibrasyon hizalı: freeze ort.tahmin 0.1120 ↔ gol oranı 0.1119 ↔ baseline 0.1121.
    Eski prototipin "seviye farkı" artefaktı YOK — iki model artık aynı ölçekte.
- Pressure etkisi artık çalışıyor (bool fix sonrası). Penaltı: gerçek profil (gk=NaN,
  play_pattern='Other') ile baseline 0.83 / freeze 0.81 — ikisi de doğru.
- Çıktı: models/xg_model_freeze.joblib (TAM veri sürümü, prototipin yerine).

## Streamlit demo
- Karar: iki modeli birlikte tut. Tek arayüz, baseline vs freeze YAN YANA, tıkladıkça
  fark sunumda canlı görülür. Savunmacı sürükleme İPTAL (0/1/2 seçici kaldı).
- Deliverable'lar (/mnt/user-data/outputs/):
  - xg_demo.html — standalone interaktif demo (çift tıkla aç, sunucu yok). İki saha,
    ortak şut, gerçek model ızgarası gömülü, penaltı tuşu, AI yorum havuzu. Paylaşılabilir.
  - streamlit_app.py + requirements.txt — gerçek deploy. İki modeli joblib'den yükler,
    sahaya tıkla, baseline vs freeze + delta, penaltı butonu, AI yorum katmanı.
  - xg_model_freeze.joblib — yeni tam-veri freeze modeli.
- AI yorum katmanı: senaryo başına VARYANT HAVUZU (MAX_VARIANTS=3) cache'lenir.
  GOOGLE_API_KEY varsa Gemini Flash (temperature 0.95), yoksa yerel havuz. Aynı şut →
  havuzda döner; farklı şut → farklı yorum; havuz dolunca yeni çağrı yok (kota minimal).
- Penaltı butonu: şutu noktaya getirir, shot_type=Penalty. Demo'da sabit (0.83/0.81),
  app'te modele canlı sorar (doğru penaltı profiliyle).

## SIRADAKİ ADIMLAR
- Streamlit'i GCP VM'e deploy (slayt 8 mimarisi). xg_demo.html GitHub Pages → tek-tık link.
- Gemini API key bağla, sunum öncesi demo senaryolarını "ön-ısıt" (canlı sunumda sıfır kota).
- İsteğe bağlı: BigQuery analiz katmanı, Power BI görseller (slayt 9/10).

## Teknik notlar
- Freeze extraction tekrar gerekirse bool bug için: bool(x) is True yerine x is True kullan,
  ya da baseline'a join et (merged yöntemi, daha sağlam). statsbombpy /tmp cache'ini temizle.
- Modeller joblib.load(...)['model'] ile yüklenir; dict ayrıca num/bool/cat listeleri +
  OOD guard + metrikler içerir.
