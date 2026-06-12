# xG Stüdyo — Manşet Sağlamlık Kontrolü

**Tarih:** 2026-06-12 · **Veri:** data/shots_freeze.parquet (75.349 şut, 2.989 maç, gol oranı 0.1119)

## Yöntem
- Match-grouped 5-fold OOF (GroupKFold, maç ID'ye göre). Her şutun tahmini o maçı görmemiş modelden gelir → leakage'siz.
- Cluster bootstrap: GA'lar için satır değil MAÇ resample edilir (n=2000). Şutlar maç içinde korele.
- AUC izotonik kalibrasyona göre değişmez → lift/gap ham skorda, sonuç kalibre modelle aynı.
- Pipeline: OneHotEncoder(ignore) → XGBoost(400,4,0.05), rs=42.

## Bölüm A — Split leakage testi (test=0.2, rs=42)
| model | row-stratified | match-grouped | Δ |
|---|---|---|---|
| baseline | 0.8019 | 0.8069 | −0.0050 |
| freeze | 0.8186 | 0.8212 | −0.0026 |
| statsbomb | 0.8191 | 0.8214 | −0.0024 |

Leakage yok: Δ'lar negatif (grouped daha yüksek, şişme değil). StatsBomb referansı freeze ile aynı miktarda kayıyor → fark sadece satır varyansı. Manşet split'ten bağımsız.

## Bölüm B — Match-grouped OOF + cluster bootstrap (n=2000)
| metrik | nokta | %95 GA | karar |
|---|---|---|---|
| baseline AUC | 0.8062 | [0.8010, 0.8116] | — |
| freeze AUC | 0.8203 | [0.8152, 0.8255] | — |
| statsbomb AUC | 0.8216 | [0.8165, 0.8270] | — |
| baseline Brier | 0.0799 | [0.0783, 0.0815] | — |
| freeze Brier | 0.0774 | [0.0759, 0.0789] | — |
| lift (freeze−base) | +0.0140 | [+0.0121, +0.0161] | GERÇEK (0 dışlanıyor) |
| gap (sb−freeze) | +0.0013 | [−0.0012, +0.0037] | AYIRT EDİLEMEZ (0 kapsanıyor) |
| Brier iyileşme | +0.0025 | [+0.0021, +0.0028] | GERÇEK |

## Sonuç
1. Lift sağlam: +0.0140, GA sıfırı dışlıyor.
2. Gap istatistiksel sıfır: model StatsBomb'tan ayırt edilemiyor.

## Sunum cümlesi
"Freeze frame geometrisini ekleyince modelimiz StatsBomb'un kendi xG'sinden istatistiksel olarak ayırt edilemez hale geldi (gap +0.001, %95 GA sıfırı kapsıyor); lift +0.014 AUC ile anlamlı."

Not: CV lift'i (+0.0140) tek-split rakamından (+0.0166) bir tık düşük — iyimserlikten arınmış sürüm. Deck'e CV + GA konmalı.
