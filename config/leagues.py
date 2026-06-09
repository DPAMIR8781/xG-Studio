"""
Seçilen lig/sezon listesi — üç kişi de TAM aynı veriyi çeksin diye tek kaynak burası.

Her giriş StatsBomb'un competition_id / season_id değerleridir.
`sb.competitions()` çıktısından doldurulacak (Claude ile birlikte kesinleştirip ekleyeceğiz).

Kullanım:
    from config.leagues import LEAGUES
    for lg in LEAGUES:
        matches = sb.matches(competition_id=lg["competition_id"],
                             season_id=lg["season_id"])
"""

LEAGUES = [
    # ÖRNEK FORMAT — gerçek değerlerle doldurulacak:
    # {"competition_id": 11,  "season_id": 90, "name": "La Liga 2020/2021"},
    # {"competition_id": 2,   "season_id": 27, "name": "Premier League 2015/2016"},
    # TODO: seçilen ligleri buraya ekle
]
