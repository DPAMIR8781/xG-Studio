"""
Seçilen lig/sezon listesi — üç kişi de TAM aynı veriyi çeksin diye tek kaynak burası.
Değerler StatsBomb sb.competitions() çıktısından alınmıştır.

Kullanım:
    from config.leagues import LEAGUES
    from statsbombpy import sb
    for lg in LEAGUES:
        matches = sb.matches(competition_id=lg["competition_id"],
                             season_id=lg["season_id"])
        # ... her maçın şutlarını topla
"""

LEAGUES = [
    # ---- La Liga (18 sezon — belkemiği, Messi çağı) ----
    {"competition_id": 11, "season_id": 278, "name": "La Liga 1973/1974"},
    {"competition_id": 11, "season_id": 37,  "name": "La Liga 2004/2005"},
    {"competition_id": 11, "season_id": 38,  "name": "La Liga 2005/2006"},
    {"competition_id": 11, "season_id": 39,  "name": "La Liga 2006/2007"},
    {"competition_id": 11, "season_id": 40,  "name": "La Liga 2007/2008"},
    {"competition_id": 11, "season_id": 41,  "name": "La Liga 2008/2009"},
    {"competition_id": 11, "season_id": 21,  "name": "La Liga 2009/2010"},
    {"competition_id": 11, "season_id": 22,  "name": "La Liga 2010/2011"},
    {"competition_id": 11, "season_id": 23,  "name": "La Liga 2011/2012"},
    {"competition_id": 11, "season_id": 24,  "name": "La Liga 2012/2013"},
    {"competition_id": 11, "season_id": 25,  "name": "La Liga 2013/2014"},
    {"competition_id": 11, "season_id": 26,  "name": "La Liga 2014/2015"},
    {"competition_id": 11, "season_id": 27,  "name": "La Liga 2015/2016"},
    {"competition_id": 11, "season_id": 2,   "name": "La Liga 2016/2017"},
    {"competition_id": 11, "season_id": 1,   "name": "La Liga 2017/2018"},
    {"competition_id": 11, "season_id": 4,   "name": "La Liga 2018/2019"},
    {"competition_id": 11, "season_id": 42,  "name": "La Liga 2019/2020"},
    {"competition_id": 11, "season_id": 90,  "name": "La Liga 2020/2021"},

    # ---- Büyük ligler (mevcut sezonlar — hacim) ----
    {"competition_id": 2,  "season_id": 44,  "name": "Premier League 2003/2004"},
    {"competition_id": 2,  "season_id": 27,  "name": "Premier League 2015/2016"},
    {"competition_id": 7,  "season_id": 27,  "name": "Ligue 1 2015/2016"},
    {"competition_id": 7,  "season_id": 108, "name": "Ligue 1 2021/2022"},
    {"competition_id": 7,  "season_id": 235, "name": "Ligue 1 2022/2023"},
    {"competition_id": 12, "season_id": 27,  "name": "Serie A 2015/2016"},
    {"competition_id": 9,  "season_id": 27,  "name": "1. Bundesliga 2015/2016"},
    {"competition_id": 9,  "season_id": 281, "name": "1. Bundesliga 2023/2024"},

    # ---- Şampiyonlar Ligi (18 sezon — çoğunlukla final/eleme, düşük hacim ama prestij) ----
    {"competition_id": 16, "season_id": 276, "name": "Champions League 1970/1971"},
    {"competition_id": 16, "season_id": 71,  "name": "Champions League 1971/1972"},
    {"competition_id": 16, "season_id": 277, "name": "Champions League 1972/1973"},
    {"competition_id": 16, "season_id": 76,  "name": "Champions League 1999/2000"},
    {"competition_id": 16, "season_id": 44,  "name": "Champions League 2003/2004"},
    {"competition_id": 16, "season_id": 37,  "name": "Champions League 2004/2005"},
    {"competition_id": 16, "season_id": 39,  "name": "Champions League 2006/2007"},
    {"competition_id": 16, "season_id": 41,  "name": "Champions League 2008/2009"},
    {"competition_id": 16, "season_id": 21,  "name": "Champions League 2009/2010"},
    {"competition_id": 16, "season_id": 22,  "name": "Champions League 2010/2011"},
    {"competition_id": 16, "season_id": 23,  "name": "Champions League 2011/2012"},
    {"competition_id": 16, "season_id": 24,  "name": "Champions League 2012/2013"},
    {"competition_id": 16, "season_id": 25,  "name": "Champions League 2013/2014"},
    {"competition_id": 16, "season_id": 26,  "name": "Champions League 2014/2015"},
    {"competition_id": 16, "season_id": 27,  "name": "Champions League 2015/2016"},
    {"competition_id": 16, "season_id": 2,   "name": "Champions League 2016/2017"},
    {"competition_id": 16, "season_id": 1,   "name": "Champions League 2017/2018"},
    {"competition_id": 16, "season_id": 4,   "name": "Champions League 2018/2019"},

    # ---- Uluslararası (çeşitlilik) ----
    {"competition_id": 43, "season_id": 3,   "name": "FIFA World Cup 2018"},
    {"competition_id": 43, "season_id": 106, "name": "FIFA World Cup 2022"},
    {"competition_id": 55, "season_id": 43,  "name": "UEFA Euro 2020"},
    {"competition_id": 55, "season_id": 282, "name": "UEFA Euro 2024"},

    # ---- Kadın futbolu (çeşitlilik + iki cinsiyet) ----
    {"competition_id": 37, "season_id": 4,   "name": "FA WSL 2018/2019"},
    {"competition_id": 37, "season_id": 42,  "name": "FA WSL 2019/2020"},
    {"competition_id": 37, "season_id": 90,  "name": "FA WSL 2020/2021"},
    {"competition_id": 37, "season_id": 281, "name": "FA WSL 2023/2024"},
    {"competition_id": 72, "season_id": 30,  "name": "Women's World Cup 2019"},
    {"competition_id": 72, "season_id": 107, "name": "Women's World Cup 2023"},
]

# İsteğe bağlı, sonradan eklenebilir (düşük hacimli ama tarihsel/çeşitlilik):
#   Serie A 1986/1987   -> cid=12 sid=86   (Maradona dönemi, az maç)
#   Eski Dünya Kupaları -> cid=43 sid=269/270/272/51/54/55 (1958-1990, seyrek)
