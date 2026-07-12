#!/usr/bin/env python3
import pandas as pd
import requests
from time import sleep
from io import StringIO
from config import LEAGUES, SEASONS, DATA_DIR

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
COLUMN_MAP = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
}

all_dfs = []
for league_code, league_name in LEAGUES.items():
    count = 0
    for season in SEASONS[-5:]:
        url = FOOTBALL_DATA_URL.format(season=season, league=league_code)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            text = resp.text.replace("\ufeff", "")
            df = pd.read_csv(StringIO(text), low_memory=False)
            df["league_code"] = league_code
            df["league_name"] = league_name
            df["season"] = season
            all_dfs.append(df)
            count += len(df)
        except Exception:
            pass
        sleep(0.15)
    print(f"  {league_code} ({league_name:25s}): {count} matches")

combined = pd.concat(all_dfs, ignore_index=True)
cols = {k: v for k, v in COLUMN_MAP.items() if k in combined.columns}
combined = combined.rename(columns=cols)
keep = list(cols.values())
for extra in ["league_code", "league_name", "season"]:
    if extra in combined.columns:
        keep.append(extra)
keep = list(dict.fromkeys(keep))
combined = combined[[c for c in keep if c in combined.columns]]
combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
combined = combined.dropna(subset=["date", "home_team", "away_team"])
combined = combined.sort_values("date").reset_index(drop=True)

out = DATA_DIR / "all_matches.parquet"
combined.to_parquet(out)
print(f"\nTotal: {len(combined):,} matches")
print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
print(f"File: {out}")
