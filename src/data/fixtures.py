import pandas as pd
import requests
from datetime import datetime, date
from typing import Optional
from io import StringIO

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
COLUMN_MAP = {
    "Div": "league_code",
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "B365>2.5": "odds_over_2_5",
    "B365<2.5": "odds_under_2_5",
}


def fetch_fixtures(target_date: Optional[date] = None) -> pd.DataFrame:
    if target_date is None:
        target_date = date.today()

    resp = requests.get(FIXTURES_URL, timeout=15)
    resp.raise_for_status()

    text = resp.text.replace("\ufeff", "").replace("ï»¿", "")
    df = pd.read_csv(StringIO(text), low_memory=False)
    cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=cols)
    keep = list(cols.values())
    df = df[[c for c in keep if c in df.columns]]

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df["date_only"] = df["date"].dt.date
    df = df[df["date_only"] == target_date].copy()

    for c in ["odds_home", "odds_draw", "odds_away", "odds_over_2_5", "odds_under_2_5"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


def get_fixtures_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No hay partidos programados para hoy."

    leagues = df.groupby("league_code").size()
    lines = [f"Total: {len(df)} partidos en {len(leagues)} ligas\n"]
    for code, count in leagues.items():
        lines.append(f"  {code}: {count}")
    return "\n".join(lines)
