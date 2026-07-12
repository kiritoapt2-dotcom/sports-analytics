import pandas as pd
import requests
from pathlib import Path
from time import sleep
from config import FOOTBALL_DATA_URL, DATA_DIR, LEAGUES, SEASONS

COLUMN_MAP = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "HTR": "result_ht",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
    "BWH": "odds_home_bw",
    "BWD": "odds_draw_bw",
    "BWA": "odds_away_bw",
    "PSH": "odds_home_ps",
    "PSD": "odds_draw_ps",
    "PSA": "odds_away_ps",
    "MaxH": "odds_home_max",
    "MaxD": "odds_draw_max",
    "MaxA": "odds_away_max",
    "AvgH": "odds_home_avg",
    "AvgD": "odds_draw_avg",
    "AvgA": "odds_away_avg",
}


def download_league_data(league: str, season: str) -> pd.DataFrame | None:
    url = FOOTBALL_DATA_URL.format(season=season, league=league)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(url)
        return df
    except Exception:
        return None


def load_all_data() -> pd.DataFrame:
    cache_file = DATA_DIR / "all_matches.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    all_dfs = []
    for league_code, league_name in LEAGUES.items():
        for season in SEASONS:
            df = download_league_data(league_code, season)
            if df is not None and not df.empty:
                df["league_code"] = league_code
                df["league_name"] = league_name
                df["season"] = season
                all_dfs.append(df)
            sleep(0.3)

    if not all_dfs:
        raise ValueError("No data downloaded")

    combined = pd.concat(all_dfs, ignore_index=True)

    cols = [c for c in COLUMN_MAP if c in combined.columns]
    combined = combined[cols + ["league_code", "league_name", "season"]].rename(
        columns=COLUMN_MAP
    )

    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=["date", "home_team", "away_team"])
    combined = combined.sort_values("date").reset_index(drop=True)

    combined.to_parquet(cache_file)
    return combined
