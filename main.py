#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from time import sleep
import requests

from config import DATA_DIR, OUTPUT_DIR, LEAGUES, SEASONS, KELLY_FRACTION, MIN_EDGE
from src.data.loader import add_features, build_team_stats
from src.models.poisson import train_poisson_model, DixonColesModel
from src.strategies.backtest import backtest_strategy, compare_strategies
from src.strategies.kelly import calculate_kelly
from src.analysis.reporter import (
    build_match_report,
    build_daily_report,
    build_performance_report,
)

COLUMN_MAP = {
    "Div": "league", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_target", "AST": "away_shots_target",
    "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
}

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download(force: bool = False) -> pd.DataFrame:
    cache = DATA_DIR / "all_matches.parquet"
    if cache.exists() and not force:
        print("  Loading cached data...")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,} matches from cache")
        return df

    all_dfs = []
    for league_code, league_name in LEAGUES.items():
        count = 0
        for season in SEASONS[-5:]:
            url = FOOTBALL_DATA_URL.format(season=season, league=league_code)
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(url)
                df["league_code"] = league_code
                df["league_name"] = league_name
                df["season"] = season
                all_dfs.append(df)
                count += len(df)
            except Exception:
                pass
            sleep(0.15)
        print(f"  {league_code:4s} ({league_name:25s}): {count} matches")

    combined = pd.concat(all_dfs, ignore_index=True)
    cols = {k: v for k, v in COLUMN_MAP.items() if k in combined.columns}
    combined = combined.rename(columns=cols)
    keep_cols = list(cols.values()) + ["league_code", "league_name", "season"]
    combined = combined[[c for c in keep_cols if c in combined.columns]]
    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=["date", "home_team", "away_team"]).sort_values("date").reset_index(drop=True)
    combined.to_parquet(cache)
    print(f"\n  Total: {len(combined):,} matches from {combined['date'].min().date()} to {combined['date'].max().date()}")
    return combined


def train(df: pd.DataFrame):
    print("\n--- TRAINING PHASE ---")
    df = add_features(df)
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    train_df = df[df["date"] < cutoff].copy()

    print(f"  Training on {len(train_df):,} matches ({len(train_df['league_code'].unique())} leagues)")
    model = train_poisson_model(train_df, reg_lambda=1.0, min_games=10)
    print(f"  Teams modeled: {len(model.team_params)}")

    test_df = df[df["date"] >= cutoff].copy()
    correct, total = 0, 0
    for _, row in test_df.iterrows():
        probs = model.predict_match(row["home_team"], row["away_team"])
        pred = max([("H", probs["home_win"]), ("D", probs["draw"]), ("A", probs["away_win"])], key=lambda x: x[1])[0]
        if pred == row["result_encoded"]:
            correct += 1
        total += 1
    acc = correct / total if total else 0
    home_rate = df[df["date"] >= cutoff]["home_win"].mean()
    print(f"  Accuracy: {correct}/{total} = {acc:.1%} (baseline: {home_rate:.1%} always-home)")

    model.metadata_ = {"accuracy": acc, "baseline": home_rate, "train_size": len(train_df), "test_size": total}
    return model, df


def predict_upcoming(model: DixonColesModel, df: pd.DataFrame) -> list[dict]:
    print("\n--- FINDING VALUE BETS ---")
    last_date = df["date"].max()
    recent = df[df["date"] > last_date - pd.Timedelta(days=90)]
    teams = sorted(set(recent["home_team"].unique()) | set(recent["away_team"].unique()))

    predictions = []
    for league in df["league_code"].unique():
        ldf = df[df["league_code"] == league]
        lteams = sorted(set(ldf["home_team"].unique()) | set(ldf["away_team"].unique()))
        lteams = [t for t in lteams if t in teams]

        for i, home in enumerate(lteams):
            for away in lteams[i + 1:]:
                h2h = ldf[((ldf["home_team"] == home) & (ldf["away_team"] == away))].tail(3)
                if len(h2h) < 2:
                    continue
                days_since = (last_date - h2h["date"].max()).days
                if days_since < 4 or days_since > 45:
                    continue

                probs = model.predict_match(home, away)
                row = h2h.iloc[-1]
                odds = {"odds_home": row["odds_home"], "odds_draw": row["odds_draw"], "odds_away": row["odds_away"]}

                bet_options = [
                    ("1", probs["home_win"], odds["odds_home"]),
                    ("X", probs["draw"], odds["odds_draw"]),
                    ("2", probs["away_win"], odds["odds_away"]),
                ]
                best = max(bet_options, key=lambda x: x[1])
                kelly = calculate_kelly(best[1], best[2], 1000, KELLY_FRACTION, MIN_EDGE)

                predictions.append({
                    "league": ldf.iloc[0]["league_name"],
                    "home_team": home, "away_team": away,
                    "predicted": best[0], "prob": best[1],
                    "odds": best[2], "edge": kelly.edge,
                    "ev": kelly.expected_value, "stake": kelly.stake,
                    "action": kelly.action, "score": probs["most_likely_score"],
                    "xg_home": probs["expected_goals_home"], "xg_away": probs["expected_goals_away"],
                    "over_2_5": probs["over_2_5"], "btts": probs["btts"],
                })

    predictions.sort(key=lambda x: -x["edge"])
    value = [p for p in predictions if p["action"] == "BET"]
    print(f"  Total fixtures analyzed: {len(predictions)}")
    print(f"  Value bets found: {len(value)}")

    if value:
        from tabulate import tabulate
        tbl = [[p["home_team"][:14], p["away_team"][:14], p["predicted"],
                f"{p['prob']:.0%}", f"{p['odds']:.2f}", f"{p['edge']:.1%}",
                f"${p['stake']:.1f}", p["score"]] for p in value[:30]]
        print(tabulate(tbl, headers=["Home", "Away", "Pick", "Prob", "Odds", "Edge", "Stake", "Score"], tablefmt="grid"))
    return predictions


def run_backtest(model: DixonColesModel, df: pd.DataFrame):
    print("\n--- BACKTEST (12 months) ---")
    df = add_features(df)
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    test = df[df["date"] >= cutoff].copy()

    pred_rows = []
    for _, row in test.iterrows():
        probs = model.predict_match(row["home_team"], row["away_team"])
        pred_rows.append({
            "home_team": row["home_team"], "away_team": row["away_team"], "date": row["date"],
            "pred_home_win": probs["home_win"], "pred_draw": probs["draw"], "pred_away_win": probs["away_win"],
        })
    pred_df = pd.DataFrame(pred_rows)

    for frac_name, frac_val in [("Full Kelly", 1.0), ("Half Kelly", 0.5), ("Quarter Kelly", 0.25), ("Tenth Kelly", 0.1)]:
        result = backtest_strategy(test, pred_df, bankroll=1000.0, kelly_fraction=frac_val)
        print(f"  {frac_name:15s}: bets={result.total_bets:4d} WR={result.win_rate:.1%} "
              f"ROI={result.roi:.1%} final=${result.final_bankroll:.0f} DD={result.max_drawdown:.1%}")

    result = backtest_strategy(test, pred_df, bankroll=1000.0, kelly_fraction=0.25)
    with open(OUTPUT_DIR / "backtest_report.txt", "w") as f:
        f.write(build_performance_report(result))
    return result


def main():
    print("=" * 60)
    print("  SPORTS BETTING ANALYTICS ENGINE v1.0")
    print("  Model: Dixon-Coles Bivariate Poisson")
    print("=" * 60)

    df = download(force=False)
    model, df = train(df)
    predictions = predict_upcoming(model, df)
    bt = run_backtest(model, df)

    with open(OUTPUT_DIR / "predictions.txt", "w") as f:
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Model accuracy: {model.metadata_['accuracy']:.1%}\n\n")
        for p in predictions[:50]:
            f.write(f"{p['league']:25s} {p['home_team']:20s} vs {p['away_team']:20s} | "
                    f"Pick: {p['predicted']} @ {p['odds']:.2f} | Prob: {p['prob']:.0%} "
                    f"Edge: {p['edge']:.1%} | {p['action']}\n")

    print(f"\n  Reports saved to {OUTPUT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
