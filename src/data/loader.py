import pandas as pd
import numpy as np


def encode_result(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "H"
    elif row["home_goals"] < row["away_goals"]:
        return "A"
    return "D"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["result_encoded"] = df.apply(encode_result, axis=1)

    df["home_win"] = (df["result_encoded"] == "H").astype(int)
    df["draw"] = (df["result_encoded"] == "D").astype(int)
    df["away_win"] = (df["result_encoded"] == "A").astype(int)

    df["over_2_5"] = (df["total_goals"] > 2.5).astype(int)
    df["btts"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

    df["implied_prob_home"] = 1.0 / df["odds_home"]
    df["implied_prob_draw"] = 1.0 / df["odds_draw"]
    df["implied_prob_away"] = 1.0 / df["odds_away"]

    margin = (
        df["implied_prob_home"] + df["implied_prob_draw"] + df["implied_prob_away"]
    )
    df["fair_prob_home"] = df["implied_prob_home"] / margin
    df["fair_prob_draw"] = df["implied_prob_draw"] / margin
    df["fair_prob_away"] = df["implied_prob_away"] / margin

    return df


def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    home = (
        df.groupby("home_team")
        .agg(
            home_games=("home_goals", "count"),
            home_goals_scored=("home_goals", "sum"),
            home_goals_conceded=("away_goals", "sum"),
            home_points=(
                "result_encoded",
                lambda x: sum(3 if r == "H" else 1 if r == "D" else 0 for r in x),
            ),
            home_shots=("home_shots", "sum"),
            home_shots_target=("home_shots_target", "sum"),
        )
        .reset_index()
        .rename(columns={"home_team": "team"})
    )

    away = (
        df.groupby("away_team")
        .agg(
            away_games=("away_goals", "count"),
            away_goals_scored=("away_goals", "sum"),
            away_goals_conceded=("home_goals", "sum"),
            away_points=(
                "result_encoded",
                lambda x: sum(3 if r == "A" else 1 if r == "D" else 0 for r in x),
            ),
            away_shots=("away_shots", "sum"),
            away_shots_target=("away_shots_target", "sum"),
        )
        .reset_index()
        .rename(columns={"away_team": "team"})
    )

    stats = home.merge(away, on="team", how="outer").fillna(0)
    stats["games"] = stats["home_games"] + stats["away_games"]
    stats["goals_scored"] = stats["home_goals_scored"] + stats["away_goals_scored"]
    stats["goals_conceded"] = (
        stats["home_goals_conceded"] + stats["away_goals_conceded"]
    )
    stats["points"] = stats["home_points"] + stats["away_points"]
    stats["goals_scored_avg"] = stats["goals_scored"] / stats["games"]
    stats["goals_conceded_avg"] = stats["goals_conceded"] / stats["games"]
    return stats


def build_recent_form(df: pd.DataFrame, team: str, date: pd.Timestamp, n: int = 5):
    team_matches = df[
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < date)
    ].tail(n)

    if team_matches.empty:
        return {"form_points": 0, "form_goals_scored": 0, "form_goals_conceded": 0}

    points = 0
    gs = 0
    gc = 0
    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            gs += m["home_goals"]
            gc += m["away_goals"]
            if m["result_encoded"] == "H":
                points += 3
            elif m["result_encoded"] == "D":
                points += 1
        else:
            gs += m["away_goals"]
            gc += m["home_goals"]
            if m["result_encoded"] == "A":
                points += 3
            elif m["result_encoded"] == "D":
                points += 1

    return {
        "form_points": points / max(len(team_matches), 1),
        "form_goals_scored": gs / max(len(team_matches), 1),
        "form_goals_conceded": gc / max(len(team_matches), 1),
    }
