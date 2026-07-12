import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss
import joblib
from pathlib import Path
from config import DATA_DIR


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for XGBoost from raw match data."""
    rows = []
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        past = df_sorted[df_sorted["date"] < date]

        def team_recent(team: str, n: int = 10):
            tm = past[
                (past["home_team"] == team) | (past["away_team"] == team)
            ].tail(n)
            return tm

        hr = team_recent(home)
        ar = team_recent(away)

        feats = {}

        feats["league_avg_home_goals"] = df_sorted[
            df_sorted["date"] < date
        ]["home_goals"].mean() if len(past) > 0 else 0
        feats["league_avg_away_goals"] = df_sorted[
            df_sorted["date"] < date
        ]["away_goals"].mean() if len(past) > 0 else 0

        for prefix, tm in [("home", hr), ("away", ar)]:
            if len(tm) == 0:
                feats[f"{prefix}_points_last_5"] = 0
                feats[f"{prefix}_goals_scored_last_5"] = 0
                feats[f"{prefix}_goals_conceded_last_5"] = 0
                feats[f"{prefix}_shots_last_5"] = 0
                feats[f"{prefix}_win_pct_10"] = 0
                continue

            last5 = tm.tail(5)
            points = 0
            gs = 0
            gc = 0
            shots = 0
            for _, m in last5.iterrows():
                if m["home_team"] == team if prefix == "home" else m["away_team"] == team:
                    gs += m["home_goals"] if prefix == "home" else m["away_goals"]
                    gc += m["away_goals"] if prefix == "home" else m["home_goals"]
                    shots += m.get("home_shots" if prefix == "home" else "away_shots", 0)
                    if m["result_encoded"] == ("H" if prefix == "home" else "A"):
                        points += 3
                    elif m["result_encoded"] == "D":
                        points += 1
                else:
                    gs += m["away_goals"] if prefix == "home" else m["home_goals"]
                    gc += m["home_goals"] if prefix == "home" else m["away_goals"]
                    shots += m.get("away_shots" if prefix == "home" else "home_shots", 0)
                    if m["result_encoded"] == ("A" if prefix == "home" else "H"):
                        points += 3
                    elif m["result_encoded"] == "D":
                        points += 1

            feats[f"{prefix}_points_last_5"] = points / max(len(last5), 1)
            feats[f"{prefix}_goals_scored_last_5"] = gs / max(len(last5), 1)
            feats[f"{prefix}_goals_conceded_last_5"] = gc / max(len(last5), 1)
            feats[f"{prefix}_shots_last_5"] = shots / max(len(last5), 1)
            feats[f"{prefix}_win_pct_10"] = sum(
                1 for _, m in tm.iterrows()
                if (m["home_team"] == team and m["result_encoded"] == "H")
                or (m["away_team"] == team and m["result_encoded"] == "A")
            ) / max(len(tm), 1) * 100

        feats["elo_diff"] = feats.get("home_win_pct_10", 0) - feats.get("away_win_pct_10", 0)
        feats["goal_diff_last_5"] = feats["home_goals_scored_last_5"] - feats["home_goals_conceded_last_5"] - feats["away_goals_scored_last_5"] + feats["away_goals_conceded_last_5"]

        h2h = past[
            ((past["home_team"] == home) & (past["away_team"] == away))
            | ((past["home_team"] == away) & (past["away_team"] == home))
        ].tail(5)
        feats["h2h_home_advantage"] = sum(
            1 for _, m in h2h.iterrows()
            if (m["home_team"] == home and m["result_encoded"] == "H")
            or (m["away_team"] == home and m["result_encoded"] == "A")
        ) / max(len(h2h), 1) * 100

        feats["home_goal_diff_rolling"] = feats["home_goals_scored_last_5"] - feats["home_goals_conceded_last_5"]
        feats["away_goal_diff_rolling"] = feats["away_goals_scored_last_5"] - feats["away_goals_conceded_last_5"]

        rows.append(
            {
                **feats,
                "home_team": home,
                "away_team": away,
                "date": date,
                "target_home": row["home_win"],
                "target_draw": row["draw"],
                "target_away": row["away_win"],
                "odds_home": row.get("odds_home", 2.0),
                "odds_draw": row.get("odds_draw", 3.5),
                "odds_away": row.get("odds_away", 3.0),
            }
        )

    return pd.DataFrame(rows)


class XGBoostPredictor:
    def __init__(self):
        self.model_home = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1,
        )
        self.model_draw = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.model_away = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1,
        )
        self.feature_cols = []
        self.is_fitted = False

    def fit(self, df_features: pd.DataFrame):
        feature_cols = [
            c for c in df_features.columns
            if c not in [
                "home_team", "away_team", "date",
                "target_home", "target_draw", "target_away",
                "odds_home", "odds_draw", "odds_away",
            ]
        ]
        self.feature_cols = feature_cols

        X = df_features[feature_cols].fillna(0).values

        y_home = (df_features["target_home"] == 1).astype(int).values
        y_draw = (df_features["target_draw"] == 1).astype(int).values
        y_away = (df_features["target_away"] == 1).astype(int).values

        self.model_home.fit(X, y_home)
        self.model_draw.fit(X, y_draw)
        self.model_away.fit(X, y_away)
        self.is_fitted = True

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = df_features[self.feature_cols].fillna(0).values
        prob_home = self.model_home.predict_proba(X)[:, 1]
        prob_draw = self.model_draw.predict_proba(X)[:, 1]
        prob_away = self.model_away.predict_proba(X)[:, 1]

        total = prob_home + prob_draw + prob_away
        return np.column_stack([
            prob_home / total,
            prob_draw / total,
            prob_away / total,
        ])

    def save(self, path: Path = DATA_DIR / "xgboost_model.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = DATA_DIR / "xgboost_model.joblib"):
        return joblib.load(path)


def train_xgboost_model(df: pd.DataFrame) -> XGBoostPredictor:
    print("Engineering features...")
    df_features = engineer_features(df)
    df_features = df_features.dropna(
        subset=[
            "target_home", "target_draw", "target_away",
            "odds_home", "odds_draw", "odds_away",
        ]
    )

    print(f"Training on {len(df_features)} samples...")
    model = XGBoostPredictor()
    model.fit(df_features)
    model.save()
    return model
