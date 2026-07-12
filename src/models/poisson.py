import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class TeamParams:
    attack: float
    defense: float


class DixonColesModel:
    def __init__(self, rho: float = 0.0):
        self.rho = rho
        self.team_params: dict[str, TeamParams] = {}
        self.league_avg_home: float = 0.0
        self.league_avg_away: float = 0.0
        self.home_advantage: float = 0.0

    def fit(self, df: pd.DataFrame, reg_lambda: float = 0.5, min_games: int = 5):
        team_counts = pd.concat([
            df.groupby("home_team").size(),
            df.groupby("away_team").size(),
        ]).groupby(level=0).sum()
        valid_teams = team_counts[team_counts >= min_games].index
        df = df[df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)]

        teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        n_teams = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        self.league_avg_home = df["home_goals"].mean()
        self.league_avg_away = df["away_goals"].mean()

        home_idx = df["home_team"].map(team_to_idx).values
        away_idx = df["away_team"].map(team_to_idx).values
        hg = df["home_goals"].values.astype(float)
        ag = df["away_goals"].values.astype(float)

        def neg_log_likelihood(params):
            home_adv = params[0]
            att = params[1 : 1 + n_teams]
            deff = params[1 + n_teams : 1 + 2 * n_teams]

            lambda_h = np.exp(home_adv + att[home_idx] - deff[away_idx])
            lambda_a = np.exp(att[away_idx] - deff[home_idx])

            ll = np.sum(poisson.logpmf(hg, lambda_h))
            ll += np.sum(poisson.logpmf(ag, lambda_a))

            if self.rho != 0:
                dc_mask = (hg <= 1) & (ag <= 1)
                if dc_mask.any():
                    tau = np.ones_like(hg)
                    m00 = (hg == 0) & (ag == 0)
                    m01 = (hg == 0) & (ag == 1)
                    m10 = (hg == 1) & (ag == 0)
                    m11 = (hg == 1) & (ag == 1)
                    tau[m00] = 1 - self.rho * lambda_h[m00] * lambda_a[m00]
                    tau[m01] = 1 + self.rho * lambda_h[m01]
                    tau[m10] = 1 + self.rho * lambda_a[m10]
                    tau[m11] = 1 + self.rho
                    tau = np.clip(tau, 1e-10, None)
                    ll += np.sum(np.log(tau[dc_mask]))

            reg = reg_lambda * np.sum(att**2) + reg_lambda * np.sum(deff**2)
            return -ll + reg

        init_params = np.zeros(1 + 2 * n_teams)
        init_params[0] = np.log(self.league_avg_home / self.league_avg_away) if self.league_avg_away > 0 else 0.1

        team_goals_home = df.groupby("home_team").agg(gs=("home_goals", "mean"), gc=("away_goals", "mean"))
        team_goals_away = df.groupby("away_team").agg(gs=("away_goals", "mean"), gc=("home_goals", "mean"))

        for i, team in enumerate(teams):
            att_val = 0.0
            def_val = 0.0
            if team in team_goals_home.index and team in team_goals_away.index:
                th = team_goals_home.loc[team]
                ta = team_goals_away.loc[team]
                gs_h = th["gs"]
                gs_a = ta["gs"]
                gc_h = ta["gc"]
                gc_a = th["gc"]
                gs_total = (gs_h + gs_a) / 2
                gc_total = (gc_h + gc_a) / 2
                att_val = np.log(max(gs_total, 0.1) / max(self.league_avg_home, 0.1))
                def_val = -np.log(max(gc_total, 0.1) / max(self.league_avg_away, 0.1))
            init_params[1 + i] = np.clip(att_val, -3, 3)
            init_params[1 + n_teams + i] = np.clip(def_val, -3, 3)

        bounds = [(None, None)] + [(-5, 5)] * (2 * n_teams)

        result = minimize(
            neg_log_likelihood,
            init_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        opt = result.x
        self.home_advantage = opt[0]
        for i, team in enumerate(teams):
            self.team_params[team] = TeamParams(
                attack=opt[1 + i], defense=opt[1 + n_teams + i]
            )

    def predict_match(self, home_team: str, away_team: str) -> dict:
        hp = self.team_params.get(home_team)
        ap = self.team_params.get(away_team)
        if hp is None:
            hp = TeamParams(attack=0.0, defense=0.0)
        if ap is None:
            ap = TeamParams(attack=0.0, defense=0.0)

        lambda_h = np.exp(self.home_advantage + hp.attack - ap.defense)
        lambda_a = np.exp(ap.attack - hp.defense)

        return self._compute_probs(lambda_h, lambda_a)

    def _compute_probs(self, lambda_h: float, lambda_a: float) -> dict:
        max_goals = 10
        i_idx, j_idx = np.meshgrid(
            np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij"
        )
        probs = poisson.pmf(i_idx, lambda_h) * poisson.pmf(j_idx, lambda_a)

        dc_mask = (i_idx <= 1) & (j_idx <= 1)
        tau = np.ones_like(probs)
        tau[0, 0] = 1 - self.rho * lambda_h * lambda_a
        tau[0, 1] = 1 + self.rho * lambda_h
        tau[1, 0] = 1 + self.rho * lambda_a
        tau[1, 1] = 1 + self.rho
        probs[dc_mask] *= tau[dc_mask]
        probs = probs / probs.sum()

        home_win = np.sum(probs * np.tril(np.ones_like(probs), k=-1))
        draw = np.trace(probs)
        away_win = np.sum(probs * np.triu(np.ones_like(probs), k=1))

        total = i_idx + j_idx
        over_2_5 = np.sum(probs[total > 2.5])
        btts = np.sum(probs[(i_idx > 0) & (j_idx > 0)])

        ml_idx = np.unravel_index(probs.argmax(), probs.shape)

        score_list = []
        for i in range(6):
            for j in range(6):
                if probs[i, j] > 0.003:
                    score_list.append((f"{i}-{j}", float(probs[i, j])))
        score_list.sort(key=lambda x: -x[1])

        return {
            "home_win": float(home_win),
            "draw": float(draw),
            "away_win": float(away_win),
            "over_2_5": float(over_2_5),
            "btts": float(btts),
            "expected_goals_home": float(lambda_h),
            "expected_goals_away": float(lambda_a),
            "most_likely_score": f"{ml_idx[0]}-{ml_idx[1]}",
            "score_probs": dict(score_list[:10]),
        }


def train_poisson_model(df: pd.DataFrame, reg_lambda: float = 0.5, min_games: int = 5) -> DixonColesModel:
    model = DixonColesModel(rho=0.0)
    model.fit(df, reg_lambda=reg_lambda, min_games=min_games)
    return model
