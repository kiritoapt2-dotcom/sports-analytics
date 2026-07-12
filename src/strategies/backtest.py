import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from src.strategies.kelly import calculate_kelly, estimate_ruin_probability


@dataclass
class BacktestResult:
    total_bets: int = 0
    won_bets: int = 0
    lost_bets: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    profit: float = 0.0
    final_bankroll: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    kelly_fraction: float = 0.0
    avg_edge: float = 0.0
    avg_odds: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Total Bets:     {self.total_bets}",
            f"Won:            {self.won_bets}",
            f"Lost:           {self.lost_bets}",
            f"Win Rate:       {self.win_rate:.2%}",
            f"Profit:         ${self.profit:.2f}",
            f"ROI:            {self.roi:.2%}",
            f"Final Bankroll: ${self.final_bankroll:.2f}",
            f"Max Drawdown:   {self.max_drawdown:.2%}",
            f"Sharpe Ratio:   {self.sharpe_ratio:.2f}",
            f"Avg Edge:       {self.avg_edge:.2%}",
            f"Avg Odds:       {self.avg_odds:.2f}",
            "-" * 60,
        ]
        return "\n".join(lines)


def backtest_strategy(
    df: pd.DataFrame,
    model_predictions: pd.DataFrame,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    stake_fn: Callable = None,
) -> BacktestResult:
    result = BacktestResult()
    result.final_bankroll = bankroll
    equity = [bankroll]
    trades = []

    merged = df.merge(
        model_predictions,
        on=["home_team", "away_team", "date"],
        how="inner",
        suffixes=("", "_pred"),
    )

    for _, row in merged.iterrows():
        actual = row["result_encoded"]
        home_prob = row.get("pred_home_win", row.get("home_win_prob", 0))
        draw_prob = row.get("pred_draw", row.get("draw_prob", 0))
        away_prob = row.get("pred_away_win", row.get("away_win_prob", 0))

        outcomes = [
            ("H", home_prob, row.get("odds_home", 2.0)),
            ("D", draw_prob, row.get("odds_draw", 3.5)),
            ("A", away_prob, row.get("odds_away", 3.0)),
        ]

        best_outcome = max(outcomes, key=lambda x: x[1])
        outcome, prob, odds = best_outcome

        kelly = calculate_kelly(
            model_prob=prob,
            decimal_odds=odds,
            bankroll=result.final_bankroll,
            fraction=kelly_fraction,
            min_edge=min_edge,
        )

        if kelly.action == "BET":
            result.total_bets += 1
            if outcome == actual:
                profit = kelly.stake * (odds - 1)
                result.won_bets += 1
            else:
                profit = -kelly.stake
                result.lost_bets += 1

            result.final_bankroll += profit
            result.profit += profit
            avg_edge = result.avg_edge * (result.total_bets - 1) / result.total_bets
            result.avg_edge += kelly.edge / result.total_bets
            avg_odds = result.avg_odds * (result.total_bets - 1) / result.total_bets
            result.avg_odds += odds / result.total_bets

            trades.append(
                {
                    "date": row["date"],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "bet": outcome,
                    "odds": odds,
                    "prob": prob,
                    "stake": kelly.stake,
                    "edge": kelly.edge,
                    "ev": kelly.expected_value,
                    "profit": profit,
                    "actual": actual,
                    "bankroll": result.final_bankroll,
                }
            )
            equity.append(result.final_bankroll)

    if result.total_bets > 0:
        result.win_rate = result.won_bets / result.total_bets
        result.roi = result.profit / bankroll
        result.kelly_fraction = kelly_fraction

        if len(equity) > 1:
            peak = equity[0]
            dd = 0.0
            for v in equity:
                if v > peak:
                    peak = v
                dd = max(dd, (peak - v) / peak)
            result.max_drawdown = dd

            returns = np.diff(equity) / equity[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)

    result.trades = trades
    result.equity_curve = equity
    return result


def compare_strategies(
    df: pd.DataFrame,
    model_predictions: pd.DataFrame,
    bankroll: float = 1000.0,
) -> dict:
    strategies = {
        "flat_2%": lambda br: (0, None),
        "full_kelly": lambda br: (1.0, None),
        "half_kelly": lambda br: (0.5, None),
        "quarter_kelly": lambda br: (0.25, None),
        "tenth_kelly": lambda br: (0.1, None),
    }

    results = {}
    for name, _ in strategies.items():
        frac = float(name.split("_")[1].replace("kelly", ""))
        if frac == 0:
            frac = 0.25
        result = backtest_strategy(
            df=df,
            model_predictions=model_predictions,
            bankroll=bankroll,
            kelly_fraction=frac,
        )
        results[name] = result

    return results
