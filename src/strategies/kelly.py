import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KellyResult:
    edge: float
    kelly_fraction: float
    fractional_kelly: float
    stake: float
    expected_value: float
    action: str


def calculate_kelly(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = 0.25,
    min_edge: float = 0.03,
    max_stake_pct: float = 0.05,
    min_stake: float = 0.5,
) -> KellyResult:
    b = decimal_odds - 1
    q = 1 - model_prob
    edge = model_prob - (1 / decimal_odds)

    ev = model_prob * b - q
    if b > 0:
        kelly_full = (b * model_prob - q) / b
    else:
        kelly_full = 0.0

    kelly_full = max(kelly_full, 0.0)
    kelly_frac = kelly_full * fraction
    stake = min(kelly_frac * bankroll, bankroll * max_stake_pct)
    stake = max(stake, 0.0)

    if edge < min_edge or stake < min_stake or ev <= 0:
        return KellyResult(
            edge=edge,
            kelly_fraction=kelly_full,
            fractional_kelly=kelly_frac,
            stake=0.0,
            expected_value=ev,
            action="PASS",
        )

    return KellyResult(
        edge=edge,
        kelly_fraction=kelly_full,
        fractional_kelly=kelly_frac,
        stake=round(stake, 2),
        expected_value=round(ev, 4),
        action="BET",
    )


def calculate_kelly_portfolio(
    bets: list[dict],
    bankroll: float,
    fraction: float = 0.25,
    min_edge: float = 0.03,
) -> list[KellyResult]:
    results = []
    for bet in bets:
        result = calculate_kelly(
            model_prob=bet["prob"],
            decimal_odds=bet["odds"],
            bankroll=bankroll,
            fraction=fraction,
            min_edge=min_edge,
        )
        results.append(result)
    return results


def estimate_ruin_probability(
    win_rate: float,
    avg_odds: float,
    bankroll: float,
    simulations: int = 10000,
    n_bets: int = 1000,
) -> dict:
    np.random.seed(42)
    final_bankrolls = []

    for _ in range(simulations):
        b = bankroll
        for _ in range(n_bets):
            if b <= 0:
                break
            kelly_frac = (avg_odds - 1) * win_rate - (1 - win_rate)
            kelly_frac = max(kelly_frac * 0.25 / (avg_odds - 1), 0)
            stake = kelly_frac * b
            if stake <= 0:
                break
            if np.random.random() < win_rate:
                b += stake * (avg_odds - 1)
            else:
                b -= stake
        final_bankrolls.append(b)

    final_bankrolls = np.array(final_bankrolls)
    return {
        "ruin_probability": float(np.mean(final_bankrolls <= 0)),
        "median_final": float(np.median(final_bankrolls)),
        "mean_final": float(np.mean(final_bankrolls)),
        "best_case": float(np.percentile(final_bankrolls, 95)),
        "worst_case": float(np.percentile(final_bankrolls, 5)),
    }
