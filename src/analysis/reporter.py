import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from typing import Optional
from src.strategies.kelly import calculate_kelly


def build_match_report(
    home_team: str,
    away_team: str,
    league: str,
    date: str,
    model_probs: dict,
    odds: Optional[dict] = None,
    bankroll: float = 1000.0,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"{home_team} vs {away_team}")
    lines.append(f"{league} | {date}")
    lines.append("=" * 70)

    lines.append("\n[1] PREDICTED SCORE & EXPECTED GOALS")
    lines.append("-" * 40)
    lines.append(f" Expected Goals (xG): {model_probs.get('expected_goals_home', 0):.2f} - {model_probs.get('expected_goals_away', 0):.2f}")
    lines.append(f" Most Likely Score:   {model_probs.get('most_likely_score', 'N/A')}")

    lines.append("\n[2] MATCH OUTCOME PROBABILITIES")
    lines.append("-" * 40)
    header = ["Market", "Model Prob", "Fair Odds"]
    if odds:
        header += ["Book Odds", "Edge", "EV", "Action"]
    table = []

    outcomes = [
        ("1 (Home)", model_probs.get("home_win", 0)),
        ("X (Draw)", model_probs.get("draw", 0)),
        ("2 (Away)", model_probs.get("away_win", 0)),
    ]

    for name, prob in outcomes:
        if odds and name[0] in ["1", "X", "2"]:
            key = {"1": "odds_home", "X": "odds_draw", "2": "odds_away"}[name[0]]
            od = odds.get(key, 0)
            if od and prob > 0:
                kelly = calculate_kelly(prob, od, bankroll)
                edge = kelly.edge
                ev = kelly.expected_value
                fair_odds = round(1 / prob, 2) if prob > 0 else 0
                action = kelly.action
                table.append([
                    name, f"{prob:.1%}", fair_odds,
                    f"{od:.2f}", f"{edge:.1%}", f"{ev:.3f}", action,
                ])
            else:
                table.append([name, f"{prob:.1%}", round(1/prob, 2) if prob > 0 else 0, "-", "-", "-", "-"])
        else:
            table.append([name, f"{prob:.1%}", round(1/prob, 2) if prob > 0 else 0])

    lines.append(tabulate(table, headers=header, tablefmt="simple"))

    lines.append("\n[3] SECONDARY MARKETS")
    lines.append("-" * 40)
    secondary = [
        ("Over 2.5 Goals", model_probs.get("over_2_5", 0)),
        ("Both Teams to Score", model_probs.get("btts", 0)),
    ]
    for name, prob in secondary:
        lines.append(f"  {name:<25} {prob:.1%}")

    lines.append("\n[4] MOST LIKELY SCORES")
    lines.append("-" * 40)
    score_probs = model_probs.get("score_probs", {})
    for score, prob in list(score_probs.items())[:5]:
        lines.append(f"  {score:<8} {prob:.1%}")

    if odds:
        kelly = calculate_kelly(
            model_probs.get("home_win", 0),
            odds.get("odds_home", 2.0),
            bankroll,
        )
        lines.append(f"\n[5] STAKING RECOMMENDATION ({kelly.action})")
        lines.append("-" * 40)
        if kelly.action == "BET":
            lines.append(f"  Stake:        ${kelly.stake:.2f}")
            lines.append(f"  Kelly:        {kelly.kelly_fraction:.2%}")
            lines.append(f"  Fractional:   {kelly.fractional_kelly:.2%} (25% Kelly)")
            lines.append(f"  Expected ROI: {kelly.expected_value:.2%}")
            lines.append(f"  Edge:         {kelly.edge:.2%}")
        else:
            lines.append("  No value bet found (edge < 3% threshold)")

    lines.append("=" * 70)
    return "\n".join(lines)


def build_daily_report(
    predictions: list[dict],
    bankroll: float = 1000.0,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"DAILY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    value_bets = [p for p in predictions if p.get("action") == "BET"]

    if value_bets:
        lines.append(f"\n🎯 VALUE BETS FOUND: {len(value_bets)}")
        lines.append("-" * 70)
        table = []
        for bet in value_bets:
            table.append([
                bet["home_team"][:15],
                bet["away_team"][:15],
                bet["predicted"],
                f"{bet['prob']:.1%}",
                f"{bet['odds']:.2f}",
                f"{bet['edge']:.1%}",
                f"${bet['stake']:.2f}",
                f"{bet['ev']:.3f}",
            ])
        lines.append(
            tabulate(
                table,
                headers=["Home", "Away", "Pick", "Prob", "Odds", "Edge", "Stake", "EV"],
                tablefmt="simple",
            )
        )
    else:
        lines.append("\n No value bets found for today (edge < 3%)")

    lines.append("\nBANKROLL STATUS")
    lines.append("-" * 40)
    total_stake = sum(p.get("stake", 0) for p in value_bets)
    lines.append(f"  Current Bankroll:  ${bankroll:.2f}")
    lines.append(f"  Total at Risk:     ${total_stake:.2f} ({total_stake/bankroll:.1%} of bankroll)")
    lines.append(f"  Exposed to:        {len(value_bets)} bet(s)")

    lines.append("=" * 70)
    lines.append("⚠️  Past performance does not guarantee future results.")
    lines.append("=" * 70)
    return "\n".join(lines)


def build_performance_report(backtest_result) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("STRATEGY PERFORMANCE REPORT")
    lines.append("=" * 70)
    lines.append(f"\n{backtest_result.summary()}")

    if backtest_result.trades:
        df_trades = pd.DataFrame(backtest_result.trades)
        lines.append("\n\nRECENT TRADES (Last 10)")
        lines.append("-" * 70)
        recent = df_trades.tail(10)
        table = []
        for _, t in recent.iterrows():
            table.append([
                str(t["date"])[:10],
                t["home"][:12],
                t["away"][:12],
                t["bet"],
                f"{t['odds']:.2f}",
                f"{t['prob']:.1%}",
                f"{t['edge']:.1%}",
                f"${t['stake']:.1f}",
                f"${t['profit']:+.1f}" if t['profit'] != 0 else "$0.0",
            ])
        lines.append(
            tabulate(
                table,
                headers=["Date", "Home", "Away", "Pick", "Odds", "Prob", "Edge", "Stake", "PnL"],
                tablefmt="simple",
            )
        )

    if len(backtest_result.equity_curve) > 1:
        eq = np.array(backtest_result.equity_curve)
        lines.append(f"\nEQUITY CURVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Start:  ${eq[0]:.2f}")
        lines.append(f"  End:    ${eq[-1]:.2f}")
        lines.append(f"  High:   ${eq.max():.2f}")
        lines.append(f"  Low:    ${eq.min():.2f}")

    return "\n".join(lines)
