import requests
import re
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup, NavigableString
from typing import Optional

BASE_URL = "https://www.flashscore.mobi"
ODDS_URL = f"{BASE_URL}/?d=0&s=5"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def to_12h(time_str: str) -> str:
    match = re.match(r"(\d+):(\d+)", time_str)
    if not match:
        return time_str
    h, m = int(match.group(1)), match.group(2)
    period = "AM" if h < 12 else "PM"
    h12 = h if 1 <= h <= 12 else (h - 12 if h > 12 else 12)
    if h == 0:
        h12 = 12
    return f"{h12}:{m} {period}"


def parse_odds_value(text: str) -> float:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_match_result(text: str) -> tuple[int, int, str]:
    text = text.strip()
    if text == "-":
        return 0, 0, "upcoming"
    if ":" in text or text == "":
        return 0, 0, "scheduled"
    match = re.match(r"(\d+)-(\d+)", text)
    if match:
        return int(match.group(1)), int(match.group(2)), "finished"
    return 0, 0, "other"


def parse_odds_bracket(text: str) -> tuple[float, float, float]:
    nums = re.findall(r"(\d+\.\d+)", text)
    if len(nums) >= 3:
        return float(nums[0]), float(nums[1]), float(nums[2])
    return 0.0, 0.0, 0.0


def implied_probability(h_odds: float, d_odds: float, a_odds: float) -> tuple[float, float, float, float]:
    if h_odds <= 0 or d_odds <= 0 or a_odds <= 0:
        return 0, 0, 0, 0
    margin = 1 / h_odds + 1 / d_odds + 1 / a_odds
    h_prob = (1 / h_odds) / margin if margin > 0 else 0
    d_prob = (1 / d_odds) / margin if margin > 0 else 0
    a_prob = (1 / a_odds) / margin if margin > 0 else 0
    return h_prob, d_prob, a_prob, margin


def fetch_todays_matches() -> pd.DataFrame:
    resp = requests.get(ODDS_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    rows = []
    current_league = ""

    for h4 in soup.find_all("h4"):
        league_text = h4.get_text(strip=True)
        for suffix in ["Standings", " Play Offs"]:
            if league_text.endswith(suffix):
                league_text = league_text[: -len(suffix)]
        league_text = league_text.strip().rstrip("-").strip()
        current_league = league_text

        siblings = []
        for sib in h4.next_siblings:
            if hasattr(sib, "name") and sib.name == "h4":
                break
            siblings.append(sib)

        matches = _parse_league_matches(current_league, siblings)
        rows.extend(matches)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = date.today()
    df = df.dropna(subset=["home_team", "away_team"])
    return df.reset_index(drop=True)


def _parse_league_matches(league: str, siblings: list) -> list[dict]:
    matches = []
    i = 0
    while i < len(siblings):
        el = siblings[i]

        if el.name == "span" and "mobi-odds" not in el.get("class", []):
            time_text = el.get_text(strip=True)
            score_text = ""
            odds_text = ""
            team_name_text = ""
            next_idx = i + 1

            # Check if next sibling is a text node (team names) or score <a>
            if next_idx < len(siblings) and isinstance(siblings[next_idx], NavigableString):
                team_name_text = siblings[next_idx].strip()
                next_idx += 1

            # Next should be score <a>
            if next_idx < len(siblings) and siblings[next_idx].name == "a":
                score_text = siblings[next_idx].get_text(strip=True)
                next_idx += 1

            # Then odds <span class="mobi-odds">
            while next_idx < len(siblings):
                s = siblings[next_idx]
                if s.name == "span" and "mobi-odds" in s.get("class", []):
                    odds_text = s.get_text(strip=True)
                    next_idx += 1
                    break
                next_idx += 1

            if not odds_text and next_idx < len(siblings):
                s = siblings[next_idx]
                if s.name == "span" and "mobi-odds" in s.get("class", []):
                    odds_text = s.get_text(strip=True)

            home_team = ""
            away_team = ""
            if team_name_text:
                parts = team_name_text.split(" - ", 1)
                if len(parts) == 2:
                    home_team = parts[0].strip()
                    away_team = parts[1].strip()

            if home_team and away_team and odds_text:
                hg, ag, status = parse_match_result(score_text)
                home_odds, draw_odds, away_odds = parse_odds_bracket(odds_text)
                h_prob, d_prob, a_prob, margin = implied_probability(home_odds, draw_odds, away_odds)

                matches.append({
                    "league": league,
                    "time": to_12h(time_text),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": hg,
                    "away_goals": ag,
                    "status": status,
                    "odds_home": home_odds,
                    "odds_draw": draw_odds,
                    "odds_away": away_odds,
                    "prob_home": round(h_prob, 4),
                    "prob_draw": round(d_prob, 4),
                    "prob_away": round(a_prob, 4),
                    "margin": round(margin, 4),
                })
                i = next_idx
                continue

        i += 1

    return matches


def identify_strong_favorites(
    df: pd.DataFrame,
    min_prob: float = 0.60,
    max_odds_home: float = 2.0,
) -> pd.DataFrame:
    if df.empty:
        return df
    candidates = df[df["status"] == "upcoming"].copy()
    candidates["favorite_prob"] = candidates[["prob_home", "prob_away"]].max(axis=1)
    candidates["favorite_team"] = candidates.apply(
        lambda r: r["home_team"] if r["prob_home"] >= r["prob_away"] else r["away_team"],
        axis=1,
    )
    candidates["is_home_fav"] = candidates["prob_home"] >= candidates["prob_away"]
    candidates = candidates[candidates["favorite_prob"] >= min_prob]
    return candidates.sort_values("favorite_prob", ascending=False).reset_index(drop=True)


def build_report(df: pd.DataFrame, strong: pd.DataFrame) -> str:
    today_str = date.today().strftime("%d/%m/%Y")
    lines = [f"📊 PREDICCIONES {today_str}"]
    lines.append(f"📋 {len(df)} partidos analizados\n")

    if strong.empty:
        lines.append("No hay candidatos para hoy.")
        return "\n".join(lines)

    lines.append("🎯 TOP CANDIDATOS (favorito >60%)")
    lines.append("→ Míralos en vivo. Min ~80: si está 0-0, 1-0,")
    lines.append("  0-1 o 1-1, apuesta UNDER 2.5 o 3.5\n")

    for _, m in strong.head(10).iterrows():
        h = m["home_team"]
        a = m["away_team"]
        fp = m["favorite_prob"]
        if m["is_home_fav"]:
            fav_odds = m["odds_home"]
            cuota = f"Local @ {fav_odds:.2f}"
        else:
            fav_odds = m["odds_away"]
            cuota = f"Visita @ {fav_odds:.2f}"
        lines.append(f"⚽ {h} vs {a}")
        lines.append(f"   🎯 Gana {cuota} ({fp:.0%} probable)")
        lines.append(f"   🕐 {m['time']} | {m['league']}")
        lines.append("")

    rest = len(strong) - 10
    if rest > 0:
        lines.append(f"... y {rest} partidos más con favorito claro\n")

    lines.append("⚠️ Apuesta con responsabilidad.")
    return "\n".join(lines)
