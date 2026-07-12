#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from src.data.flashscore_scraper import fetch_todays_matches, identify_strong_favorites, build_report
from src.alerts.telegram_bot import TelegramAlert


def main():
    print("=" * 60)
    print("  SPORTS ANALYTICS - PIPELINE DIARIO")
    print("=" * 60)

    today = date.today()
    print(f"\n  Buscando partidos para {today}...")

    df = fetch_todays_matches()
    print(f"  {len(df)} partidos encontrados")

    strong = identify_strong_favorites(df, min_prob=0.60)
    print(f"  {len(strong)} candidatos con favorito claro (>60%)")

    report = build_report(df, strong)
    print(f"\n{report}")

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        print("\n  Enviando a Telegram...")
        bot = TelegramAlert(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        ok = bot.send_sync(report)
        print(f"  {'✅ Enviado' if ok else '❌ Falló envío'}")
    else:
        print("\n  ⚠️ Telegram no configurado.")
        out_path = Path(__file__).parent / "output"
        out_path.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        with open(out_path / f"daily_{ts}.txt", "w") as f:
            f.write(report)
        print(f"  Reporte guardado en output/daily_{ts}.txt")


if __name__ == "__main__":
    main()
