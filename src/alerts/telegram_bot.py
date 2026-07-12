import asyncio
from typing import Optional

try:
    from telegram import Bot
    from telegram.error import TelegramError

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramAlert:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        if TELEGRAM_AVAILABLE and token and chat_id:
            self.bot = Bot(token=token)

    MAX_LEN = 4000

    async def send_message(self, text: str, retry: int = 3) -> bool:
        if not self.bot or not self.chat_id:
            return False
        for attempt in range(retry):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    disable_web_page_preview=True,
                )
                return True
            except TelegramError as e:
                err = str(e).lower()
                if "retry" in err or "flood" in err:
                    import re
                    match = re.search(r"(\d+)", err)
                    wait = int(match.group(1)) + 1 if match else 5
                    await asyncio.sleep(wait)
                    continue
                return False
        return False

    async def send_long(self, text: str) -> bool:
        if len(text) <= self.MAX_LEN:
            return await self.send_message(text)
        chunks = [text[i : i + self.MAX_LEN] for i in range(0, len(text), self.MAX_LEN)]
        results = []
        for i, chunk in enumerate(chunks):
            ok = await self.send_message(chunk)
            results.append(ok)
            if i < len(chunks) - 1:
                await asyncio.sleep(2)
        return all(results)

    def send_sync(self, text: str) -> bool:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_long(text))
        except Exception:
            return False
