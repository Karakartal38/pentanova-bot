"""
Pentanova Hukuk Danışman Telegram Botu
Sadece Claude API - Basit Versiyon
"""

import os
import asyncio
import logging
import random
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import anthropic

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Sen Pentanova Danışmanlık'ın kıdemli hukuk danışmanısın. 18 yıllık SGK, İş Hukuku ve KVKK tecrüben var.

ÖNEMLİ: Bugünün tarihi 5 Ocak 2026. Tüm hesaplamalarda ve tarihlerde 2026 yılını kullan.


KURALLAR:
- Her zaman "siz" diye hitap et, resmi ol
- Kısa ve net cevaplar ver
- Emoji kullanma
- "Yapay zeka olarak" gibi ifadeler KULLANMA
- Mevzuat referansı ver

2026 GÜNCEL RAKAMLAR:
- Brüt Asgari Ücret: 26.005,50 TL
- Net Asgari Ücret: 22.104,67 TL  
- Kıdem Tazminatı Tavanı: 48.369,22 TL
- SGK Tavan: 195.041,25 TL"""

GREETINGS = [
    "Hoş geldiniz {name}. SGK, İş Hukuku ve KVKK konularında size yardımcı olabilirim.",
    "Merhaba {name}. Pentanova Danışmanlık olarak hizmetinizdeyiz.",
]

claude_client = None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "Sayın Kullanıcı"
    await update.message.reply_text(random.choice(GREETINGS).format(name=name))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global claude_client
    
    if not claude_client:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            claude_client = anthropic.Anthropic(api_key=api_key)
    
    user_message = update.message.text
    chat_type = update.effective_chat.type
    
    # Grup kontrolü
    if chat_type in ["group", "supergroup"]:
        bot_username = context.bot.username
        if f"@{bot_username}" not in user_message:
            return
        user_message = user_message.replace(f"@{bot_username}", "").strip()
    
    if not user_message:
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        answer = response.content[0].text
    except Exception as e:
        logger.error(f"Claude hatası: {e}")
        answer = "Teknik bir aksaklık yaşanmaktadır. Lütfen tekrar deneyiniz."
    
    await update.message.reply_text(answer)


async def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN bulunamadı!")
        return
    
    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot başlatıldı!")
    
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
