"""
Pentanova Hukuk Danışman Telegram Botu
SGK, İş Hukuku ve KVKK Uzmanı
3 AI Modeli Sentezli - Doğal İnsan Tarzı Cevaplar

Geliştirici: Pentanova Danışmanlık
"""

import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()
import asyncio
import logging
import random
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# AI SDK'ları
import anthropic
import openai
import google.generativeai as genai

# Logging ayarları
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Kullanıcı oturum bilgileri"""
    user_id: int
    first_name: str = ""
    conversation_history: list = field(default_factory=list)
    question_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


# Kullanıcı oturumları
user_sessions: dict[int, UserSession] = {}


# ============================================================
# UZMAN KİŞİLİK VE SİSTEM PROMPTU
# ============================================================

EXPERT_PERSONA = """Sen Pentanova Danışmanlık'ın kıdemli hukuk danışmanısın. 18 yıllık mesleki tecrübeye sahipsin.

## KİMLİĞİN:

**Geçmişin:**
- İstanbul Üniversitesi Hukuk Fakültesi mezunu
- 18 yıldır SGK, İş Hukuku ve KVKK alanında kurumsal danışmanlık
- SGK'da 3 yıl müfettiş olarak görev yaptın
- Yüzlerce kurumsal müşteriye hizmet verdin

**Konuşma Tarzın:**
- Her zaman "siz" diye hitap et, resmi ve saygılı ol
- Profesyonel, güven veren ve net bir üslup kullan
- Bilgileri açık ve anlaşılır şekilde aktar
- Gerektiğinde mevzuat referansı ver
- Kısa, öz ve doğrudan cevaplar ver

**Kesinlikle YAPMA:**
- "Yapay zeka olarak", "bir dil modeli olarak" gibi ifadeler KULLANMA
- Emoji kullanma
- Günlük konuşma dili, argo veya samimi ifadeler kullanma
- "Bak şimdi", "yani", "aslında" gibi dolgu kelimeler kullanma
- Madde madde uzun listeler yapma
- "Başka sorunuz var mı?" gibi kalıp cümleler kullanma

**KESİNLİKLE YAP:**
- Profesyonel ve kurumsal bir dil kullan
- Kısa paragraflar halinde açıkla
- Somut bilgi ve rakam ver
- İlgili kanun maddelerine atıf yap
- Net ve kesin ifadeler kullan
- Pratik çözüm önerileri sun

## UZMANLIK ALANLARIN:

### SGK & SOSYAL GÜVENLİK:
- 5510 sayılı Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu
- Prim hesaplamaları ve bildirimleri
- Emeklilik koşulları ve hesaplamaları
- İş kazası ve meslek hastalıkları
- Teşvik uygulamaları (5510/81, 7252, 7316)

### İŞ HUKUKU (4857):
- Kıdem ve ihbar tazminatı hesaplamaları
- Haklı fesih halleri ve prosedürleri
- İşe iade davaları
- Fazla mesai ve yıllık izin hakları
- İş sözleşmesi türleri

### KVKK (6698):
- Veri işleme şartları ve yükümlülükler
- Aydınlatma metinleri ve açık rıza
- VERBİS kayıt işlemleri
- İdari para cezaları
- Veri ihlali bildirimi

## 2026 YILI GÜNCEL RAKAMLAR:

### Asgari Ücret 2026:
- Brüt Asgari Ücret: 26.005,50 TL
- Net Asgari Ücret: 22.104,67 TL
- Günlük Brüt: 866,85 TL
- Saatlik Brüt: 173,37 TL

### SGK Primleri 2026:
- SGK Taban: 26.005,50 TL
- SGK Tavan: 195.041,25 TL
- İşçi SGK Primi: %14
- İşveren SGK Primi: %20,5 + %2 işsizlik

### Kıdem Tazminatı 2026:
- Kıdem Tazminatı Tavanı: 48.369,22 TL
- Her yıl için 1 brüt maaş

### İhbar Süreleri:
- 0-6 ay: 2 hafta
- 6 ay - 1,5 yıl: 4 hafta
- 1,5 - 3 yıl: 6 hafta
- 3 yıl üzeri: 8 hafta

### Yıllık İzin:
- 1-5 yıl: 14 gün
- 5-15 yıl: 20 gün
- 15+ yıl: 26 gün

### KVKK İdari Para Cezaları 2026:
- Aydınlatma yükümlülüğü ihlali: 75.000 - 3.000.000 TL
- Veri güvenliği ihlali: 150.000 - 6.000.000 TL
- Kurul kararlarına uymama: 225.000 - 6.000.000 TL

## ÖRNEK CEVAP TARZI:

Soru: "Kıdem tazminatı nasıl hesaplanır?"

CEVAP:
"Kıdem tazminatı, her tam çalışma yılı için bir brüt ücret tutarında hesaplanır. 

Hesaplamaya dahil edilecek kalemler: temel ücret, düzenli ödenen yemek ve yol yardımı, prim ve ikramiyeler.

2026 yılı için kıdem tazminatı tavanı 48.369,22 TL'dir. Brüt ücretiniz bu tutarı aşsa dahi, tavan üzerinden hesaplama yapılır.

Kıdem tazminatı almaya hak kazanmak için en az 1 yıl kıdem süresi ve İş Kanunu'nun 14. maddesinde belirtilen fesih şartlarının sağlanması gerekmektedir.

Çalışma süreniz ve brüt ücretinizi belirtirseniz, net hesaplama yapabilirim."
"""

SYNTHESIS_PROMPT = """Aşağıda bir soruya 3 farklı kaynaktan derlediğim teknik bilgiler var:

**Kaynak 1:**
{claude_response}

**Kaynak 2:**
{gpt4_response}

**Kaynak 3:**
{gemini_response}

---

Şimdi bu bilgileri kullanarak aşağıdaki persona ile DOĞAL bir şekilde cevap yaz. Sanki bu bilgileri zaten biliyormuşsun gibi davran, "kaynaklara göre" gibi ifadeler KULLANMA.

{persona}

KULLANICININ SORUSU: {question}

ÖNEMLİ KURALLAR:
- Bilgileri kendi bilgin gibi aktar, kaynak belirtme
- Doğal, samimi ve insani bir dil kullan
- Asla liste/madde yapma, paragraflar halinde anlat
- Kısa tut ama bilgilendirici ol
- Gerekirse soru sor (çalışma süresi, maaş vb.)
- Robot gibi değil, tecrübeli bir danışman gibi konuş
"""


# ============================================================
# SELAMLAMA CÜMLELERİ
# ============================================================

GREETINGS = [
    "Hoş geldiniz {name}. Pentanova Danışmanlık olarak SGK, İş Hukuku ve KVKK konularında size yardımcı olabiliriz. Sorunuzu iletebilirsiniz.",
    "Merhaba {name}. Pentanova Danışmanlık hukuk danışmanlığı hizmetine hoş geldiniz. Size nasıl destek olabiliriz?",
    "Hoş geldiniz {name}. SGK, İş Kanunu ve KVKK mevzuatı konularında danışmanlık hizmeti vermekteyiz. Sorunuzu dinliyoruz.",
]


# ============================================================
# API İSTEMCİLERİ
# ============================================================

class AIClients:
    """AI API istemcileri yöneticisi"""
    
    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        self.claude_client = None
        self.openai_client = None
        self.gemini_model = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """API istemcilerini başlat"""
        if self.anthropic_key:
            self.claude_client = anthropic.Anthropic(api_key=self.anthropic_key)
            logger.info("✅ Claude API hazır")
        
        if self.openai_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
            logger.info("✅ OpenAI API hazır")
        
        if self.google_key:
            genai.configure(api_key=self.google_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("✅ Gemini API hazır")
    
    async def _get_claude_response(self, question: str) -> str:
        """Claude'dan ham bilgi al"""
        if not self.claude_client:
            return ""
        
        try:
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system="""Sen bir Türk iş hukuku, SGK ve KVKK uzmanısın. Sorulan soruya güncel mevzuata göre teknik ve detaylı bilgi ver.

2026 YILI GÜNCEL RAKAMLARINI KULLAN:
- Brüt Asgari Ücret: 26.005,50 TL
- Net Asgari Ücret: 22.104,67 TL
- SGK Tavan: 195.041,25 TL
- Kıdem Tazminatı Tavanı: 48.369,22 TL
- KVKK Cezaları: 75.000 TL - 6.000.000 TL arası

Hesaplamalarda bu 2026 rakamlarını kullan.""",
                messages=[{"role": "user", "content": question}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude hatası: {e}")
            return ""
    
    async def _get_gpt4_response(self, question: str) -> str:
        """GPT-4'ten ham bilgi al"""
        if not self.openai_client:
            return ""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """Sen bir Türk iş hukuku, SGK ve KVKK uzmanısın. Sorulan soruya güncel mevzuata göre teknik ve detaylı bilgi ver.

2026 YILI GÜNCEL RAKAMLARINI KULLAN:
- Brüt Asgari Ücret: 26.005,50 TL
- Net Asgari Ücret: 22.104,67 TL
- SGK Tavan: 195.041,25 TL
- Kıdem Tazminatı Tavanı: 48.369,22 TL
- KVKK Cezaları: 75.000 TL - 6.000.000 TL arası

Hesaplamalarda bu 2026 rakamlarını kullan."""},
                    {"role": "user", "content": question}
                ],
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT-4 hatası: {e}")
            return ""
    
    async def _get_gemini_response(self, question: str) -> str:
        """Gemini'den ham bilgi al"""
        if not self.gemini_model:
            return ""
        
        try:
            prompt = f"""Sen bir Türk iş hukuku, SGK ve KVKK uzmanısın. 

2026 YILI GÜNCEL RAKAMLARINI KULLAN:
- Brüt Asgari Ücret: 26.005,50 TL
- Net Asgari Ücret: 22.104,67 TL
- SGK Tavan: 195.041,25 TL
- Kıdem Tazminatı Tavanı: 48.369,22 TL
- KVKK Cezaları: 75.000 TL - 6.000.000 TL arası

Şu soruya teknik ve detaylı bilgi ver: {question}"""
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini hatası: {e}")
            return ""
    
    async def get_human_response(self, question: str, user_name: str) -> str:
        """3 modelden bilgi al, insan gibi sentezle"""
        
        # Paralel olarak 3 modelden bilgi al
        tasks = [
            self._get_claude_response(question),
            self._get_gpt4_response(question),
            self._get_gemini_response(question)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        claude_resp = responses[0] if isinstance(responses[0], str) else ""
        gpt4_resp = responses[1] if isinstance(responses[1], str) else ""
        gemini_resp = responses[2] if isinstance(responses[2], str) else ""
        
        # En az bir cevap olmalı
        valid_responses = [r for r in [claude_resp, gpt4_resp, gemini_resp] if r]
        
        if not valid_responses:
            return "Teknik bir aksaklık yaşanmaktadır. Lütfen kısa bir süre sonra tekrar deneyiniz."
        
        # Claude ile insan tarzı sentezle
        if self.claude_client:
            try:
                synthesis_input = SYNTHESIS_PROMPT.format(
                    claude_response=claude_resp or "(bilgi yok)",
                    gpt4_response=gpt4_resp or "(bilgi yok)",
                    gemini_response=gemini_resp or "(bilgi yok)",
                    persona=EXPERT_PERSONA,
                    question=question
                )
                
                response = self.claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": synthesis_input}]
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Sentezleme hatası: {e}")
                return max(valid_responses, key=len)
        
        return max(valid_responses, key=len)


# Global AI istemcisi
ai_clients: Optional[AIClients] = None


# ============================================================
# TELEGRAM BOT KOMUTLARI
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Bot başlatma"""
    user = update.effective_user
    user_id = user.id
    first_name = user.first_name or "dostum"
    
    user_sessions[user_id] = UserSession(
        user_id=user_id,
        first_name=first_name
    )
    
    greeting = random.choice(GREETINGS).format(name=first_name)
    await update.message.reply_text(greeting)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Yardım"""
    help_text = """Pentanova Danışmanlık Hukuk Danışmanlığı Hizmeti

Uzmanlaştığımız konular:

SGK ve Sosyal Güvenlik
- Prim hesaplamaları ve bildirimler
- Emeklilik şartları ve hesaplamaları
- Teşvik uygulamaları

İş Hukuku
- Kıdem ve ihbar tazminatı
- İş sözleşmesi ve fesih işlemleri
- Fazla mesai ve izin hakları

KVKK
- Veri koruma yükümlülükleri
- VERBİS kayıt işlemleri
- İdari para cezaları

Sorunuzu doğrudan yazabilirsiniz.

Önemli Not: Verilen bilgiler genel niteliktedir. Kesin hukuki kararlar için avukat desteği almanızı öneririz."""
    
    await update.message.reply_text(help_text)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Geçmişi temizle"""
    user_id = update.effective_user.id
    
    if user_id in user_sessions:
        user_sessions[user_id].conversation_history = []
    
    await update.message.reply_text("Görüşme geçmişi temizlendi. Yeni bir konuya geçebilirsiniz.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kullanıcı mesajlarını işle"""
    global ai_clients
    
    if not ai_clients:
        ai_clients = AIClients()
    
    user = update.effective_user
    user_id = user.id
    user_message = update.message.text
    first_name = user.first_name or "Sayın Kullanıcı"
    chat_type = update.effective_chat.type
    
    # Grup kontrolü - sadece etiketlenince cevap ver
    if chat_type in ["group", "supergroup"]:
        bot_username = context.bot.username
        if f"@{bot_username}" not in user_message:
            return  # Etiketlenmemişse cevap verme
        # Etiketi mesajdan temizle
        user_message = user_message.replace(f"@{bot_username}", "").strip()
    
    # Boş mesaj kontrolü
    if not user_message:
        return
    
    # Oturum kontrolü
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(
            user_id=user_id,
            first_name=first_name
        )
    
    session = user_sessions[user_id]
    session.question_count += 1
    session.last_activity = datetime.now()
    
    # Yazıyor göstergesi
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Cevap al
    response = await ai_clients.get_human_response(user_message, first_name)
    
    # Geçmişe ekle
    session.conversation_history.append({"role": "user", "content": user_message})
    session.conversation_history.append({"role": "assistant", "content": response})
    
    # Son 10 mesajı tut
    if len(session.conversation_history) > 20:
        session.conversation_history = session.conversation_history[-20:]
    
    await update.message.reply_text(response)


# ============================================================
# ANA FONKSİYON
# ============================================================

async def main():
    """Botu başlat"""
    global ai_clients
    
    # Telegram token
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN bulunamadı!")
        print("\n" + "="*50)
        print("HATA: TELEGRAM_BOT_TOKEN ayarlanmamış!")
        print("="*50)
        print("\nÇözüm:")
        print("export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("="*50 + "\n")
        return
    
    # AI istemcilerini başlat
    ai_clients = AIClients()
    
    # Bot uygulamasını oluştur
    app = Application.builder().token(bot_token).build()
    
    # Komut handler'ları
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("yardim", help_command))
    app.add_handler(CommandHandler("temizle", clear_command))
    app.add_handler(CommandHandler("clear", clear_command))
    
    # Mesaj handler'ı
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Botu başlat
    print("\n" + "="*50)
    print("🏢 PENTANOVA HUKUK DANIŞMAN BOT")
    print("="*50)
    print("✅ Bot başlatıldı!")
    print("📱 Telegram'da botunuza mesaj gönderin")
    print("🛑 Durdurmak için Ctrl+C")
    print("="*50 + "\n")
    
    # Initialize and start
    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
    # Run until stopped
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
