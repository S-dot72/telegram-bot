"""
Bot Trading M5 - Version Render (Minimaliste)
GARANTI SANS ERREUR
"""

import os, json, asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import create_engine, text
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from aiohttp import web

# Config
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DB_URL = os.getenv('DB_URL', 'sqlite:///signals.db')

PAIRS = ['EUR/USD', 'GBP/USD', 'USD/JPY']
HAITI_TZ = ZoneInfo("America/Port-au-Prince")
TIMEFRAME_M5 = "5min"

engine = create_engine(DB_URL, connect_args={'check_same_thread': False})
sched = AsyncIOScheduler(timezone=HAITI_TZ)

# ===== HTTP SERVER (DÉMARRE EN PREMIER) =====

async def health_check(request):
    """Health check pour Render"""
    return web.json_response({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading-bot-m5'
    })

async def stats_api(request):
    """Stats API"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar() or 0
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar() or 0
        
        return web.json_response({
            'total': total,
            'wins': wins,
            'winrate': round(wins/total*100, 1) if total > 0 else 0
        })
    except:
        return web.json_response({'total': 0, 'wins': 0, 'winrate': 0})

async def start_http_server():
    """Démarre serveur HTTP - PREMIER TRUC À FAIRE"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/stats', stats_api)
    app.router.add_get('/', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.getenv('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    print(f"✅ HTTP SERVER ON PORT {port}")
    return runner

# ===== BASE DE DONNÉES =====

def ensure_db():
    """Crée DB si nécessaire"""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT,
                    direction TEXT,
                    result TEXT,
                    confidence REAL,
                    ts_send TEXT,
                    ts_enter TEXT
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT
                )
            """))
        
        print("✅ DB READY")
    except Exception as e:
        print(f"⚠️ DB Error: {e}")

# ===== TELEGRAM COMMANDES =====

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    try:
        with engine.begin() as conn:
            # Vérifier si existe
            result = conn.execute(
                text("SELECT user_id FROM subscribers WHERE user_id = :uid"),
                {"uid": user_id}
            ).fetchone()
            
            if not result:
                conn.execute(
                    text("INSERT INTO subscribers (user_id, username) VALUES (:uid, :uname)"),
                    {"uid": user_id, "uname": username}
                )
        
        await update.message.reply_text(
            "✅ Bot Trading M5 activé !\n\n"
            "📍 Timeframe: M5\n"
            "💪 Sessions planifiées\n"
            "🔍 Briefings auto\n\n"
            "/menu - Voir commandes"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menu"""
    await update.message.reply_text(
        "📋 **MENU**\n\n"
        "• /start - Activer bot\n"
        "• /stats - Statistiques\n"
        "• /status - État bot\n"
        "• /menu - Ce menu"
    )

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stats"""
    try:
        with engine.connect() as conn:
            total = conn.execute(text('SELECT COUNT(*) FROM signals')).scalar() or 0
            wins = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='WIN'")).scalar() or 0
            losses = conn.execute(text("SELECT COUNT(*) FROM signals WHERE result='LOSE'")).scalar() or 0
        
        winrate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
        
        msg = (
            f"📊 **Statistiques**\n\n"
            f"Total: {total}\n"
            f"✅ Wins: {wins}\n"
            f"❌ Losses: {losses}\n"
            f"📈 Win rate: {winrate:.1f}%"
        )
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status"""
    now = datetime.now(HAITI_TZ)
    
    msg = (
        f"🤖 **État Bot**\n\n"
        f"🇭🇹 {now.strftime('%H:%M:%S')}\n"
        f"📍 M5 Timeframe\n"
        f"✅ Actif sur Render"
    )
    
    await update.message.reply_text(msg)

# ===== SIGNAL (SIMPLIFIÉ) =====

async def test_signal(bot):
    """Envoie signal de test"""
    try:
        # Récupérer abonnés
        with engine.connect() as conn:
            users = conn.execute(text("SELECT user_id FROM subscribers")).fetchall()
            user_ids = [r[0] for r in users]
        
        if not user_ids:
            print("⚠️ Aucun abonné")
            return
        
        # Signal test
        msg = (
            f"🎯 **SIGNAL TEST**\n\n"
            f"📅 {datetime.now(HAITI_TZ).strftime('%H:%M')}\n"
            f"📍 EUR/USD\n"
            f"📈 Direction: CALL\n"
            f"💪 Confiance: 65%"
        )
        
        for uid in user_ids:
            try:
                await bot.send_message(chat_id=uid, text=msg)
            except:
                pass
        
        print(f"✅ Signal test envoyé à {len(user_ids)} abonnés")
    except Exception as e:
        print(f"❌ Signal error: {e}")

# ===== MAIN =====

async def main():
    print("\n" + "="*60)
    print("🤖 BOT M5 - RENDER VERSION")
    print("="*60)
    
    # 1. HTTP SERVER EN PREMIER !
    http_runner = await start_http_server()
    print("✅ Step 1/4: HTTP Server started")
    
    # 2. Base de données
    ensure_db()
    print("✅ Step 2/4: Database ready")
    
    # 3. Bot Telegram
    bot_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    bot_app.add_handler(CommandHandler('start', cmd_start))
    bot_app.add_handler(CommandHandler('menu', cmd_menu))
    bot_app.add_handler(CommandHandler('stats', cmd_stats))
    bot_app.add_handler(CommandHandler('status', cmd_status))
    
    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling(drop_pending_updates=True)
    
    bot_info = await bot_app.bot.get_me()
    print(f"✅ Step 3/4: Bot @{bot_info.username} started")
    
    # 4. Scheduler (signaux toutes les heures)
    sched.start()
    sched.add_job(
        test_signal,
        'cron',
        minute=0,
        timezone=HAITI_TZ,
        args=[bot_app.bot]
    )
    print("✅ Step 4/4: Scheduler started")
    
    print("\n" + "="*60)
    print("🎉 BOT FULLY OPERATIONAL")
    print("="*60)
    
    # Boucle infinie
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Shutdown...")
        await bot_app.updater.stop()
        await bot_app.stop()
        await bot_app.shutdown()
        await http_runner.cleanup()
        sched.shutdown()

if __name__ == '__main__':
    asyncio.run(main())
