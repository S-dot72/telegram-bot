# auto_verifier.py
"""
AutoResultVerifier
Fichier complet et corrigé pour la vérification automatique des résultats.

Principales améliorations :
- Requête simple en SQL + filtrage/time parsing robuste en Python (évite différences SQLite)
- _get_today_stats() utilise bornes UTC pour être fiable
- Gestion des timestamps tolérante (ISO Z / ISO+offset / 'YYYY-MM-DD HH:MM:SS')
- Logs additionnels pour debugging
- Respect des limites API (sleep)
- Envoi de rapports Telegram aux admins
"""

import asyncio
import time
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import text

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key, bot=None):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.bot = bot
        self.admin_chat_ids = []
        self.utc_tz = timezone.utc
        self.local_tz = ZoneInfo("America/Port-au-Prince")

    def set_bot(self, bot):
        self.bot = bot

    def add_admin(self, chat_id):
        if chat_id not in self.admin_chat_ids:
            self.admin_chat_ids.append(chat_id)
            print(f"✅ Admin {chat_id} ajouté pour recevoir les rapports")

    async def verify_pending_signals(self, limit=50):
        try:
            print("\n" + "="*60)
            print(f"🔍 VÉRIFICATION AUTOMATIQUE - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print("="*60)

            query = text("""
                SELECT id, pair, direction, ts_enter, confidence, 
                       COALESCE(timeframe, 5) as timeframe,
                       COALESCE(gale_level, 0) as gale_level,
                       COALESCE(max_gales, 2) as max_gales
                FROM signals
                WHERE result IS NULL
                ORDER BY ts_enter DESC
                LIMIT :limit
            """)

            with self.engine.connect() as conn:
                rows = conn.execute(query, {'limit': limit}).fetchall()

            print(f"📌 Rows fetched for pending check: {len(rows)}")

            pending = []
            now_utc = datetime.now(timezone.utc)

            for row in rows:
                try:
                    sid = row[0]
                    pair = row[1]
                    direction = row[2]
                    ts_enter_raw = row[3]
                    confidence = row[4]
                    timeframe = int(row[5] or 5)
                    max_gales = int(row[7] if len(row) > 7 and row[7] is not None else 2)
                except Exception as e:
