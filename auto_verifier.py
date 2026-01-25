import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json
import random
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
import re

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self._session = requests.Session()
        
        # Pour OTC (crypto)
        self.crypto_endpoints = {
            'binance': 'https://api.binance.com/api/v3/klines',
            'bybit': 'https://api.bybit.com/v5/market/kline',
            'kucoin': 'https://api.kucoin.com/api/v1/market/candles'
        }
        
        print(f"[VERIF] ✅ AutoResultVerifier initialisé - Mode réel activé")
        print(f"[VERIF] 🎯 CORRECTION: Bougie M1 = HH:MM:00 à HH:MM:59 (59 secondes)")

    def _map_pair_to_symbol(self, pair: str, exchange: str = 'binance') -> str:
        """Convertit une paire format TradingView en symbole d'API"""
        mapping = {
            'binance': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT',
                'EUR/USD': 'EURUSDT',
                'GBP/USD': 'GBPUSDT',
                'USD/JPY': 'JPYUSDT',
                'AUD/USD': 'AUDUSDT'
            },
            'bybit': {
                'BTC/USD': 'BTCUSDT',
                'ETH/USD': 'ETHUSDT',
                'TRX/USD': 'TRXUSDT',
                'LTC/USD': 'LTCUSDT'
            },
            'kucoin': {
                'BTC/USD': 'BTC-USDT',
                'ETH/USD': 'ETH-USDT',
                'TRX/USD': 'TRX-USDT',
                'LTC/USD': 'LTC-USDT'
            }
        }
        return mapping.get(exchange, {}).get(pair, pair.replace('/', ''))

    def _get_candle_at_time(self, pair: str, candle_start: datetime, is_otc: bool = False) -> Tuple[float, float, float, float]:
        """
        Récupère la bougie M1 qui COMMENCE à candle_start
        IMPORTANT: Une bougie M1 va de HH:MM:00 à HH:MM:59 (59 secondes)
        """
        try:
            # CORRECTION: Une bougie M1 dure 59 secondes et 999ms, pas 60 secondes
            print(f"[VERIF_CANDLE] 🔍 Recherche bougie M1 pour {pair}")
            print(f"[VERIF_CANDLE] 🕐 Début bougie: {candle_start}")
            print(f"[VERIF_CANDLE] 🕐 Fin bougie: {candle_start.replace(second=59, microsecond=999999)}")
            
            if is_otc:
                return self._get_crypto_candle(pair, candle_start)
            else:
                return self._get_forex_candle(pair, candle_start)
                
        except Exception as e:
            print(f"[VERIF_CANDLE] ⚠️ Erreur récupération bougie: {e}")
            return None, None, None, None

    def _get_crypto_candle(self, pair: str, candle_start: datetime) -> Tuple[float, float, float, float]:
        """Récupère une bougie crypto spécifique (début à candle_start)"""
        try:
            symbol = self._map_pair_to_symbol(pair, 'bybit')
            
            # Timestamp en millisecondes pour le début de la bougie
            start_ms = int(candle_start.timestamp() * 1000)
            end_ms = start_ms + 59000  # +59 secondes (pas 60!)
            
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'spot',
                'symbol': symbol,
                'interval': '1',
                'start': start_ms,
                'end': end_ms,
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('retCode') == 0 and data.get('result'):
                    klines = data['result']['list']
                    
                    if klines:
                        candle = klines[0]
                        candle_actual_start = int(candle[0])
                        
                        # Vérifier que c'est la bonne bougie
                        if abs(candle_actual_start - start_ms) <= 1000:  # Tolérance 1 seconde
                            open_price = float(candle[1])
                            high_price = float(candle[2])
                            low_price = float(candle[3])
                            close_price = float(candle[4])
                            
                            print(f"[VERIF_CRYPTO] ✅ Bougie trouvée: {candle_start}")
                            print(f"[VERIF_CRYPTO] 📊 O={open_price:.5f}, C={close_price:.5f}")
                            
                            return open_price, high_price, low_price, close_price
                        else:
                            print(f"[VERIF_CRYPTO] ⚠️ Décalage bougie: {candle_actual_start} vs {start_ms}")
            
            print(f"[VERIF_CRYPTO] ⚠️ Pas de bougie pour {pair} à {candle_start}")
            return None, None, None, None
            
        except Exception as e:
            print(f"[VERIF_CRYPTO] ❌ Erreur: {e}")
            return None, None, None, None

    def _get_forex_candle(self, pair: str, candle_start: datetime) -> Tuple[float, float, float, float]:
        """Récupère une bougie forex spécifique (début à candle_start)"""
        try:
            # Formater pour TwelveData - ils utilisent des minutes pleines
            start_date = candle_start.strftime('%Y-%m-%d %H:%M:%S')
            # Pour Forex, on prend simplement la minute suivante
            end_date = (candle_start + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'start_date': start_date,
                'end_date': end_date,
                'apikey': self.api_key,
                'outputsize': 1,
                'format': 'JSON'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    candle = data['values'][0]
                    
                    open_price = float(candle['open'])
                    high_price = float(candle['high'])
                    low_price = float(candle['low'])
                    close_price = float(candle['close'])
                    
                    print(f"[VERIF_FOREX] ✅ Bougie trouvée: {candle_start}")
                    print(f"[VERIF_FOREX] 📊 O={open_price:.5f}, C={close_price:.5f}")
                    
                    return open_price, high_price, low_price, close_price
            
            print(f"[VERIF_FOREX] ⚠️ Pas de bougie pour {pair} à {candle_start}")
            return None, None, None, None
            
        except Exception as e:
            print(f"[VERIF_FOREX] ❌ Erreur: {e}")
            return None, None, None, None

    async def verify_single_signal(self, signal_id):
        """
        VÉRIFICATION CORRECTE pour trading binaire M1 - VERSION FINALE
        Règle: Signal à HH:MM → Trade la bougie HH:MM (même bougie)
        """
        try:
            print(f"\n[VERIF] 🔍 Vérification signal #{signal_id}")
            print(f"[VERIF] 🎯 Règle: Signal à HH:MM → Trade même bougie HH:MM")
            
            # Récupérer le signal
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, confidence, payload_json
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[VERIF] ❌ Signal #{signal_id} non trouvé")
                return None
            
            signal_id, pair, direction, ts_enter, confidence, payload_json = signal
            
            # Vérifier si déjà vérifié
            with self.engine.connect() as conn:
                already_verified = conn.execute(
                    text("SELECT result FROM signals WHERE id = :sid AND result IS NOT NULL"),
                    {"sid": signal_id}
                ).fetchone()
            
            if already_verified:
                result = already_verified[0]
                print(f"[VERIF] ✅ Signal #{signal_id} déjà vérifié: {result}")
                return result
            
            print(f"[VERIF] 📊 Signal #{signal_id} - {pair} {direction}")
            print(f"[VERIF] 🕐 Heure signal: {ts_enter}")
            print(f"[VERIF] 💪 Confiance: {confidence:.1%}")
            
            # Analyser le payload
            is_otc = False
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC')
                except:
                    pass
            
            # Convertir ts_enter en datetime
            if isinstance(ts_enter, str):
                ts_enter = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
            
            # LOGIQUE CORRECTE FINALE:
            # 1. Signal à HH:MM:XX → Normaliser à HH:MM:00 (début de la bougie)
            signal_time_normalized = ts_enter.replace(second=0, microsecond=0)
            
            # 2. La bougie tradée est celle qui COMMENCE à cette heure
            #    Bougie: HH:MM:00 → HH:MM:59
            candle_start = signal_time_normalized
            candle_end = signal_time_normalized.replace(second=59, microsecond=999999)
            
            print(f"\n[VERIF] 🔧 LOGIQUE M1 CORRECTE:")
            print(f"[VERIF] 🕐 Signal reçu: {ts_enter}")
            print(f"[VERIF] 🕐 Signal normalisé: {signal_time_normalized}")
            print(f"[VERIF] 🕐 Bougie tradée: {candle_start} → {candle_end}")
            print(f"[VERIF] 📊 Règle: Même bougie, durée 59 secondes")
            
            # 3. Récupérer la bougie (même bougie)
            open_price, high_price, low_price, close_price = self._get_candle_at_time(
                pair, candle_start, is_otc
            )
            
            print(f"\n[VERIF] 📈 PRIX DE LA BOUGIE TRADÉE:")
            print(f"[VERIF] 💰 Open (entrée à {candle_start}): {open_price}")
            print(f"[VERIF] 💰 Close (sortie à {candle_end}): {close_price}")
            
            # Vérifier les données
            if open_price is None or close_price is None:
                print(f"[VERIF] ❌ Données manquantes - INVALID")
                reason = f"Bougie M1 manquante pour {pair} à {candle_start}"
                details = {
                    'reason': reason,
                    'entry_price': None,
                    'exit_price': None,
                    'pips': 0.0,
                    'gale_level': 0
                }
                self._update_signal_result(signal_id, 'INVALID', details)
                return 'INVALID'
            
            # 4. Déterminer le résultat
            if direction == "CALL":
                if close_price > open_price:
                    result = "WIN"
                    reason_detail = f"Bougie HAUSSIÈRE: Close ({close_price:.6f}) > Open ({open_price:.6f})"
                elif close_price < open_price:
                    result = "LOSE"
                    reason_detail = f"Bougie BAISSIÈRE: Close ({close_price:.6f}) < Open ({open_price:.6f})"
                else:
                    result = "DRAW"
                    reason_detail = f"Bougie PLATE: Close ({close_price:.6f}) = Open ({open_price:.6f})"
            else:  # PUT
                if close_price < open_price:
                    result = "WIN"
                    reason_detail = f"Bougie BAISSIÈRE: Close ({close_price:.6f}) < Open ({open_price:.6f})"
                elif close_price > open_price:
                    result = "LOSE"
                    reason_detail = f"Bougie HAUSSIÈRE: Close ({close_price:.6f}) > Open ({open_price:.6f})"
                else:
                    result = "DRAW"
                    reason_detail = f"Bougie PLATE: Close ({close_price:.6f}) = Open ({open_price:.6f})"
            
            # 5. Calculer les métriques
            price_change = close_price - open_price
            price_change_pct = (price_change / open_price * 100) if open_price != 0 else 0
            
            if is_otc:
                pips = abs(price_change)
                diff_text = f"${price_change:+.6f}"
            else:
                pips = abs(price_change) * 10000
                diff_text = f"{price_change:+.5f} ({pips:.1f} pips)"
            
            # 6. Construire la raison
            reason = (
                f"M1: {pair} {direction} | "
                f"Bougie: {candle_start.strftime('%H:%M:%S')}-{candle_end.strftime('%H:%M:%S')} | "
                f"Open: {open_price:.6f} | Close: {close_price:.6f} | "
                f"Change: {diff_text} ({price_change_pct:+.3f}%) | "
                f"{reason_detail} | Result: {result}"
            )
            
            details = {
                'reason': reason,
                'entry_price': float(open_price),
                'exit_price': float(close_price),
                'pips': float(pips),
                'gale_level': 0
            }
            
            self._update_signal_result(signal_id, result, details)
            
            print(f"\n[VERIF] 🎯 RÉSULTAT: {result}")
            print(f"[VERIF] 📊 Bougie: {candle_start.strftime('%H:%M:%S')}-{candle_end.strftime('%H:%M:%S')}")
            print(f"[VERIF] 💰 Open: {open_price:.6f}")
            print(f"[VERIF] 💰 Close: {close_price:.6f}")
            print(f"[VERIF] 📈 Changement: {price_change:.6f} ({price_change_pct:+.3f}%)")
            
            if result == "WIN":
                print(f"[VERIF] 🎉 FÉLICITATIONS ! Trade gagnant !")
            elif result == "LOSE":
                print(f"[VERIF] 📉 Trade perdant - prochaine fois !")
            
            return result
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur verify_single_signal: {e}")
            import traceback
            traceback.print_exc()
            
            details = {
                'reason': f'Erreur système: {str(e)[:100]}',
                'entry_price': None,
                'exit_price': None,
                'pips': 0.0,
                'gale_level': 0
            }
            self._update_signal_result(signal_id, 'ERROR', details)
            return 'ERROR'

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour résultat dans DB"""
        try:
            reason = details.get('reason', '')
            entry_price = details.get('entry_price')
            exit_price = details.get('exit_price')
            pips = details.get('pips')
            
            print(f"[VERIF] 💾 Sauvegarde résultat #{signal_id}: {result}")
            
            with self.engine.begin() as conn:
                # Vérifier les colonnes disponibles
                table_info = conn.execute(
                    text("PRAGMA table_info(signals)")
                ).fetchall()
                
                columns = [row[1] for row in table_info]
                
                if all(col in columns for col in ['entry_price', 'exit_price', 'pips', 'ts_exit']):
                    query = text("""
                        UPDATE signals
                        SET result = :result, 
                            reason = :reason,
                            entry_price = :entry_price,
                            exit_price = :exit_price,
                            pips = :pips,
                            ts_exit = :ts_exit
                        WHERE id = :id
                    """)
                    
                    conn.execute(query, {
                        'result': result,
                        'reason': reason,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pips': pips,
                        'ts_exit': datetime.now(timezone.utc).isoformat(),
                        'id': signal_id
                    })
                else:
                    query = text("""
                        UPDATE signals
                        SET result = :result, 
                            reason = :reason
                        WHERE id = :id
                    """)
                    
                    conn.execute(query, {
                        'result': result,
                        'reason': reason,
                        'id': signal_id
                    })
            
            print(f"[VERIF] ✅ Résultat sauvegardé pour signal #{signal_id}")
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur _update_signal_result: {e}")
            import traceback
            traceback.print_exc()

    def analyze_win_loss_discrepancy(self, signal_id: int):
        """
        Analyse spécifique pour comprendre pourquoi un WIN réel est marqué LOSE
        """
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, result, reason, 
                               entry_price, exit_price, payload_json
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[ANALYSIS] ❌ Signal #{signal_id} non trouvé")
                return
            
            (sig_id, pair, direction, ts_enter, db_result, db_reason, 
             db_entry_price, db_exit_price, payload_json) = signal
            
            # Convertir ts_enter
            if isinstance(ts_enter, str):
                ts_enter_dt = datetime.fromisoformat(ts_enter.replace('Z', '+00:00'))
            else:
                ts_enter_dt = ts_enter
            
            # Analyser le payload
            is_otc = False
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC')
                except:
                    pass
            
            print(f"\n{'='*80}")
            print(f"[ANALYSIS] 🔍 ANALYSE DISCREPANCE WIN/LOSE - Signal #{signal_id}")
            print(f"{'='*80}")
            print(f"[ANALYSIS] 📊 Pair: {pair}")
            print(f"[ANALYSIS] 📊 Direction: {direction}")
            print(f"[ANALYSIS] 🕐 Heure signal DB: {ts_enter_dt}")
            print(f"[ANALYSIS] 📊 Résultat DB: {db_result}")
            print(f"[ANALYSIS] 💰 Prix DB - Entry: {db_entry_price}, Exit: {db_exit_price}")
            
            if db_entry_price and db_exit_price:
                db_change = db_exit_price - db_entry_price
                print(f"[ANALYSIS] 📈 Changement DB: {db_change:.6f}")
            
            # Normaliser l'heure du signal
            signal_normalized = ts_enter_dt.replace(second=0, microsecond=0)
            
            # Essayer TROIS hypothèses différentes:
            print(f"\n[ANALYSIS] 🔧 TEST DES 3 HYPOTHÈSES:")
            
            # Hypothèse 1: Même bougie (HH:MM:00 → HH:MM:59)
            candle1_start = signal_normalized
            candle1_end = signal_normalized.replace(second=59, microsecond=999999)
            
            # Hypothèse 2: Bougie suivante (HH:MM+1:00 → HH:MM+1:59)
            candle2_start = signal_normalized + timedelta(minutes=1)
            candle2_end = candle2_start.replace(second=59, microsecond=999999)
            
            # Hypothèse 3: Ancienne logique buggée (HH:MM:00 → HH:MM+1:00)
            candle3_start = signal_normalized
            candle3_end = signal_normalized + timedelta(minutes=1)
            
            hypotheses = [
                ("Même bougie", candle1_start, candle1_end),
                ("Bougie suivante", candle2_start, candle2_end),
                ("Ancienne bug (2min)", candle3_start, candle3_end)
            ]
            
            for i, (name, start, end) in enumerate(hypotheses, 1):
                print(f"\n[ANALYSIS] 🧪 Hypothèse {i}: {name}")
                print(f"[ANALYSIS]    Début: {start}")
                print(f"[ANALYSIS]    Fin: {end}")
                
                # Récupérer la bougie
                open_price, _, _, close_price = self._get_candle_at_time(pair, start, is_otc)
                
                if open_price and close_price:
                    change = close_price - open_price
                    
                    if direction == "CALL":
                        result = "WIN" if change > 0 else "LOSE" if change < 0 else "DRAW"
                    else:
                        result = "WIN" if change < 0 else "LOSE" if change > 0 else "DRAW"
                    
                    print(f"[ANALYSIS]    Open: {open_price:.6f}")
                    print(f"[ANALYSIS]    Close: {close_price:.6f}")
                    print(f"[ANALYSIS]    Changement: {change:.6f}")
                    print(f"[ANALYSIS]    Résultat: {result}")
                    
                    # Vérifier si ça correspond au résultat réel (que tu sais être WIN)
                    print(f"[ANALYSIS]    Correspond au WIN réel: {'✅ OUI' if result == 'WIN' else '❌ NON'}")
            
            print(f"\n[ANALYSIS] 🎯 RECOMMANDATION:")
            print(f"[ANALYSIS] 1. Vérifie l'heure EXACTE de ton trade sur Pocket Option")
            print(f"[ANALYSIS] 2. Compare avec les bougies ci-dessus")
            print(f"[ANALYSIS] 3. La bonne hypothèse est celle qui donne WIN")
            
            print(f"\n[ANALYSIS] ✅ Analyse terminée")
            
        except Exception as e:
            print(f"[ANALYSIS] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()

    async def fix_specific_signal(self, signal_id: int, force_result: str = None):
        """
        Corrige manuellement un signal spécifique
        """
        try:
            print(f"\n[FIX] 🔧 Correction manuelle signal #{signal_id}")
            
            if force_result:
                # Forcer un résultat spécifique
                with self.engine.connect() as conn:
                    signal = conn.execute(
                        text("SELECT pair, direction FROM signals WHERE id = :sid"),
                        {"sid": signal_id}
                    ).fetchone()
                
                if signal:
                    pair, direction = signal
                    reason = f"Correction manuelle - Résultat forcé: {force_result}"
                    
                    details = {
                        'reason': reason,
                        'entry_price': None,
                        'exit_price': None,
                        'pips': 0.0,
                        'gale_level': 0
                    }
                    
                    self._update_signal_result(signal_id, force_result, details)
                    print(f"[FIX] ✅ Signal #{signal_id} forcé à: {force_result}")
                    return force_result
            
            # Sinon, re-vérifier normalement
            return await self.verify_single_signal(signal_id)
            
        except Exception as e:
            print(f"[FIX] ❌ Erreur: {e}")
            return 'ERROR'

    def get_win_loss_stats(self):
        """
        Statistiques détaillées des WIN/LOSE
        """
        try:
            with self.engine.connect() as conn:
                stats = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN result = 'INVALID' THEN 1 ELSE 0 END) as invalid,
                        SUM(CASE WHEN result = 'ERROR' THEN 1 ELSE 0 END) as errors
                    FROM signals
                    WHERE result IS NOT NULL
                """)).fetchone()
            
            total, wins, losses, invalid, errors = stats
            
            return {
                'total': total or 0,
                'wins': wins or 0,
                'losses': losses or 0,
                'invalid': invalid or 0,
                'errors': errors or 0,
                'win_rate': round((wins or 0) / max(1, (wins or 0) + (losses or 0)) * 100, 1)
            }
            
        except Exception as e:
            print(f"[STATS] ❌ Erreur: {e}")
            return {}
