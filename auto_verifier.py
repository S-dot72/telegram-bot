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
        print(f"[VERIF] 🎯 CORRECTION: Trade M1 = MÊME bougie (open vs close)")

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

    def _get_candle_at_time(self, pair: str, timestamp: datetime, is_otc: bool = False) -> Tuple[float, float, float, float]:
        """
        Récupère la bougie M1 qui CONTIENT le timestamp
        Retourne (open, high, low, close) de la même bougie
        """
        try:
            # NORMALISER à la minute pleine (début de la bougie)
            candle_start = timestamp.replace(second=0, microsecond=0)
            
            print(f"[VERIF_CANDLE] 🔍 Recherche bougie M1 pour {pair}")
            print(f"[VERIF_CANDLE] 🕐 Timestamp: {timestamp}")
            print(f"[VERIF_CANDLE] 🕐 Début bougie: {candle_start}")
            print(f"[VERIF_CANDLE] 🕐 Fin bougie: {candle_start + timedelta(minutes=1)}")
            
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
            end_ms = start_ms + 60000  # +1 minute
            
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
                        if candle_actual_start == start_ms:
                            open_price = float(candle[1])
                            high_price = float(candle[2])
                            low_price = float(candle[3])
                            close_price = float(candle[4])
                            
                            print(f"[VERIF_CRYPTO] ✅ Bougie trouvée: {candle_start}")
                            print(f"[VERIF_CRYPTO] 📊 O={open_price:.5f}, C={close_price:.5f}")
                            
                            return open_price, high_price, low_price, close_price
                        else:
                            print(f"[VERIF_CRYPTO] ⚠️ Mauvais début bougie: {candle_actual_start} vs {start_ms}")
            
            print(f"[VERIF_CRYPTO] ⚠️ Pas de bougie pour {pair} à {candle_start}")
            return None, None, None, None
            
        except Exception as e:
            print(f"[VERIF_CRYPTO] ❌ Erreur: {e}")
            return None, None, None, None

    def _get_forex_candle(self, pair: str, candle_start: datetime) -> Tuple[float, float, float, float]:
        """Récupère une bougie forex spécifique (début à candle_start)"""
        try:
            # Formater pour TwelveData
            start_date = candle_start.strftime('%Y-%m-%d %H:%M:%S')
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
        VÉRIFICATION CORRECTE pour trading binaire M1
        Règle: Trade M1 = MÊME bougie (open vs close)
        """
        try:
            print(f"\n[VERIF] 🔍 Vérification signal #{signal_id}")
            print(f"[VERIF] 🎯 Règle: Trade M1 = OPEN vs CLOSE de la MÊME bougie")
            
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
            
            # CORRECTION CRITIQUE: 
            # En trading binaire M1, on trade la bougie qui commence à l'heure du signal
            # Ex: signal à 14:41 -> on trade la bougie 14:41:00-14:41:59
            
            # 1. Trouver le début de la bougie M1 (normaliser à la minute pleine)
            candle_start = ts_enter.replace(second=0, microsecond=0)
            candle_end = candle_start + timedelta(minutes=1)
            
            print(f"\n[VERIF] 🔧 LOGIQUE M1 CORRECTE:")
            print(f"[VERIF] 🕐 Signal reçu: {ts_enter}")
            print(f"[VERIF] 🕐 Bougie tradée: {candle_start} → {candle_end}")
            print(f"[VERIF] 📊 Durée: 1 minute exacte (même bougie)")
            
            # 2. Récupérer la bougie COMPLÈTE (open, high, low, close)
            open_price, high_price, low_price, close_price = self._get_candle_at_time(
                pair, candle_start, is_otc
            )
            
            print(f"\n[VERIF] 📈 PRIX DE LA BOUGIE:")
            print(f"[VERIF] 💰 Open (entrée): {open_price}")
            print(f"[VERIF] 💰 Close (sortie): {close_price}")
            print(f"[VERIF] 📊 High: {high_price}, Low: {low_price}")
            
            # Vérifier les données
            if open_price is None or close_price is None:
                print(f"[VERIF] ❌ Données manquantes - INVALID")
                reason = f"Bougie M1 incomplète pour {pair} à {candle_start}"
                details = {
                    'reason': reason,
                    'entry_price': None,
                    'exit_price': None,
                    'pips': 0.0,
                    'gale_level': 0
                }
                self._update_signal_result(signal_id, 'INVALID', details)
                return 'INVALID'
            
            # 3. Déterminer le résultat
            # Règle: CALL = gagnant si close > open
            #        PUT = gagnant si close < open
            if direction == "CALL":
                if close_price > open_price:
                    result = "WIN"
                    reason_detail = f"Close ({close_price:.6f}) > Open ({open_price:.6f})"
                elif close_price < open_price:
                    result = "LOSE"
                    reason_detail = f"Close ({close_price:.6f}) < Open ({open_price:.6f})"
                else:
                    result = "DRAW"
                    reason_detail = f"Close ({close_price:.6f}) = Open ({open_price:.6f})"
            else:  # PUT
                if close_price < open_price:
                    result = "WIN"
                    reason_detail = f"Close ({close_price:.6f}) < Open ({open_price:.6f})"
                elif close_price > open_price:
                    result = "LOSE"
                    reason_detail = f"Close ({close_price:.6f}) > Open ({open_price:.6f})"
                else:
                    result = "DRAW"
                    reason_detail = f"Close ({close_price:.6f}) = Open ({open_price:.6f})"
            
            # 4. Calculer les métriques
            price_change = close_price - open_price
            price_change_pct = (price_change / open_price * 100) if open_price != 0 else 0
            
            if is_otc and ('BTC' in pair or 'ETH' in pair or 'TRX' in pair or 'LTC' in pair):
                pips = abs(price_change)
                diff_text = f"${price_change:+.6f}"
            else:
                pips = abs(price_change) * 10000
                diff_text = f"{price_change:+.5f} ({pips:.1f} pips)"
            
            # 5. Construire la raison
            reason = (
                f"M1: {pair} {direction} | "
                f"Bougie: {candle_start.strftime('%H:%M')}-{candle_end.strftime('%H:%M')} | "
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
            print(f"[VERIF] 📊 Direction: {direction}")
            print(f"[VERIF] 📊 Bougie: {candle_start.strftime('%H:%M')}-{candle_end.strftime('%H:%M')}")
            print(f"[VERIF] 💰 Open: {open_price:.6f}")
            print(f"[VERIF] 💰 Close: {close_price:.6f}")
            print(f"[VERIF] 📈 Changement: {price_change:.6f} ({price_change_pct:+.3f}%)")
            print(f"[VERIF] ✅ Trade M1 correctement évalué")
            
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

    # ... [Le reste des méthodes reste inchangé] ...
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

    def debug_signal_logic(self, signal_id: int):
        """
        DEBUG: Affiche la logique de vérification pour un signal
        """
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, ts_enter, result, reason, payload_json
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[DEBUG] ❌ Signal #{signal_id} non trouvé")
                return
            
            sig_id, pair, direction, ts_enter, result, reason, payload_json = signal
            
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
            
            print(f"\n{'='*60}")
            print(f"[DEBUG] 🔧 LOGIQUE M1 - Signal #{signal_id}")
            print(f"{'='*60}")
            print(f"[DEBUG] 📊 Pair: {pair}")
            print(f"[DEBUG] 📊 Direction: {direction}")
            print(f"[DEBUG] 🕐 Heure signal: {ts_enter_dt}")
            
            # Calculer la bougie M1
            candle_start = ts_enter_dt.replace(second=0, microsecond=0)
            candle_end = candle_start + timedelta(minutes=1)
            
            print(f"\n[DEBUG] 🔧 CALCUL BOUGIE M1:")
            print(f"[DEBUG] 🕐 Début bougie: {candle_start}")
            print(f"[DEBUG] 🕐 Fin bougie: {candle_end}")
            print(f"[DEBUG] ⏱️ Durée: 1 minute")
            
            # Récupérer la bougie
            print(f"\n[DEBUG] 🔍 RÉCUPÉRATION DONNÉES:")
            open_price, high_price, low_price, close_price = self._get_candle_at_time(
                pair, candle_start, is_otc
            )
            
            if open_price and close_price:
                print(f"[DEBUG] 📈 Bougie trouvée:")
                print(f"[DEBUG] 💰 Open: {open_price:.6f}")
                print(f"[DEBUG] 💰 Close: {close_price:.6f}")
                print(f"[DEBUG] 📊 High: {high_price:.6f}, Low: {low_price:.6f}")
                
                # Calculer le résultat
                price_change = close_price - open_price
                
                print(f"\n[DEBUG] 🎯 CALCUL RÉSULTAT:")
                print(f"[DEBUG] 📈 Changement: {price_change:.6f}")
                
                if direction == "CALL":
                    expected = "UP"
                    result_calc = "WIN" if price_change > 0 else "LOSE" if price_change < 0 else "DRAW"
                else:
                    expected = "DOWN"
                    result_calc = "WIN" if price_change < 0 else "LOSE" if price_change > 0 else "DRAW"
                
                print(f"[DEBUG] 📊 Direction attendue: {expected}")
                print(f"[DEBUG] 📊 Direction réelle: {'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'}")
                print(f"[DEBUG] 🎯 Résultat calculé: {result_calc}")
                print(f"[DEBUG] 📊 Résultat actuel: {result}")
                
                if result and result_calc != result:
                    print(f"\n[DEBUG] ⚠️ INCOHÉRENCE DÉTECTÉE!")
                    print(f"[DEBUG] ❌ Résultat DB: {result}")
                    print(f"[DEBUG] ✅ Résultat calculé: {result_calc}")
                    print(f"[DEBUG] 💡 Le signal pourrait être mal évalué")
                else:
                    print(f"\n[DEBUG] ✅ Cohérence vérifiée")
            
            print(f"\n[DEBUG] ✅ Diagnostic terminé")
            
        except Exception as e:
            print(f"[DEBUG] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()

    def get_m1_statistics(self):
        """
        Statistiques spécifiques pour les trades M1
        """
        try:
            with self.engine.connect() as conn:
                stats = conn.execute(text("""
                    SELECT 
                        pair,
                        direction,
                        COUNT(*) as total,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN result IN ('INVALID', 'ERROR') THEN 1 ELSE 0 END) as invalid,
                        AVG(confidence) as avg_confidence
                    FROM signals
                    WHERE result IS NOT NULL
                    GROUP BY pair, direction
                    ORDER BY pair, direction
                """)).fetchall()
            
            result = {}
            for pair, direction, total, wins, losses, invalid, avg_conf in stats:
                valid_total = wins + losses
                if valid_total > 0:
                    win_rate = wins / valid_total
                    
                    if pair not in result:
                        result[pair] = {}
                    
                    result[pair][direction] = {
                        'total': total,
                        'valid_trades': valid_total,
                        'wins': wins,
                        'losses': losses,
                        'invalid': invalid,
                        'win_rate': round(win_rate, 3),
                        'avg_confidence': round(avg_conf * 100, 1) if avg_conf else 0
                    }
            
            # Calculer les totaux par paire
            for pair in result:
                total_calls = result[pair].get('CALL', {}).get('valid_trades', 0)
                total_puts = result[pair].get('PUT', {}).get('valid_trades', 0)
                total_trades = total_calls + total_puts
                
                if total_trades > 0:
                    total_wins = (result[pair].get('CALL', {}).get('wins', 0) + 
                                 result[pair].get('PUT', {}).get('wins', 0))
                    overall_win_rate = total_wins / total_trades
                    
                    result[pair]['OVERALL'] = {
                        'total_trades': total_trades,
                        'total_wins': total_wins,
                        'total_losses': total_trades - total_wins,
                        'win_rate': round(overall_win_rate, 3)
                    }
            
            return result
            
        except Exception as e:
            print(f"[STATS] ❌ Erreur: {e}")
            return {}
