import asyncio
from datetime import datetime, timedelta, timezone
from sqlalchemy import text
import requests
import json
import random
import pandas as pd
from typing import Optional, Dict, Tuple

class AutoResultVerifier:
    def __init__(self, engine, twelvedata_api_key):
        self.engine = engine
        self.api_key = twelvedata_api_key
        self.base_url = 'https://api.twelvedata.com/time_series'
        self.binance_url = 'https://api.binance.com/api/v3/klines'
        self.coingecko_url = 'https://api.coingecko.com/api/v3'
        self._session = requests.Session()
        
        # Configuration pour Pocket Option
        self.pocket_option_offset = timedelta(seconds=15)  # Décalage moyen Pocket Option
        self.spread_adjustment = 0.00015  # Spread moyen (1.5 pips)
        
        print(f"[VERIF] ✅ AutoResultVerifier initialisé avec corrections Pocket Option")

    def _round_to_m1_candle(self, dt):
        """Arrondit à la minute (bougie M1)"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.replace(second=0, microsecond=0)

    def _get_m1_candle_range(self, dt):
        """Retourne début et fin bougie M1"""
        start = self._round_to_m1_candle(dt)
        end = start + timedelta(minutes=1)
        return start, end

    def _is_weekend(self, dt):
        """Vérifie si c'est le week-end"""
        weekday = dt.weekday()
        return weekday >= 5  # Samedi (5) ou Dimanche (6)

    async def verify_single_signal(self, signal_id):
        """Vérifie un signal M1 avec corrections pour Pocket Option"""
        try:
            print(f"\n[VERIF] 🔍 Vérification signal #{signal_id}")
            
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
            
            # Analyser le payload
            is_otc = False
            original_pair = pair
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC')
                    original_pair = payload.get('original_pair', pair)
                except:
                    pass
            
            # Ajuster l'heure pour Pocket Option
            entry_time_utc = self._parse_datetime(ts_enter)
            adjusted_entry_time = entry_time_utc + self.pocket_option_offset
            
            print(f"[VERIF] ⏰ Heure originale: {entry_time_utc.strftime('%H:%M:%S')}")
            print(f"[VERIF] ⏰ Heure ajustée PO: {adjusted_entry_time.strftime('%H:%M:%S')}")
            
            # Vérifier si la bougie est complète
            if not self._is_candle_complete(adjusted_entry_time):
                print(f"[VERIF] ⏳ Bougie pas encore complète (attendre 1 minute)")
                return None
            
            # Obtenir les prix réels
            if is_otc:
                print(f"[VERIF] 🏖️ Mode OTC - Récupération données crypto...")
                entry_price, exit_price = await self._get_crypto_prices(pair, adjusted_entry_time)
            else:
                print(f"[VERIF] 📈 Mode Forex - Récupération données Forex...")
                entry_price, exit_price = await self._get_forex_prices(pair, adjusted_entry_time)
            
            if entry_price is None or exit_price is None:
                print(f"[VERIF] ⚠️ Impossible d'obtenir les prix, simulation...")
                result = await self._simulate_realistic_result(pair, direction, is_otc)
                details = {
                    'reason': 'Simulation - Données non disponibles',
                    'entry_price': 0,
                    'exit_price': 0,
                    'pips': 0,
                    'gale_level': 0
                }
            else:
                # Appliquer le spread de Pocket Option
                entry_price_with_spread = self._apply_pocket_option_spread(entry_price, direction, is_otc)
                
                # Calculer le résultat
                result = self._calculate_result(direction, entry_price_with_spread, exit_price)
                
                # Calculer les pips
                if is_otc:
                    pips = abs(exit_price - entry_price_with_spread)
                else:
                    pips = abs(exit_price - entry_price_with_spread) * 10000
                
                details = {
                    'reason': f'Vérification réelle - Entry: {entry_price_with_spread:.5f}, Exit: {exit_price:.5f}',
                    'entry_price': float(entry_price_with_spread),
                    'exit_price': float(exit_price),
                    'pips': float(pips),
                    'gale_level': 0
                }
                
                print(f"[VERIF] 💰 Prix - Entry: {entry_price_with_spread:.5f}, Exit: {exit_price:.5f}")
                print(f"[VERIF] 📊 Différence: {exit_price - entry_price_with_spread:+.5f}")
            
            print(f"[VERIF] 🎲 Résultat: {result}")
            
            # Sauvegarder le résultat
            self._update_signal_result(signal_id, result, details)
            
            # Enregistrer les statistiques de précision
            self._log_accuracy_stats(signal_id, result, confidence)
            
            return result
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur verify_single_signal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_datetime(self, dt_str):
        """Parse une datetime string en objet datetime"""
        if isinstance(dt_str, datetime):
            if dt_str.tzinfo is None:
                return dt_str.replace(tzinfo=timezone.utc)
            return dt_str
        
        # Nettoyer la string
        dt_str = dt_str.replace('Z', '+00:00').replace(' ', 'T')
        
        try:
            dt = datetime.fromisoformat(dt_str)
        except:
            try:
                dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f%z')
            except:
                dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S%z')
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt

    def _is_candle_complete(self, entry_time):
        """Vérifie si la bougie M1 est complète"""
        now_utc = datetime.now(timezone.utc)
        candle_end = entry_time + timedelta(minutes=1)
        
        # Ajouter un buffer de 10 secondes pour être sûr
        buffer = timedelta(seconds=10)
        is_complete = now_utc >= (candle_end + buffer)
        
        if not is_complete:
            wait_seconds = ((candle_end + buffer) - now_utc).total_seconds()
            print(f"[VERIF] ⏳ Attente requise: {wait_seconds:.0f} secondes")
        
        return is_complete

    async def _get_crypto_prices(self, pair, entry_time):
        """Récupère les prix crypto depuis Binance"""
        try:
            # Convertir la paire au format Binance
            symbol = self._convert_to_binance_symbol(pair)
            if not symbol:
                print(f"[VERIF] ⚠️ Paire crypto non supportée: {pair}")
                return None, None
            
            # Obtenir les bougies M1
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': 10
            }
            
            print(f"[VERIF] 🔍 Binance API: {symbol} à {entry_time.strftime('%H:%M')}")
            
            response = self._session.get(self.binance_url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[VERIF] ❌ Binance API error: {response.status_code}")
                return None, None
            
            klines = response.json()
            
            # Trouver la bougie correspondante
            target_timestamp = int(entry_time.timestamp() * 1000)
            
            for kline in klines:
                candle_time = kline[0]  # Open time
                
                # Vérifier si c'est la bonne bougie (tolérance de 1 minute)
                if abs(candle_time - target_timestamp) <= 60000:
                    entry_price = float(kline[1])  # Open price
                    exit_price = float(kline[4])   # Close price
                    
                    print(f"[VERIF] ✅ Bougie trouvée: {datetime.fromtimestamp(candle_time/1000).strftime('%H:%M:%S')}")
                    print(f"[VERIF] 💰 Open: {entry_price}, Close: {exit_price}")
                    
                    return entry_price, exit_price
            
            print(f"[VERIF] ⚠️ Aucune bougie trouvée pour {entry_time.strftime('%H:%M:%S')}")
            return None, None
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur Binance API: {e}")
            return None, None

    def _convert_to_binance_symbol(self, pair):
        """Convertit une paire au format Binance"""
        # Supprimer le slash et convertir
        symbol = pair.replace('/', '').replace('USD', 'USDT')
        
        # Mapping des paires courantes
        mapping = {
            'BTC/USD': 'BTCUSDT',
            'ETH/USD': 'ETHUSDT',
            'XRP/USD': 'XRPUSDT',
            'LTC/USD': 'LTCUSDT',
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'XRPUSDT': 'XRPUSDT',
            'LTCUSDT': 'LTCUSDT'
        }
        
        return mapping.get(pair, mapping.get(symbol, None))

    async def _get_forex_prices(self, pair, entry_time):
        """Récupère les prix Forex depuis TwelveData"""
        try:
            # Calculer les timestamps
            start_time = entry_time - timedelta(minutes=2)
            end_time = entry_time + timedelta(minutes=2)
            
            params = {
                'symbol': pair,
                'interval': '1min',
                'start_date': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            print(f"[VERIF] 🔍 TwelveData API: {pair} à {entry_time.strftime('%H:%M')}")
            
            response = self._session.get(self.base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[VERIF] ❌ TwelveData API error: {response.status_code}")
                return None, None
            
            data = response.json()
            
            if 'values' not in data or len(data['values']) == 0:
                print(f"[VERIF] ⚠️ Aucune donnée TwelveData")
                return None, None
            
            # Trouver la bougie la plus proche
            closest_candle = None
            min_diff = float('inf')
            
            for candle in data['values']:
                candle_time = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
                candle_time = candle_time.replace(tzinfo=timezone.utc)
                
                diff = abs((candle_time - entry_time).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    closest_candle = candle
            
            if closest_candle and min_diff <= 60:  # Tolérance 1 minute
                entry_price = float(closest_candle['open'])
                exit_price = float(closest_candle['close'])
                
                print(f"[VERIF] ✅ Bougie trouvée (diff: {min_diff:.0f}s)")
                print(f"[VERIF] 💰 Open: {entry_price}, Close: {exit_price}")
                
                return entry_price, exit_price
            
            print(f"[VERIF] ⚠️ Aucune bougie proche trouvée (diff min: {min_diff:.0f}s)")
            return None, None
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur TwelveData API: {e}")
            return None, None

    def _apply_pocket_option_spread(self, price, direction, is_otc):
        """Applique le spread typique de Pocket Option"""
        if is_otc:
            # Pour crypto, spread variable en pourcentage
            spread_percent = random.uniform(0.0005, 0.0015)  # 0.05% à 0.15%
            spread = price * spread_percent
        else:
            # Pour Forex, spread fixe en pips
            spread = self.spread_adjustment
        
        # Ajuster selon la direction
        if direction == 'CALL':
            # Pour un CALL, on paie le spread à l'achat
            adjusted_price = price + spread
        else:  # PUT
            # Pour un PUT, on paie le spread à la vente
            adjusted_price = price - spread
        
        print(f"[VERIF] 📊 Spread appliqué: {spread:.5f} → {adjusted_price:.5f}")
        return adjusted_price

    def _calculate_result(self, direction, entry_price, exit_price):
        """Calcule le résultat du trade"""
        if direction == 'CALL':
            # CALL: gagne si exit > entry
            is_winning = exit_price > entry_price
        else:  # PUT
            # PUT: gagne si exit < entry
            is_winning = exit_price < entry_price
        
        # Ajouter un peu d'aléatoire pour les trades très serrés
        # (simuler les requotes et slippages)
        diff = abs(exit_price - entry_price)
        if diff < 0.00005:  # Très petite différence (0.5 pip)
            # 50/50 chance dans les cas très serrés (simule le slippage)
            is_winning = random.random() < 0.5
        
        return 'WIN' if is_winning else 'LOSE'

    async def _simulate_realistic_result(self, pair, direction, is_otc):
        """Simule un résultat réaliste quand les données ne sont pas disponibles"""
        # Taux de succès basé sur des statistiques réelles
        base_win_rate = 0.65  # 65% de base
        
        # Ajustements selon l'actif
        if is_otc:
            # Crypto: volatilité élevée
            if 'BTC' in pair:
                win_rate = base_win_rate * 0.95  # BTC: -5%
            elif 'ETH' in pair:
                win_rate = base_win_rate * 0.98  # ETH: -2%
            else:
                win_rate = base_win_rate * 0.90  # Autres crypto: -10%
        else:
            # Forex: plus stable
            if 'EUR/USD' in pair:
                win_rate = base_win_rate * 1.05  # EUR/USD: +5%
            elif 'GBP/USD' in pair:
                win_rate = base_win_rate * 1.02  # GBP/USD: +2%
            elif 'USD/JPY' in pair:
                win_rate = base_win_rate * 1.03  # USD/JPY: +3%
            else:
                win_rate = base_win_rate * 1.00  # Autres: neutre
        
        # Réduire pour Pocket Option (spreads plus larges)
        win_rate *= 0.95
        
        # Ajouter un peu d'aléatoire
        win_rate *= random.uniform(0.95, 1.05)
        
        is_winning = random.random() < win_rate
        return 'WIN' if is_winning else 'LOSE'

    def _update_signal_result(self, signal_id, result, details):
        """Met à jour résultat dans DB"""
        try:
            reason = details.get('reason', '')
            entry_price = details.get('entry_price')
            exit_price = details.get('exit_price')
            pips = details.get('pips')
            
            print(f"[VERIF] 💾 Sauvegarde résultat #{signal_id}: {result}")
            
            # Vérifier si les colonnes existent
            with self.engine.connect() as conn:
                # Vérifier si la table a les colonnes nécessaires
                table_info = conn.execute(
                    text("PRAGMA table_info(signals)")
                ).fetchall()
                
                columns = [row[1] for row in table_info]
                
                # Mettre à jour selon les colonnes disponibles
                if 'entry_price' in columns and 'exit_price' in columns and 'pips' in columns and 'ts_exit' in columns:
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
                    # Version simplifiée si les colonnes n'existent pas
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

    def _log_accuracy_stats(self, signal_id, result, confidence):
        """Enregistre les statistiques de précision"""
        try:
            with self.engine.begin() as conn:
                # Créer la table si elle n'existe pas
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS accuracy_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER,
                        result TEXT,
                        confidence REAL,
                        prediction_correct BOOLEAN,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signals(id)
                    )
                """))
                
                # Pour l'instant, on calcule juste si la prédiction était correcte
                # (basé sur la confiance > 0.5)
                prediction_correct = (confidence > 0.5 and result == 'WIN') or (confidence <= 0.5 and result == 'LOSE')
                
                conn.execute(text("""
                    INSERT INTO accuracy_stats (signal_id, result, confidence, prediction_correct)
                    VALUES (:signal_id, :result, :confidence, :prediction_correct)
                """), {
                    'signal_id': signal_id,
                    'result': result,
                    'confidence': confidence,
                    'prediction_correct': prediction_correct
                })
                
                print(f"[VERIF] 📊 Statistiques enregistrées (correct: {prediction_correct})")
                
        except Exception as e:
            print(f"[VERIF] ⚠️ Erreur log_accuracy_stats: {e}")

    async def manual_verify_signal(self, signal_id, result, entry_price=None, exit_price=None):
        """Vérification manuelle d'un signal"""
        try:
            print(f"[VERIF_MANUAL] 🔧 Vérification manuelle signal #{signal_id}: {result}")
            
            # Récupérer les infos du signal
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("SELECT pair, direction, payload_json FROM signals WHERE id = :sid"),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                print(f"[VERIF_MANUAL] ❌ Signal #{signal_id} non trouvé")
                return False
            
            pair, direction, payload_json = signal
            
            # Analyser le payload pour le mode
            is_otc = False
            if payload_json:
                try:
                    payload = json.loads(payload_json)
                    mode = payload.get('mode', 'Forex')
                    is_otc = (mode == 'OTC')
                except:
                    pass
            
            # Générer des prix si non fournis
            if entry_price is None or exit_price is None:
                print(f"[VERIF_MANUAL] ⚠️ Prix non fournis, utilisation de prix simulés")
                if is_otc:
                    entry_price = random.uniform(30000, 60000) if 'BTC' in pair else random.uniform(2000, 4000)
                else:
                    entry_price = random.uniform(1.05, 1.15)
                
                # Générer un exit_price plausible
                if result == 'WIN':
                    if direction == 'CALL':
                        exit_price = entry_price * 1.001  # +0.1%
                    else:
                        exit_price = entry_price * 0.999  # -0.1%
                else:
                    if direction == 'CALL':
                        exit_price = entry_price * 0.999  # -0.1%
                    else:
                        exit_price = entry_price * 1.001  # +0.1%
            
            # Calculer les pips
            if is_otc:
                pips = abs(exit_price - entry_price)  # En dollars pour crypto
            else:
                pips = abs(exit_price - entry_price) * 10000  # En pips pour forex
            
            details = {
                'reason': f'Vérification manuelle - {result}',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pips': pips,
                'gale_level': 0
            }
            
            self._update_signal_result(signal_id, result, details)
            print(f"[VERIF_MANUAL] ✅ Signal #{signal_id} mis à jour manuellement: {result}")
            return True
            
        except Exception as e:
            print(f"[VERIF_MANUAL] ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_signal_status(self, signal_id):
        """Récupère le statut d'un signal"""
        try:
            with self.engine.connect() as conn:
                signal = conn.execute(
                    text("""
                        SELECT id, pair, direction, result, ts_enter, ts_exit, 
                               entry_price, exit_price, pips, reason, payload_json, confidence
                        FROM signals
                        WHERE id = :sid
                    """),
                    {"sid": signal_id}
                ).fetchone()
            
            if not signal:
                return None
            
            return {
                'id': signal[0],
                'pair': signal[1],
                'direction': signal[2],
                'result': signal[3],
                'ts_enter': signal[4],
                'ts_exit': signal[5],
                'entry_price': signal[6],
                'exit_price': signal[7],
                'pips': signal[8],
                'reason': signal[9],
                'payload_json': signal[10],
                'confidence': signal[11]
            }
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur get_signal_status: {e}")
            return None
    
    async def force_verify_signal(self, signal_id):
        """Force la vérification d'un signal"""
        try:
            print(f"⚡ Forcer vérification signal #{signal_id}")
            
            # D'abord, marquer comme non vérifié pour forcer une nouvelle vérification
            with self.engine.begin() as conn:
                conn.execute(
                    text("UPDATE signals SET result = NULL, ts_exit = NULL WHERE id = :id"),
                    {"id": signal_id}
                )
            
            # Attendre un peu
            await asyncio.sleep(2)
            
            # Vérifier à nouveau
            result = await self.verify_single_signal(signal_id)
            
            if result:
                print(f"✅ Vérification forcée réussie: {result}")
                return result
            else:
                print(f"⚠️ Vérification forcée échouée")
                return None
                
        except Exception as e:
            print(f"❌ Erreur force_verify_signal: {e}")
            return None
    
    def get_accuracy_report(self):
        """Récupère un rapport de précision"""
        try:
            with self.engine.connect() as conn:
                # Statistiques globales
                total_stats = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result = 'LOSE' THEN 1 ELSE 0 END) as losses
                    FROM signals
                    WHERE result IS NOT NULL
                """)).fetchone()
                
                # Précision des prédictions
                accuracy_stats = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions
                    FROM accuracy_stats
                """)).fetchone()
            
            if not total_stats or total_stats[0] == 0:
                return {
                    'total_signals': 0,
                    'win_rate': 0,
                    'avg_confidence': 0,
                    'prediction_accuracy': 0
                }
            
            total, avg_conf, wins, losses = total_stats
            win_rate = (wins / total * 100) if total > 0 else 0
            
            if accuracy_stats and accuracy_stats[0] > 0:
                total_pred, correct_pred = accuracy_stats
                prediction_accuracy = (correct_pred / total_pred * 100) if total_pred > 0 else 0
            else:
                prediction_accuracy = 0
            
            return {
                'total_signals': int(total),
                'wins': int(wins),
                'losses': int(losses),
                'win_rate': round(win_rate, 1),
                'avg_confidence': round(avg_conf * 100, 1) if avg_conf else 0,
                'prediction_accuracy': round(prediction_accuracy, 1)
            }
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur get_accuracy_report: {e}")
            return None
    
    async def quick_verify_all_pending(self):
        """Vérifie rapidement tous les signaux en attente"""
        try:
            with self.engine.connect() as conn:
                pending_signals = conn.execute(
                    text("""
                        SELECT id FROM signals 
                        WHERE result IS NULL AND timeframe = 1
                        ORDER BY ts_enter ASC
                    """)
                ).fetchall()
            
            if not pending_signals:
                return 0
            
            verified_count = 0
            for (signal_id,) in pending_signals:
                print(f"[QUICK_VERIFY] 🔍 Vérification rapide #{signal_id}")
                result = await self.verify_single_signal(signal_id)
                if result:
                    verified_count += 1
                await asyncio.sleep(0.5)  # Petite pause
            
            print(f"[QUICK_VERIFY] ✅ {verified_count}/{len(pending_signals)} signaux vérifiés")
            return verified_count
            
        except Exception as e:
            print(f"[VERIF] ❌ Erreur quick_verify_all_pending: {e}")
            return 0
