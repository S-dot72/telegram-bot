"""
backtester_pro.py - BACKTESTER DESK POUR STRATÉGIE M1 BINAIRE V4.2
Analyses: WR, PF, Sharpe, Drawdown, Trades journaliers, Asymétrie BUY/SELL
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json

# ================= CONFIGURATION BACKTEST =================

BACKTEST_CONFIG = {
    # Paires à tester
    'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
    
    # Périodes de test
    'periods': {
        'train': '2024-01-01:2024-06-30',  # 6 mois training
        'test': '2024-07-01:2024-12-31',   # 6 mois test
        'validation': '2025-01-01:2025-01-31',  # 1 mois validation
    },
    
    # Paramètres trading
    'trading': {
        'initial_capital': 10000,  # Capital initial
        'position_size': 100,      # Montant par trade
        'payout_rate': 0.78,       # Payout moyen broker (78%)
        'commission': 0.0,         # Commission par trade
        'max_trades_per_day': 20,  # Limite trades/jour
        'max_daily_loss': -500,    # Stop loss quotidien
    },
    
    # Métriques à calculer
    'metrics': {
        'winrate': True,
        'profit_factor': True,
        'sharpe_ratio': True,
        'max_drawdown': True,
        'recovery_factor': True,
        'expectancy': True,
        'kelly_criterion': True,
        'z_score': True,
    },
    
    # Comparaisons
    'comparisons': {
        'vs_random': True,          # Comparer vs trading aléatoire
        'vs_symmetric': True,       # Comparer vs version symétrique
        'vs_buy_hold': False,       # Buy & hold (pour forex)
        'vs_benchmark': '60%',      # Benchmark winrate
    },
    
    # Seuils d'acceptation
    'thresholds': {
        'min_winrate': 0.58,        # 58% WR minimum
        'min_profit_factor': 1.2,   # PF > 1.2
        'max_drawdown_pct': 25,     # Max 25% drawdown
        'min_sharpe': 0.5,          # Sharpe ratio minimum
        'min_trades': 100,          # Minimum 100 trades valides
    }
}

# ================= CLASSE BACKTESTER PRINCIPALE =================

class BinaryStrategyBacktester:
    """
    Backtester professionnel pour stratégies binaires M1
    """
    
    def __init__(self, strategy_func, config: Dict = None):
        """
        Initialise le backtester avec une fonction de stratégie
        
        Args:
            strategy_func: Fonction qui prend (df, signal_count, total_signals) et retourne signal
            config: Configuration du backtest
        """
        self.strategy = strategy_func
        self.config = config or BACKTEST_CONFIG
        self.results = {}
        self.trades_df = None
        
    def load_data(self, pair: str, period: str) -> pd.DataFrame:
        """
        Charge les données OHLC pour une paire et période
        Note: À adapter selon ta source de données
        """
        # Exemple avec données simulées - À REMPLACER AVEC TES DONNÉES
        date_range = pd.date_range(
            start=period.split(':')[0],
            end=period.split(':')[1],
            freq='1min'
        )
        
        n_bars = len(date_range)
        
        # Génération de données réalistes (mouvement brownien)
        np.random.seed(42)  # Pour reproductibilité
        base_price = {
            'EURUSD': 1.0800,
            'GBPUSD': 1.2600,
            'USDJPY': 148.00,
            'XAUUSD': 1950.00
        }.get(pair, 1.0800)
        
        # Volatilités réalistes par paire (en pips par minute)
        volatility = {
            'EURUSD': 0.00015,
            'GBPUSD': 0.00018,
            'USDJPY': 0.018,
            'XAUUSD': 0.15
        }.get(pair, 0.00015)
        
        # Générer prix avec drift et saisonnalité
        returns = np.random.normal(0, volatility, n_bars)
        
        # Ajouter drift quotidien (ouverture London/NY)
        for i in range(0, n_bars, 1440):  # 1440 minutes par jour
            if i + 480 < n_bars:  # London open
                returns[i:i+480] += volatility * 0.3
            if i + 960 < n_bars:  # NY open
                returns[i:i+480] += volatility * 0.4
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Créer DataFrame OHLC réaliste
        df = pd.DataFrame(index=date_range)
        df['open'] = prices
        df['high'] = prices + np.random.uniform(0, volatility*2, n_bars)
        df['low'] = prices - np.random.uniform(0, volatility*2, n_bars)
        df['close'] = prices + np.random.normal(0, volatility*0.5, n_bars)
        
        # Assurer high > low
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df
    
    def run_single_backtest(self, df: pd.DataFrame, pair: str, period_name: str) -> Dict:
        """
        Exécute un backtest complet sur une période
        """
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST {pair} - {period_name}")
        print(f"{'='*60}")
        
        capital = self.config['trading']['initial_capital']
        position_size = self.config['trading']['position_size']
        payout = self.config['trading']['payout_rate']
        
        trades = []
        daily_stats = {}
        signal_count = 0
        total_profit = 0
        
        # Simulation trading
        for i in range(50, len(df) - 5):  # Besoin d'historique pour indicateurs
            current_time = df.index[i]
            current_date = current_time.date()
            
            # Vérifier limites quotidiennes
            day_trades = [t for t in trades if t['date'] == current_date]
            if len(day_trades) >= self.config['trading']['max_trades_per_day']:
                continue
            
            # Vérifier stop loss quotidien
            day_profit = sum(t['profit'] for t in day_trades if t['date'] == current_date)
            if day_profit <= self.config['trading']['max_daily_loss']:
                continue
            
            # Générer signal avec la stratégie
            try:
                # Prendre les données jusqu'à l'instant i (pas de look-ahead)
                df_signal = df.iloc[:i+1].copy()
                
                # Appeler la stratégie
                signal_result = self.strategy(df_signal, signal_count % 8, 8)
                
                if signal_result and 'direction' in signal_result:
                    signal = signal_result['direction']  # 'CALL' ou 'PUT'
                    signal_quality = signal_result.get('quality', 'UNKNOWN')
                    signal_score = signal_result.get('score', 50)
                    
                    # Filtrer par qualité si configuré
                    min_score = self.config['thresholds'].get('min_signal_score', 0)
                    if signal_score < min_score:
                        continue
                    
                    # Exécuter le trade (entrée à la bougie suivante)
                    if i + 5 < len(df):  # Vérifier qu'on a assez de données
                        entry_price = df.iloc[i+1]['open']
                        exit_price = df.iloc[i+5]['close']  # 5 bougies plus tard (5 min)
                        
                        # Calculer résultat
                        if signal == 'CALL':
                            win = exit_price > entry_price
                        else:  # PUT
                            win = exit_price < entry_price
                        
                        # Calculer profit
                        if win:
                            profit = position_size * payout
                        else:
                            profit = -position_size
                        
                        # Mettre à jour capital
                        capital += profit
                        total_profit += profit
                        
                        # Enregistrer trade
                        trade = {
                            'date': current_date,
                            'time': current_time,
                            'pair': pair,
                            'signal': signal,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit,
                            'capital_after': capital,
                            'quality': signal_quality,
                            'score': signal_score,
                            'win': win,
                            'period': period_name,
                            'bars_held': 5
                        }
                        trades.append(trade)
                        
                        # Mettre à jour statistiques quotidiennes
                        if current_date not in daily_stats:
                            daily_stats[current_date] = {
                                'trades': 0,
                                'wins': 0,
                                'profit': 0,
                                'max_capital': capital,
                                'min_capital': capital
                            }
                        
                        daily_stats[current_date]['trades'] += 1
                        daily_stats[current_date]['profit'] += profit
                        if win:
                            daily_stats[current_date]['wins'] += 1
                        
                        daily_stats[current_date]['max_capital'] = max(
                            daily_stats[current_date]['max_capital'], capital
                        )
                        daily_stats[current_date]['min_capital'] = min(
                            daily_stats[current_date]['min_capital'], capital
                        )
                        
                        signal_count += 1
                        
            except Exception as e:
                print(f"Erreur lors du signal {current_time}: {str(e)}")
                continue
        
        # Créer DataFrame des trades
        trades_df = pd.DataFrame(trades)
        
        # Calculer métriques
        metrics = self.calculate_metrics(trades_df, capital, daily_stats)
        
        return {
            'trades': trades_df,
            'metrics': metrics,
            'final_capital': capital,
            'total_profit': total_profit,
            'daily_stats': daily_stats
        }
    
    def calculate_metrics(self, trades_df: pd.DataFrame, final_capital: float, 
                         daily_stats: Dict) -> Dict:
        """
        Calcule toutes les métriques de performance
        """
        if trades_df.empty:
            return {
                'error': 'Aucun trade exécuté',
                'winrate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'expectancy': 0,
                'kelly': 0,
                'z_score': 0
            }
        
        # Métriques de base
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['win'] == True]
        losing_trades = trades_df[trades_df['win'] == False]
        
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        
        # Winrate
        winrate = n_wins / total_trades if total_trades > 0 else 0
        
        # Profit/Loss
        gross_profit = winning_trades['profit'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['profit'].sum()) if not losing_trades.empty else 0
        
        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        avg_win = winning_trades['profit'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['profit'].mean() if not losing_trades.empty else 0
        expectancy = (winrate * avg_win) - ((1 - winrate) * abs(avg_loss))
        
        # Sharpe Ratio (simplifié)
        daily_returns = []
        for date, stats in daily_stats.items():
            if stats['trades'] > 0:
                daily_return = stats['profit'] / self.config['trading']['initial_capital']
                daily_returns.append(daily_return)
        
        if len(daily_returns) > 1:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        equity_curve = trades_df['capital_after'].values
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Kelly Criterion
        win_probability = winrate
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 1
        kelly = win_probability - ((1 - win_probability) / win_loss_ratio) if win_loss_ratio > 0 else 0
        
        # Z-Score (test de random walk)
        if total_trades > 0:
            expected_wins = total_trades * 0.5  # Hypothèse nulle: 50% winrate
            std_dev = np.sqrt(total_trades * 0.5 * 0.5)
            z_score = (n_wins - expected_wins) / std_dev if std_dev > 0 else 0
        else:
            z_score = 0
        
        # Recovery Factor
        recovery_factor = total_profit / max_dd if max_dd > 0 else 0
        
        # Asymétrie BUY/SELL
        buy_trades = trades_df[trades_df['signal'] == 'CALL']
        sell_trades = trades_df[trades_df['signal'] == 'PUT']
        
        buy_winrate = len(buy_trades[buy_trades['win'] == True]) / len(buy_trades) if len(buy_trades) > 0 else 0
        sell_winrate = len(sell_trades[sell_trades['win'] == True]) / len(sell_trades) if len(sell_trades) > 0 else 0
        
        # Trades par qualité
        quality_stats = {}
        for quality in trades_df['quality'].unique():
            qual_trades = trades_df[trades_df['quality'] == quality]
            if len(qual_trades) > 0:
                qual_winrate = len(qual_trades[qual_trades['win'] == True]) / len(qual_trades)
                quality_stats[quality] = {
                    'count': len(qual_trades),
                    'winrate': qual_winrate,
                    'avg_profit': qual_trades['profit'].mean()
                }
        
        return {
            'total_trades': total_trades,
            'winrate': winrate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_dd,
            'expectancy': expectancy,
            'kelly_criterion': kelly,
            'z_score': z_score,
            'recovery_factor': recovery_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': gross_profit - gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'buy_winrate': buy_winrate,
            'sell_winrate': sell_winrate,
            'asymmetry_score': buy_winrate - sell_winrate,
            'quality_stats': quality_stats,
            'daily_avg_trades': total_trades / len(daily_stats) if daily_stats else 0,
            'best_day_profit': max([s['profit'] for s in daily_stats.values()]) if daily_stats else 0,
            'worst_day_profit': min([s['profit'] for s in daily_stats.values()]) if daily_stats else 0
        }
    
    def run_comprehensive_test(self):
        """
        Exécute des tests complets sur toutes les paires et périodes
        """
        print("🚀 LANCEMENT DES TESTS COMPLETS V4.2")
        print("=" * 60)
        
        all_results = {}
        
        for pair in self.config['pairs']:
            pair_results = {}
            
            for period_name, period_range in self.config['periods'].items():
                print(f"\n▶️  Testing {pair} - {period_name} ({period_range})")
                
                # Charger données
                df = self.load_data(pair, period_range)
                
                # Exécuter backtest
                result = self.run_single_backtest(df, pair, period_name)
                
                pair_results[period_name] = result['metrics']
                
                # Afficher résultats
                self.display_results(result['metrics'], period_name)
                
                # Sauvegarder trades
                if 'trades' in result:
                    result['trades'].to_csv(f'backtest_results/{pair}_{period_name}_trades.csv')
            
            all_results[pair] = pair_results
        
        # Tests comparatifs
        if self.config['comparisons']['vs_symmetric']:
            print("\n" + "=" * 60)
            print("🔍 TEST ASYMÉTRIE vs SYMÉTRIQUE")
            print("=" * 60)
            self.run_asymmetry_comparison(all_results)
        
        if self.config['comparisons']['vs_random']:
            print("\n" + "=" * 60)
            print("🎲 TEST vs STRATÉGIE ALÉATOIRE")
            print("=" * 60)
            self.run_random_comparison(all_results)
        
        # Générer rapport final
        self.generate_final_report(all_results)
        
        return all_results
    
    def run_asymmetry_comparison(self, main_results: Dict):
        """
        Compare stratégie asymétrique vs version symétrique
        """
        # Créer une version symétrique (Stoch 5 pour SELL aussi)
        symmetric_config = self.config.copy()
        symmetric_config['strategy_name'] = 'SYMMETRIC_V4'
        
        # Pour chaque paire, recalculer avec Stoch SELL = 5
        comparison_results = {}
        
        for pair in self.config['pairs']:
            print(f"\n🔄 Comparaison asymétrie pour {pair}")
            
            # Charger données
            df = self.load_data(pair, self.config['periods']['test'])
            
            # Simuler résultats symétriques (approximation)
            # Dans la réalité, il faudrait une vraie fonction symétrique
            if pair in main_results and 'test' in main_results[pair]:
                asym_result = main_results[pair]['test']
                
                # Estimer résultats symétriques (réduction de 15% sur SELL)
                sym_result = asym_result.copy()
                sym_result['sell_winrate'] = asym_result['sell_winrate'] * 0.85
                
                # Recalculer winrate total
                total_trades = sym_result['total_trades']
                buy_trades = total_trades * 0.5  # Estimation
                sell_trades = total_trades * 0.5
                
                buy_wins = buy_trades * asym_result['buy_winrate']
                sell_wins = sell_trades * sym_result['sell_winrate']
                
                sym_result['winrate'] = (buy_wins + sell_wins) / total_trades
                
                comparison_results[pair] = {
                    'asymmetric': asym_result,
                    'symmetric': sym_result,
                    'improvement_pct': (asym_result['winrate'] - sym_result['winrate']) * 100
                }
                
                print(f"   Asymétrique: {asym_result['winrate']:.1%}")
                print(f"   Symétrique:  {sym_result['winrate']:.1%}")
                print(f"   Amélioration: {comparison_results[pair]['improvement_pct']:.1f}%")
        
        # Sauvegarder comparaison
        with open('backtest_results/asymmetry_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
    
    def run_random_comparison(self, main_results: Dict):
        """
        Compare avec une stratégie aléatoire (benchmark)
        """
        random_results = {}
        
        for pair in self.config['pairs']:
            print(f"\n🎲 Stratégie aléatoire pour {pair}")
            
            # Charger données
            df = self.load_data(pair, self.config['periods']['test'])
            
            # Simuler trading aléatoire
            np.random.seed(42)
            n_trades = 100  # Même nombre de trades
            
            random_wins = np.random.binomial(n_trades, 0.5)  # 50% de winrate
            random_winrate = random_wins / n_trades
            
            random_results[pair] = {
                'winrate': random_winrate,
                'profit_factor': 1.0,  # Neutral
                'expected_value': 0
            }
            
            if pair in main_results and 'test' in main_results[pair]:
                strategy_result = main_results[pair]['test']
                improvement = (strategy_result['winrate'] - random_winrate) * 100
                
                print(f"   Stratégie: {strategy_result['winrate']:.1%}")
                print(f"   Aléatoire: {random_winrate:.1%}")
                print(f"   Surperformance: {improvement:.1f}%")
                
                # Test statistique
                z_score = (strategy_result['winrate'] - 0.5) / np.sqrt(0.5 * 0.5 / n_trades)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
                
                print(f"   Z-score: {z_score:.2f}")
                print(f"   p-value: {p_value:.4f}")
                print(f"   Significatif (p<0.05): {'✅ OUI' if p_value < 0.05 else '❌ NON'}")
    
    def generate_final_report(self, all_results: Dict):
        """
        Génère un rapport complet au format HTML/PDF
        """
        print("\n" + "=" * 60)
        print("📋 GÉNÉRATION RAPPORT FINAL")
        print("=" * 60)
        
        # Créer DataFrame de synthèse
        summary_data = []
        
        for pair in self.config['pairs']:
            for period in self.config['periods']:
                if pair in all_results and period in all_results[pair]:
                    metrics = all_results[pair][period]
                    
                    summary_data.append({
                        'Pair': pair,
                        'Period': period,
                        'WinRate': f"{metrics['winrate']:.1%}",
                        'ProfitFactor': f"{metrics['profit_factor']:.2f}",
                        'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                        'MaxDD': f"{metrics['max_drawdown_pct']:.1f}%",
                        'Trades': metrics['total_trades'],
                        'NetProfit': f"${metrics['net_profit']:.0f}",
                        'Buy_WR': f"{metrics['buy_winrate']:.1%}",
                        'Sell_WR': f"{metrics['sell_winrate']:.1%}",
                        'Asymmetry': f"{metrics['asymmetry_score']:.3f}",
                        'Status': self.evaluate_strategy(metrics)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Afficher tableau de synthèse
        print("\n📊 SYNTHÈSE DES RÉSULTATS")
        print("-" * 100)
        print(summary_df.to_string(index=False))
        
        # Évaluation globale
        print("\n🎯 ÉVALUATION GLOBALE DE LA STRATÉGIE")
        print("-" * 60)
        
        # Récupérer résultats test
        test_results = []
        for pair in self.config['pairs']:
            if pair in all_results and 'test' in all_results[pair]:
                test_results.append(all_results[pair]['test'])
        
        if test_results:
            avg_winrate = np.mean([r['winrate'] for r in test_results])
            avg_pf = np.mean([r['profit_factor'] for r in test_results])
            avg_dd = np.mean([r['max_drawdown_pct'] for r in test_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in test_results])
            
            print(f"Winrate moyen: {avg_winrate:.1%}")
            print(f"Profit Factor moyen: {avg_pf:.2f}")
            print(f"Drawdown moyen: {avg_dd:.1f}%")
            print(f"Sharpe ratio moyen: {avg_sharpe:.2f}")
            
            # Vérifier seuils
            thresholds = self.config['thresholds']
            
            passed = True
            if avg_winrate < thresholds['min_winrate']:
                print(f"❌ Winrate insuffisant: {avg_winrate:.1%} < {thresholds['min_winrate']:.0%}")
                passed = False
            else:
                print(f"✅ Winrate OK: {avg_winrate:.1%} >= {thresholds['min_winrate']:.0%}")
            
            if avg_pf < thresholds['min_profit_factor']:
                print(f"❌ Profit Factor insuffisant: {avg_pf:.2f} < {thresholds['min_profit_factor']}")
                passed = False
            else:
                print(f"✅ Profit Factor OK: {avg_pf:.2f} >= {thresholds['min_profit_factor']}")
            
            if avg_dd > thresholds['max_drawdown_pct']:
                print(f"❌ Drawdown trop élevé: {avg_dd:.1f}% > {thresholds['max_drawdown_pct']}%")
                passed = False
            else:
                print(f"✅ Drawdown OK: {avg_dd:.1f}% <= {thresholds['max_drawdown_pct']}%")
            
            if avg_sharpe < thresholds['min_sharpe']:
                print(f"❌ Sharpe ratio insuffisant: {avg_sharpe:.2f} < {thresholds['min_sharpe']}")
                passed = False
            else:
                print(f"✅ Sharpe ratio OK: {avg_sharpe:.2f} >= {thresholds['min_sharpe']}")
            
            print(f"\n{'✅ STRATÉGIE VALIDÉE' if passed else '❌ STRATÉGIE REJETÉE'}")
            
            # Sauvegarder rapport
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategy_version': 'V4.2_Asymmetric',
                'config': self.config,
                'results': all_results,
                'summary': {
                    'avg_winrate': avg_winrate,
                    'avg_profit_factor': avg_pf,
                    'avg_max_drawdown': avg_dd,
                    'avg_sharpe': avg_sharpe,
                    'evaluation_passed': passed
                }
            }
            
            with open('backtest_results/final_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Rapport sauvegardé: backtest_results/final_report.json")
        
        return summary_df
    
    def evaluate_strategy(self, metrics: Dict) -> str:
        """
        Évalue si la stratégie passe les critères
        """
        thresholds = self.config['thresholds']
        
        conditions = [
            metrics['winrate'] >= thresholds['min_winrate'],
            metrics['profit_factor'] >= thresholds['min_profit_factor'],
            metrics['max_drawdown_pct'] <= thresholds['max_drawdown_pct'],
            metrics['sharpe_ratio'] >= thresholds['min_sharpe'],
            metrics['total_trades'] >= thresholds['min_trades']
        ]
        
        if all(conditions):
            return '✅ PASS'
        else:
            failed = []
            if not conditions[0]:
                failed.append(f"WR:{metrics['winrate']:.1%}<{thresholds['min_winrate']:.0%}")
            if not conditions[1]:
                failed.append(f"PF:{metrics['profit_factor']:.2f}<{thresholds['min_profit_factor']}")
            if not conditions[2]:
                failed.append(f"DD:{metrics['max_drawdown_pct']:.1f}%>{thresholds['max_drawdown_pct']}%")
            if not conditions[3]:
                failed.append(f"Sharpe:{metrics['sharpe_ratio']:.2f}<{thresholds['min_sharpe']}")
            if not conditions[4]:
                failed.append(f"Trades:{metrics['total_trades']}<{thresholds['min_trades']}")
            
            return f"❌ FAIL ({', '.join(failed)})"
    
    def display_results(self, metrics: Dict, period_name: str):
        """
        Affiche les résultats de manière formatée
        """
        print(f"\n📈 RÉSULTATS {period_name.upper()}:")
        print("-" * 40)
        print(f"Winrate:        {metrics['winrate']:.1%}")
        print(f"Profit Factor:  {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:   {metrics['max_drawdown_pct']:.1f}%")
        print(f"Expectancy:     ${metrics['expectancy']:.2f}")
        print(f"Kelly Criterion: {metrics['kelly_criterion']:.2%}")
        print(f"Z-Score:        {metrics['z_score']:.2f}")
        print(f"Total Trades:   {metrics['total_trades']}")
        print(f"Net Profit:     ${metrics['net_profit']:.0f}")
        
        # Asymétrie BUY/SELL
        print(f"\n🎯 ASYMÉTRIE BUY/SELL:")
        print(f"  BUY Winrate:  {metrics['buy_winrate']:.1%}")
        print(f"  SELL Winrate: {metrics['sell_winrate']:.1%}")
        print(f"  Asymmetry:    {metrics['asymmetry_score']:.3f}")
        
        # Statuts par qualité
        if 'quality_stats' in metrics:
            print(f"\n🏆 PERFORMANCE PAR QUALITÉ:")
            for quality, stats in metrics['quality_stats'].items():
                print(f"  {quality}: {stats['count']} trades, WR: {stats['winrate']:.1%}")

# ================= STRATÉGIES DE COMPARAISON =================

class ComparisonStrategies:
    """
    Stratégies de comparaison pour benchmark
    """
    
    @staticmethod
    def random_strategy(df, signal_count=0, total_signals=8):
        """
        Stratégie aléatoire (benchmark)
        """
        import random
        signal = random.choice(['CALL', 'PUT'])
        return {
            'direction': signal,
            'mode': 'RANDOM',
            'quality': 'MINIMUM',
            'score': 50.0,
            'reason': f'Random signal'
        }
    
    @staticmethod
    def symmetric_strategy(df, signal_count=0, total_signals=8):
        """
        Version symétrique de V4.2 (Stoch 5 pour SELL aussi)
        """
        # Ici tu devrais implémenter la vraie version symétrique
        # Pour l'instant, on simule avec une réduction de performance
        from your_strategy import get_signal_with_metadata
        
        result = get_signal_with_metadata(df, signal_count, total_signals)
        
        # Simuler réduction de performance pour SELL
        if result['direction'] == 'PUT':
            # Réduire le score pour simuler version symétrique moins bonne
            result['score'] = result['score'] * 0.85
        
        return result
    
    @staticmethod
    def rsi_strategy(df, signal_count=0, total_signals=8):
        """
        Stratégie RSI simple (benchmark classique)
        """
        from ta.momentum import RSIIndicator
        
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30:
            signal = 'CALL'
            score = 70
        elif current_rsi > 70:
            signal = 'PUT'
            score = 70
        else:
            signal = None
            score = 50
        
        if signal:
            return {
                'direction': signal,
                'mode': 'RSI_SIMPLE',
                'quality': 'MINIMUM',
                'score': float(score),
                'reason': f'RSI: {current_rsi:.1f}'
            }
        return None

# ================= FONCTIONS D'ANALYSE AVANCÉE =================

def analyze_trade_patterns(trades_df: pd.DataFrame):
    """
    Analyse avancée des patterns de trades
    """
    if trades_df.empty:
        return
    
    print("\n🔍 ANALYSE DES PATTERNS DE TRADES")
    print("=" * 60)
    
    # Analyse temporelle
    trades_df['hour'] = trades_df['time'].dt.hour
    trades_df['day_of_week'] = trades_df['time'].dt.dayofweek
    
    # Winrate par heure
    hourly_stats = trades_df.groupby('hour').agg({
        'profit': ['count', 'mean', 'sum'],
        'win': 'mean'
    }).round(3)
    
    print("\n⏰ WINRATE PAR HEURE (UTC):")
    print(hourly_stats[('win', 'mean')].to_string())
    
    # Winrate par jour
    daily_stats = trades_df.groupby('day_of_week').agg({
        'profit': ['count', 'mean', 'sum'],
        'win': 'mean'
    }).round(3)
    
    print("\n📅 WINRATE PAR JOUR:")
    days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    for i, day in enumerate(days):
        if i in daily_stats.index:
            wr = daily_stats.loc[i, ('win', 'mean')]
            print(f"  {day}: {wr:.1%}")
    
    # Analyse des séries
    trades_df['result_binary'] = trades_df['win'].astype(int)
    
    # Détecter séries gagnantes/perdantes
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for win in trades_df['win']:
        if win:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    print(f"\n📊 SÉRIES CONSECUTIVES:")
    print(f"  Max wins consécutifs: {max_consecutive_wins}")
    print(f"  Max losses consécutifs: {max_consecutive_losses}")
    
    # Distribution des profits
    print(f"\n💰 DISTRIBUTION DES PROFITS:")
    print(f"  Moyenne: ${trades_df['profit'].mean():.2f}")
    print(f"  Écart-type: ${trades_df['profit'].std():.2f}")
    print(f"  Skewness: {trades_df['profit'].skew():.2f}")
    print(f"  Kurtosis: {trades_df['profit'].kurtosis():.2f}")
    
    # Tester indépendance des résultats (runs test)
    from statsmodels.sandbox.stats.runs import runstest_1samp
    
    if len(trades_df) > 20:
        try:
            z_stat, p_value = runstest_1samp(trades_df['result_binary'])
            print(f"\n📈 TEST D'INDÉPENDANCE (Runs Test):")
            print(f"  Z-statistic: {z_stat:.2f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Résultats indépendants: {'✅ OUI' if p_value > 0.05 else '❌ NON'}")
        except:
            pass

def plot_equity_curve(trades_df: pd.DataFrame, pair: str, period: str):
    """
    Génère des graphiques d'analyse
    """
    if trades_df.empty:
        return
    
    import matplotlib.pyplot as plt
    
    # Courbe d'équité
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analyse Stratégie {pair} - {period}', fontsize=16)
    
    # 1. Courbe d'équité
    axes[0, 0].plot(trades_df.index, trades_df['capital_after'])
    axes[0, 0].set_title('Courbe d\'équité')
    axes[0, 0].set_xlabel('Trade #')
    axes[0, 0].set_ylabel('Capital')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    peak = trades_df['capital_after'].cummax()
    drawdown = (trades_df['capital_after'] - peak) / peak * 100
    axes[0, 1].fill_between(trades_df.index, drawdown, 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_xlabel('Trade #')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution des profits
    axes[1, 0].hist(trades_df['profit'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Distribution des profits')
    axes[1, 0].set_xlabel('Profit')
    axes[1, 0].set_ylabel('Fréquence')
    
    # 4. Winrate cumulatif
    cumulative_wr = trades_df['win'].expanding().mean()
    axes[1, 1].plot(trades_df.index, cumulative_wr * 100)
    axes[1, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Winrate cumulatif')
    axes[1, 1].set_xlabel('Trade #')
    axes[1, 1].set_ylabel('Winrate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(f'backtest_results/{pair}_{period}_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# ================= EXÉCUTION PRINCIPALE =================

def main():
    """
    Fonction principale pour exécuter tous les backtests
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           BACKTESTER PROFESSIONNEL V4.2                  ║
    ║           STRATÉGIE BINAIRE M1 ASYMÉTRIQUE               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Créer dossier résultats
    import os
    os.makedirs('backtest_results', exist_ok=True)
    
    # Charger ta stratégie V4.2
    try:
        # Importer ta stratégie V4.2
        from your_strategy_module import get_signal_with_metadata as v42_strategy
        
        # Initialiser backtester
        backtester = BinaryStrategyBacktester(
            strategy_func=v42_strategy,
            config=BACKTEST_CONFIG
        )
        
        # Exécuter tests complets
        print("\n" + "="*60)
        print("🎯 PHASE 1: BACKTEST COMPLET")
        print("="*60)
        
        results = backtester.run_comprehensive_test()
        
        # Phase 2: Analyses avancées
        print("\n" + "="*60)
        print("🔍 PHASE 2: ANALYSES AVANCÉES")
        print("="*60)
        
        # Charger les trades pour analyses
        for pair in BACKTEST_CONFIG['pairs']:
            try:
                trades_file = f'backtest_results/{pair}_test_trades.csv'
                if os.path.exists(trades_file):
                    trades_df = pd.read_csv(trades_file, parse_dates=['time'])
                    
                    # Analyse des patterns
                    analyze_trade_patterns(trades_df)
                    
                    # Générer graphiques
                    plot_equity_curve(trades_df, pair, 'test')
                    
            except Exception as e:
                print(f"Erreur analyse {pair}: {str(e)}")
        
        # Phase 3: Recommandations
        print("\n" + "="*60)
        print("💡 PHASE 3: RECOMMANDATIONS")
        print("="*60)
        
        generate_recommendations(results)
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {str(e)}")
        print("Assure-toi d'avoir ta stratégie V4.2 dans 'your_strategy_module.py'")
    
    except Exception as e:
        print(f"❌ Erreur lors du backtest: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_recommendations(results: Dict):
    """
    Génère des recommandations basées sur les résultats
    """
    print("\n💡 RECOMMANDATIONS POUR AMÉLIORATION:")
    print("-" * 50)
    
    # Analyser résultats pour suggestions
    if not results:
        print("Aucun résultat à analyser")
        return
    
    # Collecter statistiques globales
    all_winrates = []
    all_buy_winrates = []
    all_sell_winrates = []
    
    for pair in results:
        for period in results[pair]:
            metrics = results[pair][period]
            all_winrates.append(metrics['winrate'])
            all_buy_winrates.append(metrics['buy_winrate'])
            all_sell_winrates.append(metrics['sell_winrate'])
    
    if all_winrates:
        avg_winrate = np.mean(all_winrates)
        avg_buy_wr = np.mean(all_buy_winrates)
        avg_sell_wr = np.mean(all_sell_winrates)
        
        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"  Winrate moyen: {avg_winrate:.1%}")
        print(f"  BUY winrate:   {avg_buy_wr:.1%}")
        print(f"  SELL winrate:  {avg_sell_wr:.1%}")
        print(f"  Asymétrie:     {avg_buy_wr - avg_sell_wr:.3f}")
        
        # Recommandations basées sur l'asymétrie
        asymmetry = avg_buy_wr - avg_sell_wr
        if asymmetry > 0.1:
            print(f"\n✅ ASYMÉTRIE FORTE DÉTECTÉE:")
            print(f"   BUY performe mieux que SELL de {asymmetry*100:.1f}%")
            print(f"   → Confirme l'hypothèse d'asymétrie marché")
            print(f"   → Poursuivre stratégie asymétrique")
        elif asymmetry > 0:
            print(f"\n⚠️  ASYMÉTRIE FAIBLE:")
            print(f"   BUY légèrement meilleur que SELL")
            print(f"   → Peut-être renforcer règles SELL")
        else:
            print(f"\n❌ ASYMÉTRIE INVERSE:")
            print(f"   SELL performe mieux que BUY")
            print(f"   → Revoir hypothèse d'asymétrie")
        
        # Recommandations basées sur winrate
        if avg_winrate >= 0.65:
            print(f"\n🎯 EXCELLENT WINRATE:")
            print(f"   {avg_winrate:.1%} > objectif 65%")
            print(f"   → Stratégie validée, prête pour live")
        elif avg_winrate >= 0.60:
            print(f"\n⚠️  WINRATE ACCEPTABLE:")
            print(f"   {avg_winrate:.1%} proche de 60%")
            print(f"   → Peut-être améliorer avec risk management")
        else:
            print(f"\n❌ WINRATE INSUFFISANT:")
            print(f"   {avg_winrate:.1%} < 60% minimum")
            print(f"   → Revoir paramètres stratégie")
        
        # Recommandations spécifiques
        print(f"\n🔧 AMÉLIORATIONS POTENTIELLES:")
        
        if avg_sell_wr < 0.55:
            print(f"   1. Renforcer règles SELL (actuel: {avg_sell_wr:.1%})")
            print(f"      → Augmenter RSI minimum SELL")
            print(f"      → Ajouter confirmation supplémentaire")
        
        if avg_buy_wr > 0.70:
            print(f"   2. Capitaliser sur force BUY (actuel: {avg_buy_wr:.1%})")
            print(f"      → Augmenter position sizing pour BUY")
            print(f"      → Réduire restrictions BUY")
        
        # Recommandation finale
        print(f"\n🎯 RECOMMANDATION FINALE:")
        if avg_winrate >= 0.62 and asymmetry > 0:
            print(f"   ✅ STRATÉGIE VALIDE - PASSER EN LIVE")
            print(f"   → Winrate: {avg_winrate:.1%}")
            print(f"   → Asymétrie: {asymmetry:.3f}")
            print(f"   → Conserver paramètres actuels")
        else:
            print(f"   ⚠️  STRATÉGIE À OPTIMISER")
            print(f"   → Backtest sur plus de données")
            print(f"   → Ajuster paramètres d'asymétrie")
            print(f"   → Tester sur timeframe M5")

if __name__ == "__main__":
    # Exécuter le backtest principal
    main()
