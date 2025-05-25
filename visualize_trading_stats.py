#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualiseur de statistiques de trading simplifié

Ce script permet de visualiser des statistiques de trading à partir de fichiers
CSV, JSON ou d'autres formats exportés par le script train_rl_agent.py.

Usage:
    python visualize_trading_stats.py --file logs/trade_history.csv --type trades
    python visualize_trading_stats.py --file logs/portfolio_history.csv --type portfolio
"""

import os
import sys
import json
import argparse
import datetime
from collections import defaultdict

# Importations optionnelles
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("AVERTISSEMENT: pandas n'est pas installé. Certaines fonctionnalités seront limitées.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("AVERTISSEMENT: rich n'est pas installé. L'affichage sera en mode texte simple.")
    
    # Classe minimale pour compatibilité
    class Console:
        def print(self, text, *args, **kwargs):
            # Ignorer les balises de formatage rich
            import re
            clean_text = re.sub(r'\[.*?\]', '', text)
            clean_text = clean_text.replace('[/', '')
            print(clean_text)
    console = Console()

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("AVERTISSEMENT: matplotlib n'est pas installé. Les graphiques ne seront pas disponibles.")


def load_data(file_path, data_type="trades"):
    """
    Charge les données depuis un fichier
    
    Args:
        file_path (str): Chemin vers le fichier
        data_type (str): Type de données ('trades', 'portfolio', 'tier')
        
    Returns:
        data: Les données chargées (DataFrame ou dict)
    """
    if not os.path.exists(file_path):
        console.print(f"[red]Erreur: Le fichier {file_path} n'existe pas[/red]")
        return None
    
    # Déterminer le format de fichier
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if PANDAS_AVAILABLE:
            if file_ext == '.csv':
                data = pd.read_csv(file_path)
                # Convertir les timestamps en datetime si présents
                if 'timestamp' in data.columns:
                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                    except:
                        pass
                return data
            elif file_ext == '.json':
                data = pd.read_json(file_path)
                return data
            elif file_ext == '.pickle' or file_ext == '.pkl':
                data = pd.read_pickle(file_path)
                return data
        else:
            # Fallback pour les formats simples sans pandas
            if file_ext == '.csv':
                data = []
                with open(file_path, 'r') as f:
                    headers = f.readline().strip().split(',')
                    for line in f:
                        values = line.strip().split(',')
                        data.append(dict(zip(headers, values)))
                return data
            elif file_ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                return data
    except Exception as e:
        console.print(f"[red]Erreur lors du chargement du fichier: {e}[/red]")
        return None
    
    console.print(f"[yellow]Format de fichier non pris en charge: {file_ext}[/yellow]")
    return None


def display_trades_stats(data):
    """
    Affiche les statistiques de trading
    
    Args:
        data: DataFrame ou liste de trades
    """
    if data is None:
        return
    
    # Extraire les statistiques clés
    if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        if 'action' in data.columns:
            buy_trades = data[data['action'] == 'BUY'].shape[0]
            sell_trades = data[data['action'] == 'SELL'].shape[0]
        elif 'type' in data.columns:
            buy_trades = data[data['type'] == 'BUY'].shape[0]
            sell_trades = data[data['type'] == 'SELL'].shape[0]
        else:
            buy_trades = sell_trades = 0
            
        total_trades = len(data)
        
        # PnL
        if 'pnl' in data.columns:
            total_pnl = data['pnl'].sum()
            avg_pnl = data[data['pnl'] != 0]['pnl'].mean() if any(data['pnl'] != 0) else 0
            win_trades = (data['pnl'] > 0).sum()
            loss_trades = (data['pnl'] < 0).sum()
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        elif 'gain' in data.columns:
            total_pnl = data['gain'].sum()
            avg_pnl = data[data['gain'] != 0]['gain'].mean() if any(data['gain'] != 0) else 0
            win_trades = (data['gain'] > 0).sum()
            loss_trades = (data['gain'] < 0).sum()
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            total_pnl = avg_pnl = win_trades = loss_trades = win_rate = 0
            
        # Analyser par palier
        if 'tier' in data.columns:
            tier_stats = data.groupby('tier').agg({
                'pnl' if 'pnl' in data.columns else 'gain': ['sum', 'mean', 'count']
            })
            has_tiers = True
        else:
            has_tiers = False
    else:
        # Version non-pandas
        total_trades = len(data)
        buy_trades = sum(1 for trade in data if trade.get('action', trade.get('type', '')) == 'BUY')
        sell_trades = sum(1 for trade in data if trade.get('action', trade.get('type', '')) == 'SELL')
        
        pnl_key = 'pnl' if 'pnl' in data[0] else 'gain' if 'gain' in data[0] else None
        if pnl_key:
            total_pnl = sum(float(trade.get(pnl_key, 0)) for trade in data)
            pnl_values = [float(trade.get(pnl_key, 0)) for trade in data if float(trade.get(pnl_key, 0)) != 0]
            avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
            win_trades = sum(1 for trade in data if float(trade.get(pnl_key, 0)) > 0)
            loss_trades = sum(1 for trade in data if float(trade.get(pnl_key, 0)) < 0)
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            total_pnl = avg_pnl = win_trades = loss_trades = win_rate = 0
            
        # Analyser par palier
        has_tiers = 'tier' in data[0] if data else False
        if has_tiers:
            tier_stats = defaultdict(lambda: {'sum': 0, 'count': 0, 'values': []})
            for trade in data:
                tier = trade.get('tier', 0)
                pnl = float(trade.get(pnl_key, 0)) if pnl_key else 0
                tier_stats[tier]['sum'] += pnl
                tier_stats[tier]['count'] += 1
                tier_stats[tier]['values'].append(pnl)
            
            # Calculer les moyennes
            for tier in tier_stats:
                values = tier_stats[tier]['values']
                tier_stats[tier]['mean'] = sum(values) / len(values) if values else 0
    
    # Afficher les statistiques
    if RICH_AVAILABLE:
        console.print("\n[bold green]═════════ RÉSUMÉ DES TRADES ═════════[/bold green]")
        
        summary = f"""
        **Statistiques générales:**
        * Nombre total de trades: {total_trades}
        * Achats: {buy_trades}, Ventes: {sell_trades}
        * PnL total: ${total_pnl:.2f}
        * PnL moyen par trade: ${avg_pnl:.2f}
        * Trades gagnants: {win_trades}, Trades perdants: {loss_trades}
        * Taux de réussite: {win_rate:.1f}%
        """
        console.print(Panel(Markdown(summary), title="Performance globale", border_style="green"))
        
        # Afficher les trades
        table = Table(title="Échantillon de trades")
        table.add_column("Index", justify="right")
        table.add_column("Timestamp", justify="left")
        if has_tiers:
            table.add_column("Tier", justify="center")
        table.add_column("Paire", justify="left")
        table.add_column("Action", justify="center")
        table.add_column("Prix", justify="right")
        table.add_column("Quantité", justify="right")
        table.add_column("Montant", justify="right")
        table.add_column("PnL", justify="right")
        
        # Limiter à 10 lignes pour l'affichage
        sample_size = min(10, len(data))
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            sample = data.sample(sample_size) if len(data) > sample_size else data
            for idx, row in sample.iterrows():
                pnl_style = "green" if row.get('pnl', row.get('gain', 0)) > 0 else "red" if row.get('pnl', row.get('gain', 0)) < 0 else ""
                action_style = "green" if row.get('action', row.get('type', '')) == "BUY" else "red" if row.get('action', row.get('type', '')) == "SELL" else ""
                
                row_data = [
                    str(idx),
                    str(row.get('timestamp', 'N/A')),
                ]
                if has_tiers:
                    row_data.append(str(row.get('tier', 'N/A')))
                row_data.extend([
                    str(row.get('pair', 'N/A')),
                    f"[{action_style}]{row.get('action', row.get('type', 'N/A'))}[/{action_style}]",
                    f"{row.get('price', 0):.4f}",
                    f"{row.get('qty', 0):.6f}",
                    f"{row.get('amount', row.get('montant', 0)):.2f}",
                    f"[{pnl_style}]{row.get('pnl', row.get('gain', 0)):.2f}[/{pnl_style}]"
                ])
                table.add_row(*row_data)
        else:
            import random
            sample = random.sample(data, sample_size) if len(data) > sample_size else data
            for i, trade in enumerate(sample):
                pnl_value = float(trade.get('pnl', trade.get('gain', 0)))
                pnl_style = "green" if pnl_value > 0 else "red" if pnl_value < 0 else ""
                action = trade.get('action', trade.get('type', 'N/A'))
                action_style = "green" if action == "BUY" else "red" if action == "SELL" else ""
                
                row_data = [
                    str(i),
                    str(trade.get('timestamp', 'N/A')),
                ]
                if has_tiers:
                    row_data.append(str(trade.get('tier', 'N/A')))
                row_data.extend([
                    str(trade.get('pair', 'N/A')),
                    f"[{action_style}]{action}[/{action_style}]",
                    f"{float(trade.get('price', 0)):.4f}",
                    f"{float(trade.get('qty', 0)):.6f}",
                    f"{float(trade.get('amount', trade.get('montant', 0))):.2f}",
                    f"[{pnl_style}]{pnl_value:.2f}[/{pnl_style}]"
                ])
                table.add_row(*row_data)
        
        console.print(table)
        
        # Afficher les statistiques par palier si disponibles
        if has_tiers:
            console.print("\n[bold green]═════════ STATISTIQUES PAR PALIER ═════════[/bold green]")
            tier_table = Table(title="Performance par palier")
            tier_table.add_column("Palier", justify="center")
            tier_table.add_column("# Trades", justify="right")
            tier_table.add_column("PnL Total", justify="right")
            tier_table.add_column("PnL Moyen", justify="right")
            
            if PANDAS_AVAILABLE:
                for tier, row in tier_stats.iterrows():
                    if isinstance(row, pd.Series):
                        pnl_sum = row[('pnl' if 'pnl' in data.columns else 'gain', 'sum')]
                        pnl_style = "green" if pnl_sum > 0 else "red" if pnl_sum < 0 else ""
                        tier_table.add_row(
                            str(tier),
                            str(int(row[('pnl' if 'pnl' in data.columns else 'gain', 'count')])),
                            f"[{pnl_style}]{pnl_sum:.2f}[/{pnl_style}]",
                            f"{row[('pnl' if 'pnl' in data.columns else 'gain', 'mean')]:.2f}"
                        )
            else:
                for tier, stats in tier_stats.items():
                    pnl_sum = stats['sum']
                    pnl_style = "green" if pnl_sum > 0 else "red" if pnl_sum < 0 else ""
                    tier_table.add_row(
                        str(tier),
                        str(stats['count']),
                        f"[{pnl_style}]{pnl_sum:.2f}[/{pnl_style}]",
                        f"{stats['mean']:.2f}"
                    )
            
            console.print(tier_table)
    else:
        # Affichage en mode texte simple
        print("\n========= RÉSUMÉ DES TRADES =========")
        print(f"Nombre total de trades: {total_trades}")
        print(f"Achats: {buy_trades}, Ventes: {sell_trades}")
        print(f"PnL total: ${total_pnl:.2f}")
        print(f"PnL moyen par trade: ${avg_pnl:.2f}")
        print(f"Trades gagnants: {win_trades}, Trades perdants: {loss_trades}")
        print(f"Taux de réussite: {win_rate:.1f}%")
        
        # Afficher les statistiques par palier si disponibles
        if has_tiers:
            print("\n========= STATISTIQUES PAR PALIER =========")
            if PANDAS_AVAILABLE:
                for tier, row in tier_stats.iterrows():
                    pnl_sum = row[('pnl' if 'pnl' in data.columns else 'gain', 'sum')]
                    print(f"Palier {tier}: {int(row[('pnl' if 'pnl' in data.columns else 'gain', 'count')])} trades, "
                          f"PnL total: ${pnl_sum:.2f}, PnL moyen: ${row[('pnl' if 'pnl' in data.columns else 'gain', 'mean')]:.2f}")
            else:
                for tier, stats in tier_stats.items():
                    print(f"Palier {tier}: {stats['count']} trades, PnL total: ${stats['sum']:.2f}, "
                          f"PnL moyen: ${stats['mean']:.2f}")


def display_portfolio_stats(data):
    """
    Affiche les statistiques du portefeuille
    
    Args:
        data: DataFrame ou liste d'états du portefeuille
    """
    if data is None:
        return
    
    # Extraire les statistiques clés
    if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        # Calculer les statistiques globales
        if 'portfolio_value' in data.columns or 'capital' in data.columns:
            key = 'portfolio_value' if 'portfolio_value' in data.columns else 'capital'
            initial_value = data[key].iloc[0] if not data.empty else 0
            final_value = data[key].iloc[-1] if not data.empty else 0
            min_value = data[key].min()
            max_value = data[key].max()
            
            # Calculer le rendement global
            total_return = ((final_value / initial_value) - 1) * 100 if initial_value > 0 else 0
            
            # Calculer le drawdown maximal
            data['peak'] = data[key].cummax()
            data['drawdown'] = (data[key] - data['peak']) / data['peak'] * 100
            max_drawdown = data['drawdown'].min()
            
            # Calculer le Sharpe approximatif (si reward est disponible)
            if 'reward' in data.columns and len(data) > 1:
                returns = data['reward']
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe = None
                
            # Analyser par palier
            if 'tier' in data.columns:
                tier_stats = data.groupby('tier').agg({
                    key: ['mean', 'min', 'max', 'count'],
                    'reward': ['sum', 'mean'] if 'reward' in data.columns else []
                })
                has_tiers = True
            else:
                has_tiers = False
        else:
            initial_value = final_value = min_value = max_value = total_return = max_drawdown = 0
            sharpe = None
            has_tiers = False
    else:
        # Version non-pandas
        if not data:
            initial_value = final_value = min_value = max_value = total_return = max_drawdown = 0
            sharpe = None
            has_tiers = False
        else:
            key = 'portfolio_value' if 'portfolio_value' in data[0] else 'capital' if 'capital' in data[0] else None
            if key:
                values = [float(item.get(key, 0)) for item in data]
                initial_value = values[0] if values else 0
                final_value = values[-1] if values else 0
                min_value = min(values) if values else 0
                max_value = max(values) if values else 0
                
                # Calculer le rendement global
                total_return = ((final_value / initial_value) - 1) * 100 if initial_value > 0 else 0
                
                # Calculer le drawdown maximal
                peak = 0
                max_drawdown = 0
                for val in values:
                    peak = max(peak, val)
                    drawdown = (val - peak) / peak * 100 if peak > 0 else 0
                    max_drawdown = min(max_drawdown, drawdown)
                
                # Calculer le Sharpe approximatif (si reward est disponible)
                if 'reward' in data[0]:
                    from statistics import mean, stdev
                    rewards = [float(item.get('reward', 0)) for item in data]
                    sharpe = mean(rewards) / stdev(rewards) if len(rewards) > 1 and stdev(rewards) > 0 else 0
                else:
                    sharpe = None
                    
                # Analyser par palier
                has_tiers = 'tier' in data[0]
                if has_tiers:
                    tier_stats = defaultdict(lambda: {'values': [], 'rewards': []})
                    for item in data:
                        tier = item.get('tier', 0)
                        tier_stats[tier]['values'].append(float(item.get(key, 0)))
                        if 'reward' in item:
                            tier_stats[tier]['rewards'].append(float(item.get('reward', 0)))
                    
                    # Calculer les statistiques par palier
                    for tier in tier_stats:
                        values = tier_stats[tier]['values']
                        tier_stats[tier]['mean'] = sum(values) / len(values) if values else 0
                        tier_stats[tier]['min'] = min(values) if values else 0
                        tier_stats[tier]['max'] = max(values) if values else 0
                        tier_stats[tier]['count'] = len(values)
                        
                        rewards = tier_stats[tier]['rewards']
                        if rewards:
                            tier_stats[tier]['reward_sum'] = sum(rewards)
                            tier_stats[tier]['reward_mean'] = sum(rewards) / len(rewards)
            else:
                initial_value = final_value = min_value = max_value = total_return = max_drawdown = 0
                sharpe = None
                has_tiers = False
    
    # Afficher les statistiques
    if RICH_AVAILABLE:
        console.print("\n[bold green]═════════ RÉSUMÉ DU PORTEFEUILLE ═════════[/bold green]")
        
        summary = f"""
        **Statistiques globales:**
        * Valeur initiale: ${initial_value:.2f}
        * Valeur finale: ${final_value:.2f}
        * Rendement total: {total_return:.2f}%
        * Valeur minimale: ${min_value:.2f}
        * Valeur maximale: ${max_value:.2f}
        * Drawdown maximal: {max_drawdown:.2f}%
        """
        if sharpe is not None:
            summary += f"* Sharpe Ratio approximatif: {sharpe:.2f}\n"
            
        console.print(Panel(Markdown(summary), title="Performance du portefeuille", border_style="blue"))
        
        # Afficher l'évolution du portefeuille
        if MATPLOTLIB_AVAILABLE and PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(data['step'] if 'step' in data.columns else range(len(data)), 
                         data[key], label='Valeur du portefeuille', color='blue')
                plt.title('Évolution de la valeur du portefeuille')
                plt.xlabel('Étape')
                plt.ylabel('Valeur (USDT)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                console.print(f"[yellow]Erreur lors de la génération du graphique: {e}[/yellow]")
        
        # Afficher les statistiques par palier si disponibles
        if has_tiers:
            console.print("\n[bold green]═════════ STATISTIQUES PAR PALIER ═════════[/bold green]")
            tier_table = Table(title="Performance par palier")
            tier_table.add_column("Palier", justify="center")
            tier_table.add_column("# Étapes", justify="right")
            tier_table.add_column("Valeur Moyenne", justify="right")
            tier_table.add_column("Valeur Min", justify="right")
            tier_table.add_column("Valeur Max", justify="right")
            if PANDAS_AVAILABLE and 'reward' in data.columns:
                tier_table.add_column("Reward Total", justify="right")
                tier_table.add_column("Reward Moyen", justify="right")
            
            if PANDAS_AVAILABLE:
                for tier, row in tier_stats.iterrows():
                    if isinstance(row, pd.Series):
                        row_data = [
                            str(tier),
                            str(int(row[(key, 'count')])),
                            f"{row[(key, 'mean')]:.2f}",
                            f"{row[(key, 'min')]:.2f}",
                            f"{row[(key, 'max')]:.2f}",
                        ]
                        if 'reward' in data.columns:
                            row_data.extend([
                                f"{row[('reward', 'sum')]:.2f}",
                                f"{row[('reward', 'mean')]:.4f}"
                            ])
                        tier_table.add_row(*row_data)
            else:
                for tier, stats in tier_stats.items():
                    row_data = [
                        str(tier),
                        str(stats['count']),
                        f"{stats['mean']:.2f}",
                        f"{stats['min']:.2f}",
                        f"{stats['max']:.2f}",
                    ]
                    if 'reward_sum' in stats:
                        row_data.extend([
                            f"{stats['reward_sum']:.2f}",
                            f"{stats['reward_mean']:.4f}"
                        ])
                    tier_table.add_row(*row_data)
            
            console.print(tier_table)
    else:
        # Affichage en mode texte simple
        print("\n========= RÉSUMÉ DU PORTEFEUILLE =========")
        print(f"Valeur initiale: ${initial_value:.2f}")
        print(f"Valeur finale: ${final_value:.2f}")
        print(f"Rendement total: {total_return:.2f}%")
        print(f"Valeur minimale: ${min_value:.2f}")
        print(f"Valeur maximale: ${max_value:.2f}")
        print(f"Drawdown maximal: {max_drawdown:.2f}%")
        if sharpe is not None:
            print(f"Sharpe Ratio approximatif: {sharpe:.2f}")
        
        # Afficher les statistiques par palier si disponibles
        if has_tiers:
            print("\n========= STATISTIQUES PAR PALIER =========")
            if PANDAS_AVAILABLE:
                for tier, row in tier_stats.iterrows():
                    print(f"Palier {tier}: {int(row[(key, 'count')])} étapes, "
                          f"Valeur moyenne: ${row[(key, 'mean')]:.2f}, "
                          f"Min: ${row[(key, 'min')]:.2f}, Max: ${row[(key, 'max')]:.2f}")
                    if 'reward' in data.columns:
                        print(f"  Reward total: {row[('reward', 'sum')]:.2f}, "
                              f"Reward moyen: {row[('reward', 'mean')]:.4f}")
            else:
                for tier, stats in tier_stats.items():
                    print(f"Palier {tier}: {stats['count']} étapes, "
                          f"Valeur moyenne: ${stats['mean']:.2f}, "
                          f"Min: ${stats['min']:.2f}, Max: ${stats['max']:.2f}")
                    if 'reward_sum' in stats:
                        print(f"  Reward total: {stats['reward_sum']:.2f}, "
                              f"Reward moyen: {stats['reward_mean']:.4f}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Visualiseur de statistiques de trading simplifié",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Chemin vers le fichier de données')
    parser.add_argument('--type', '-t', type=str, choices=['trades', 'portfolio', 'tier'],
                       default='trades', help='Type de données')
    
    args = parser.parse_args()
    
    # Vérifier si le fichier existe
    if not os.path.exists(args.file):
        console.print(f"[red]Erreur: Le fichier {args.file} n'existe pas[/red]")
        return 1
    
    # Charger les données
    data = load_data(args.file, args.type)
    if data is None:
        return 1
    
    # Afficher les statistiques selon le type
    if args.type == 'trades':
        display_trades_stats(data)
    elif args.type == 'portfolio':
        display_portfolio_stats(data)
    elif args.type == 'tier':
        # Afficher les statistiques par palier (même logique que portfolio)
        display_portfolio_stats(data)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())