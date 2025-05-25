import os
# Configurer l'environnement pour la compatibilité avec TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Réduire les logs TF
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Pour compatibilité avec les modèles Keras

import numpy as np
import logging
import yaml
import argparse
from pathlib import Path
import pandas as pd
import time
import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Import conditionnel de matplotlib pour les visualisations
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("AVERTISSEMENT: matplotlib n'est pas installé. Les visualisations graphiques ne seront pas disponibles.")

# Import conditionnel des dépendances de Stable Baselines 3
try:
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.callbacks import CallbackList
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("AVERTISSEMENT: Stable Baselines 3 ou PyTorch n'est pas installé. Certaines fonctionnalités seront limitées.")
    # Classes factices pour éviter les erreurs
    class BaseCallback:
        def __init__(self, verbose=0): pass

from rl_environment.multi_asset_env import MultiAssetEnv

# Initialiser Rich Console pour l'affichage amélioré
console = Console()

# Callback pour gérer les NaN et Inf
class NanInfFixCallback(BaseCallback):
    """
    Callback pour détecter et corriger les valeurs NaN et Inf dans les paramètres du modèle.
    """
    def __init__(self, verbose=0):
        super(NanInfFixCallback, self).__init__(verbose)
        self.nan_encountered = 0
        self.inf_encountered = 0

    def _on_step(self):
        # Vérifier si PyTorch est disponible
        if not SB3_AVAILABLE:
            return True
            
        # Vérifier et corriger les NaN dans les gradients
        for param in self.model.policy.parameters():
            # Vérifier si le gradient existe et contient des NaN
            if param.grad is not None:
                nan_mask = th.isnan(param.grad)
                inf_mask = th.isinf(param.grad)
                
                if nan_mask.any():
                    self.nan_encountered += nan_mask.sum().item()
                    param.grad[nan_mask] = 0.0
                    
                if inf_mask.any():
                    self.inf_encountered += inf_mask.sum().item()
                    param.grad[inf_mask] = 0.0
                    
        # Vérifier et corriger les NaN dans les paramètres
        for param in self.model.policy.parameters():
            nan_mask = th.isnan(param.data)
            inf_mask = th.isinf(param.data)
            
            if nan_mask.any():
                self.nan_encountered += nan_mask.sum().item()
                param.data[nan_mask] = 0.0
                
            if inf_mask.any():
                self.inf_encountered += inf_mask.sum().item()
                param.data[inf_mask] = 0.0

        # Log uniquement si des NaN ou Inf ont été trouvés et corrigés
        if self.nan_encountered > 0 or self.inf_encountered > 0:
            self.logger.record("debug/nan_encountered", self.nan_encountered)
            self.logger.record("debug/inf_encountered", self.inf_encountered)
            if self.verbose > 0:
                print(f"Corrigé {self.nan_encountered} NaN et {self.inf_encountered} Inf")
            # Réinitialiser les compteurs pour le prochain pas
            self.nan_encountered = 0
            self.inf_encountered = 0
            
        return True

# Callback pour collecter l'historique détaillé des trades
class TradeHistoryCallback(BaseCallback):
    """
    Callback pour collecter l'historique des trades et calculer des statistiques.
    """
    def __init__(self, verbose=0, log_history=False):
        super(TradeHistoryCallback, self).__init__(verbose)
        self.log_history = log_history
        self.start_time = time.time()
        self.episode_rewards = []
        self.current_rewards = 0
        self.trade_history = []
        self.tier_stats = {}  # Statistiques par palier
        self.trades_df = None  # DataFrame de trades pour l'analyse
        self.history_df = None  # DataFrame d'historique pour l'analyse
        self.tier_df = None  # DataFrame de statistiques par palier
        self.action_map = {0: "MARKET", 1: "LIMIT", 2: "STOP", 3: "TAKE_PROFIT", 4: "TRAILING_STOP"}
        
    def _on_step(self):
        # Si nous n'enregistrons pas l'historique, retourner immédiatement
        if not self.log_history:
            return True
            
        # Récupérer l'environnement (nous prenons le premier dans le cas d'environnements vectorisés)
        env = self.training_env.envs[0]
        
        # Collecter les données actuelles si disponibles dans l'historique de l'environnement
        if hasattr(env, 'history') and len(env.history) > 0:
            latest_entry = env.history[-1]
            
            # Ajouter le temps écoulé
            elapsed_time = time.time() - self.start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            
            # Récupérer et formater les données
            step_data = {
                'step': self.num_timesteps,
                'timestamp': latest_entry.get('timestamp', 'N/A'),
                'tier': env._current_tier() if hasattr(env, '_current_tier') else 0,
                'cash': env.cash,
                'portfolio_value': latest_entry.get('portfolio_value', 0),
                'cum_reward': self.current_rewards,
                'time_elapsed': elapsed_str
            }
            
            # Ajouter des informations pour chaque paire d'actifs
            for i, pair in enumerate(env.pairs):
                # Récupérer l'action de façon plus sûre
                actions = self.locals.get('actions', [[0]])
                action_idx = 0  # Par défaut MARKET
                if actions and len(actions) > 0:
                    # S'assurer que nous accédons correctement aux actions
                    if isinstance(actions[0], (list, tuple, np.ndarray)) and len(actions[0]) > 1:
                        action_idx = actions[0][1] if i == 0 else 0
                step_data[f'action_{pair}'] = self.action_map.get(action_idx, "HOLD")
                step_data[f'price_{pair}'] = latest_entry.get(f'price_{pair}', 0)
                step_data[f'qty_{pair}'] = latest_entry.get(f'qty_{pair}', 0)
                
            # Ajouter les frais et PnL si disponibles
            step_data['fee'] = latest_entry.get('fee', 0)
            step_data['pnl'] = latest_entry.get('pnl', 0)
            
            # Ajouter l'entrée à notre historique
            self.trade_history.append(step_data)
            
            # Mettre à jour les statistiques par palier
            tier = step_data['tier']
            if tier not in self.tier_stats:
                self.tier_stats[tier] = {
                    'trades': 0,
                    'buys': 0,
                    'sells': 0,
                    'total_gains': 0,
                    'total_losses': 0,
                    'pnl_trades': []
                }
            
            # Incrémenter les compteurs de trades si une action a été prise
            if step_data.get('pnl', 0) != 0:
                self.tier_stats[tier]['trades'] += 1
                if step_data.get('pnl', 0) > 0:
                    self.tier_stats[tier]['total_gains'] += step_data.get('pnl', 0)
                else:
                    self.tier_stats[tier]['total_losses'] += abs(step_data.get('pnl', 0))
                
                # Compter les achats/ventes
                for pair in env.pairs:
                    action = step_data.get(f'action_{pair}', "HOLD")
                    if action == "BUY":
                        self.tier_stats[tier]['buys'] += 1
                    elif action == "SELL":
                        self.tier_stats[tier]['sells'] += 1
                
                # Enregistrer le PnL pour les statistiques
                self.tier_stats[tier]['pnl_trades'].append(step_data.get('pnl', 0))
            
        # Mettre à jour la récompense cumulée de façon sûre
        rewards = self.locals.get('rewards', None)
        if rewards is not None and isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) > 0:
            self.current_rewards += rewards[0]
            
        # Réinitialiser le compteur de récompenses à la fin d'un épisode de façon sûre
        dones = self.locals.get('dones', None)
        if dones is not None and isinstance(dones, (list, tuple, np.ndarray)) and len(dones) > 0 and dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
            
        return True
        
    def prepare_dataframes(self, env):
            """Prépare les DataFrames à partir des données collectées"""
            if hasattr(env, 'trade_log') and len(env.trade_log) > 0:
                self.trades_df = pd.DataFrame(env.trade_log)
                # Convertir les timestamps en datetime si ce n'est pas déjà fait
                if 'timestamp' in self.trades_df.columns and not pd.api.types.is_datetime64_any_dtype(self.trades_df['timestamp']):
                    self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            
                # Calculer des métriques additionnelles pour l'analyse
                if 'pnl' in self.trades_df.columns:
                    self.trades_df['cumulative_pnl'] = self.trades_df['pnl'].cumsum()
                    # Ajouter une colonne pour le résultat du trade (Gain/Perte/Neutre)
                    self.trades_df['result'] = self.trades_df['pnl'].apply(
                        lambda x: 'Gain' if x > 0 else 'Perte' if x < 0 else 'Neutre')
        
            if hasattr(env, 'history') and len(env.history) > 0:
                self.history_df = pd.DataFrame(env.history)
                # Convertir les timestamps en datetime si ce n'est pas déjà fait
                if 'timestamp' in self.history_df.columns and not pd.api.types.is_datetime64_any_dtype(self.history_df['timestamp']):
                    self.history_df['timestamp'] = pd.to_datetime(self.history_df['timestamp'])
            
                # Calculer des métriques additionnelles pour l'analyse
                if 'portfolio_value' in self.history_df.columns:
                    # Calculer le rendement étape par étape
                    self.history_df['return'] = self.history_df['portfolio_value'].pct_change()
                    # Calculer le drawdown
                    self.history_df['peak'] = self.history_df['portfolio_value'].cummax()
                    self.history_df['drawdown'] = (self.history_df['portfolio_value'] - self.history_df['peak']) / self.history_df['peak'] * 100
        
            # Créer le DataFrame des statistiques par palier
            if self.history_df is not None and 'tier' in self.history_df.columns:
                # Analyse plus détaillée par palier
                metrics = {
                    'portfolio_value': ['mean', 'min', 'max', 'std'],
                    'reward': ['sum', 'mean', 'count', 'std'],
                    'pnl': ['sum', 'mean']
                }
                # Ne garder que les colonnes qui existent
                metrics = {k: v for k, v in metrics.items() if k in self.history_df.columns}
            
                self.tier_df = self.history_df.groupby('tier').agg(metrics).reset_index()
                # Aplatir les colonnes multi-index
                if isinstance(self.tier_df.columns, pd.MultiIndex):
                    self.tier_df.columns = ['_'.join(col).strip('_') for col in self.tier_df.columns.values]
    
    def display_step_history(self):
        """Affiche un tableau avec l'historique des étapes"""
        if self.trades_df is None or self.trades_df.empty:
            console.print("[yellow]Aucun historique de trades disponible[/yellow]")
            return
            
        # Limiter le nombre de lignes affichées si trop grand
        max_rows = 20
        df = self.trades_df.copy()
        if len(df) > max_rows:
            console.print(f"[yellow]Affichage des {max_rows} dernières transactions sur {len(df)} au total[/yellow]")
            df = df.tail(max_rows)
        
        # Créer un tableau Rich pour l'affichage
        table = Table(title="Journal des transactions de trading")
        
        # Ajouter les colonnes principales
        table.add_column("Step", justify="right")
        table.add_column("Timestamp", justify="left")
        table.add_column("Tier", justify="right")
        table.add_column("Pair", justify="left")
        table.add_column("Action", justify="center", style="bold")
        table.add_column("Prix", justify="right")
        table.add_column("Quantité", justify="right")
        table.add_column("Montant ($)", justify="right")
        table.add_column("Frais", justify="right")
        table.add_column("Cash Avant", justify="right")
        table.add_column("Cash Après", justify="right")
        table.add_column("PnL", justify="right")
        
        # Ajouter les lignes
        for _, row in df.iterrows():
            pnl_style = "green" if row.get('pnl', 0) > 0 else "red" if row.get('pnl', 0) < 0 else ""
            action_style = "green" if row.get('action', "") == "BUY" else "red" if row.get('action', "") == "SELL" else ""
            
            table.add_row(
                str(row.get('step', 'N/A')),
                str(row.get('timestamp', 'N/A')),
                str(row.get('tier', 'N/A')),
                str(row.get('pair', 'N/A')),
                f"[{action_style}]{row.get('action', 'N/A')}[/{action_style}]",
                f"{row.get('price', 0):.4f}",
                f"{row.get('qty', 0):.6f}",
                f"{row.get('amount', 0):.2f}",
                f"{row.get('fee', 0):.2f}",
                f"{row.get('cash_before', 0):.2f}",
                f"{row.get('cash_after', 0):.2f}",
                f"[{pnl_style}]{row.get('pnl', 0):.2f}[/{pnl_style}]"
            )
            
        # Afficher le tableau
        console.print(table)
        
        # Afficher également un résumé des performances de trading
        if len(self.trades_df) > 0:
            buy_trades = self.trades_df[self.trades_df['action'] == 'BUY'].shape[0]
            sell_trades = self.trades_df[self.trades_df['action'] == 'SELL'].shape[0]
            total_pnl = self.trades_df['pnl'].sum()
            avg_pnl = self.trades_df[self.trades_df['pnl'] != 0]['pnl'].mean() if any(self.trades_df['pnl'] != 0) else 0
            win_rate = (self.trades_df['pnl'] > 0).mean() * 100 if not self.trades_df.empty else 0
            
            summary = f"""
            **Résumé des transactions:**
            * Nombre total de trades: {len(self.trades_df)}
            * Achats: {buy_trades}, Ventes: {sell_trades}
            * PnL total: ${total_pnl:.2f}
            * PnL moyen par trade: ${avg_pnl:.2f}
            * Taux de réussite: {win_rate:.1f}%
            """
            console.print(Panel(Markdown(summary), title="Résumé des performances", border_style="green"))
        
    def display_tier_statistics(self):
        """Affiche un résumé des statistiques par palier"""
        if self.tier_df is None or self.tier_df.empty:
            console.print("[yellow]Aucune statistique par palier disponible[/yellow]")
            return
            
        # Créer un tableau Rich pour l'affichage
        table = Table(title="Résumé par palier")
        
        # Ajouter les colonnes
        table.add_column("Tier", justify="right")
        table.add_column("# Steps", justify="right")
        table.add_column("# Trades", justify="right")
        table.add_column("# Buys", justify="right")
        table.add_column("# Sells", justify="right")
        table.add_column("Total PnL ($)", justify="right")
        table.add_column("Avg Portfolio", justify="right")
        table.add_column("Min Portfolio", justify="right")
        table.add_column("Max Portfolio", justify="right")
        table.add_column("Avg Reward", justify="right")
        
        # Calculer le nombre de trades par palier à partir de trades_df
        trades_by_tier = {}
        buys_by_tier = {}
        sells_by_tier = {}
        
        if self.trades_df is not None and not self.trades_df.empty and 'tier' in self.trades_df.columns:
            tier_groups = self.trades_df.groupby('tier')
            for tier, group in tier_groups:
                trades_by_tier[tier] = len(group)
                buys_by_tier[tier] = len(group[group['action'] == 'BUY'])
                sells_by_tier[tier] = len(group[group['action'] == 'SELL'])
        
        # Ajouter une ligne pour chaque palier
        for _, row in self.tier_df.iterrows():
            tier = row['tier']
            tier_style = ""
            
            # Déterminer le style en fonction du PnL
            if row['pnl_sum'] > 0:
                tier_style = "green"
            elif row['pnl_sum'] < 0:
                tier_style = "red"
            
            table.add_row(
                str(int(tier)),
                str(int(row['steps_count'])),
                str(trades_by_tier.get(tier, 0)),
                str(buys_by_tier.get(tier, 0)),
                str(sells_by_tier.get(tier, 0)),
                f"[{tier_style}]{row['pnl_sum']:.2f}[/{tier_style}]",
                f"{row['portfolio_mean']:.2f}",
                f"{row['portfolio_min']:.2f}",
                f"{row['portfolio_max']:.2f}",
                f"{row['reward_mean']:.4f}"
            )
            
        # Afficher le tableau
        console.print(table)
        
    def plot_portfolio_evolution(self, save_path=None):
        """Trace l'évolution de la valeur du portefeuille au fil du temps"""
        if not MATPLOTLIB_AVAILABLE:
            console.print("[yellow]matplotlib n'est pas installé. Impossible de générer des visualisations graphiques.[/yellow]")
            console.print("[green]Installer matplotlib avec: pip install matplotlib[/green]")
            return
            
        if self.history_df is None or self.history_df.empty:
            console.print("[yellow]Aucune donnée d'historique disponible pour générer le graphique[/yellow]")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.history_df['step'], self.history_df['portfolio_value'], label='Valeur du portefeuille', color='blue', linewidth=2)
        
        # Marquer les trades sur le graphique
        if self.trades_df is not None and not self.trades_df.empty:
            buy_points = self.trades_df[self.trades_df['action'] == 'BUY']
            sell_points = self.trades_df[self.trades_df['action'] == 'SELL']
            
            # Récupérer les valeurs du portefeuille pour les points de trade
            if not buy_points.empty:
                buy_portfolio_values = []
                for step in buy_points['step']:
                    matching_history = self.history_df[self.history_df['step'] == step]
                    if not matching_history.empty:
                        buy_portfolio_values.append(matching_history.iloc[0]['portfolio_value'])
                    else:
                        buy_portfolio_values.append(None)
                buy_points = buy_points.assign(portfolio_value=buy_portfolio_values)
                buy_points = buy_points.dropna(subset=['portfolio_value'])
                plt.scatter(buy_points['step'], buy_points['portfolio_value'], color='green', marker='^', label='Achat', s=50)
            
            if not sell_points.empty:
                sell_portfolio_values = []
                for step in sell_points['step']:
                    matching_history = self.history_df[self.history_df['step'] == step]
                    if not matching_history.empty:
                        sell_portfolio_values.append(matching_history.iloc[0]['portfolio_value'])
                    else:
                        sell_portfolio_values.append(None)
                sell_points = sell_points.assign(portfolio_value=sell_portfolio_values)
                sell_points = sell_points.dropna(subset=['portfolio_value'])
                plt.scatter(sell_points['step'], sell_points['portfolio_value'], color='red', marker='v', label='Vente', s=50)
        
        # Ajouter les annotations pour les paliers
        if 'tier' in self.history_df.columns:
            tier_changes = self.history_df[self.history_df['tier'].diff() != 0]
            for idx, row in tier_changes.iterrows():
                if idx > 0:  # Ignorer le premier point (pas de changement réel)
                    plt.axvline(x=row['step'], color='gray', linestyle='--', alpha=0.5)
                    plt.text(row['step'], plt.ylim()[1] * 0.95, f"Palier {int(row['tier'])}", 
                             rotation=90, verticalalignment='top', horizontalalignment='right')
        
        plt.title('Évolution de la valeur du portefeuille')
        plt.xlabel('Étape')
        plt.ylabel('Valeur (USDT)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]Graphique sauvegardé: {save_path}[/green]")
        else:
            plt.show()
        
    def plot_reward_evolution(self, save_path=None):
        """Trace l'évolution du reward cumulé au fil du temps"""
        if not MATPLOTLIB_AVAILABLE:
            console.print("[yellow]matplotlib n'est pas installé. Impossible de générer des visualisations graphiques.[/yellow]")
            console.print("[green]Installer matplotlib avec: pip install matplotlib[/green]")
            return
            
        if self.history_df is None or self.history_df.empty or 'cum_reward' not in self.history_df.columns:
            console.print("[yellow]Aucune donnée de reward disponible pour générer le graphique[/yellow]")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.history_df['step'], self.history_df['cum_reward'], label='Reward cumulé', color='purple', linewidth=2)
        
        # Ajouter les annotations pour les paliers
        if 'tier' in self.history_df.columns:
            tier_changes = self.history_df[self.history_df['tier'].diff() != 0]
            for idx, row in tier_changes.iterrows():
                if idx > 0:  # Ignorer le premier point (pas de changement réel)
                    plt.axvline(x=row['step'], color='gray', linestyle='--', alpha=0.5)
        
        plt.title('Évolution du reward cumulé')
        plt.xlabel('Étape')
        plt.ylabel('Reward cumulé')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]Graphique sauvegardé: {save_path}[/green]")
        else:
            plt.show()
            
    def plot_tier_statistics(self, save_path=None):
        """Trace les statistiques par palier"""
        if not MATPLOTLIB_AVAILABLE:
            console.print("[yellow]matplotlib n'est pas installé. Impossible de générer des visualisations graphiques.[/yellow]")
            console.print("[green]Installer matplotlib avec: pip install matplotlib[/green]")
            return
            
        if self.tier_df is None or self.tier_df.empty:
            console.print("[yellow]Aucune statistique par palier disponible pour générer le graphique[/yellow]")
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: PnL par palier
        tiers = self.tier_df['tier'].astype(int)
        pnl_values = self.tier_df['pnl_sum']
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        axs[0, 0].bar(tiers, pnl_values, color=colors)
        axs[0, 0].set_title('PnL total par palier')
        axs[0, 0].set_xlabel('Palier')
        axs[0, 0].set_ylabel('PnL ($)')
        axs[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Nombre de steps par palier
        axs[0, 1].bar(tiers, self.tier_df['steps_count'], color='blue', alpha=0.7)
        axs[0, 1].set_title('Nombre d\'étapes par palier')
        axs[0, 1].set_xlabel('Palier')
        axs[0, 1].set_ylabel('Nombre d\'étapes')
        axs[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Valeur moyenne du portefeuille par palier
        axs[1, 0].bar(tiers, self.tier_df['portfolio_mean'], color='purple', alpha=0.7)
        axs[1, 0].set_title('Valeur moyenne du portefeuille par palier')
        axs[1, 0].set_xlabel('Palier')
        axs[1, 0].set_ylabel('Valeur moyenne ($)')
        axs[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Reward moyen par palier
        axs[1, 1].bar(tiers, self.tier_df['reward_mean'], color='orange', alpha=0.7)
        axs[1, 1].set_title('Reward moyen par palier')
        axs[1, 1].set_xlabel('Palier')
        axs[1, 1].set_ylabel('Reward moyen')
        axs[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[green]Graphique des statistiques par palier sauvegardé: {save_path}[/green]")
        else:
            plt.show()
# Traitement des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Entraînement de l'agent RL avec PPO", 
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--total-timesteps', type=int, help='Nombre total de timesteps pour l\'entraînement')
parser.add_argument('--n-envs', type=int, default=1, help='Nombre d\'environnements parallèles')
parser.add_argument('--skip-encoder', action='store_true', help='Désactiver l\'encodeur et utiliser les features brutes')
parser.add_argument('--log-history', action='store_true', help='Activer la collecte et l\'affichage des statistiques détaillées')
parser.add_argument('--evaluate', action='store_true', help='Évaluer le modèle après l\'entraînement')
parser.add_argument('--skip-training', action='store_true', help='Ignorer l\'entraînement et utiliser uniquement les fonctionnalités de collecte de données')
parser.add_argument('--export-charts', action='store_true', help='Exporter les graphiques en PNG')
parser.add_argument('--export-csv', action='store_true', help='Exporter les données en CSV')
parser.add_argument('--export-all', action='store_true', help='Exporter toutes les données et graphiques')
parser.add_argument('--export-dir', type=str, help='Répertoire personnalisé pour l\'export des données')

# Ajouter une description détaillée du fonctionnement
parser.epilog = """
FORMULES ET MÉCANISMES UTILISÉS:

1. Rendement (log-return):
   r_t = ln(V_t / (V_t-1 + ε))
   
2. Reward shaping:
   reward = r_t × REWARD_MULTIPLIER  si r_t ≥ 0
   reward = r_t × PENALTY_MULTIPLIER si r_t < 0
   
   où les multiplicateurs dépendent du palier (tier) actuel.

3. PnL d'un trade:
   - BUY:  Δposition = +montant_usdt/prix
   - SELL: pnl = (qty_sold×price) - fee - (qty_sold×entry_price)

4. Contraintes sur les ordres:
   - Montant minimal + frais ≥ 10 $
   - Nombre max de positions = fonction du palier (cash)
   - Pas de short, pas de levier

TYPES D'ORDRES SUPPORTÉS:

| Type                       | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| Market                     | Exécution immédiate au meilleur prix du carnet               |
| Limit                      | Exécution à un prix déterminé, placé dans le carnet d'ordres |
| Stop                       | Déclenche un Market order au stop price spécifié             |
| Take-Profit                | Vente automatique à un prix cible spécifié                   |
| Trailing-Stop              | Stop price qui suit le marché à un pourcentage fixe          |
"""

args = parser.parse_args()

# Charger la configuration
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis : {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {config_path}: {e}")
        return None

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_rl_agent")

# Chemin vers la racine du projet et le fichier de configuration
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
config_path = script_dir / "config.yaml"

# Charger la configuration
config = load_config(config_path)
if not config:
    logger.critical("Impossible de charger la configuration. Arrêt du script.")
    exit(1)

# Récupérer les paramètres d'entraînement RL depuis la configuration
total_timesteps = args.total_timesteps if args.total_timesteps else config["rl_training"]["total_timesteps"]
batch_size = config["rl_training"]["batch_size"]
# Réduit le learning rate pour plus de stabilité
learning_rate = 1e-4  # Réduit depuis 1e-3 pour améliorer la stabilité
log_interval_sb3 = 1  # SB3 loggue par épisode par défaut via TensorBoard. Pour les logs console, c'est via `verbose`.

fc_layer_params = tuple(config["rl_training"]["fc_layer_params"])

# Récupérer les chemins depuis la configuration
project_root = Path(config["project_root"])
DATA_PATH = config["paths"]["merged_features_file"]
SB3_MODEL_SAVE_PATH = f"{config['paths']['rl_agent_model_dir']}/rl_agent_sb3_ppo_multiasset"  # SB3 ajoutera .zip
SB3_LOG_PATH = config["paths"]["rl_agent_logs_dir"]

# Construire les chemins absolus
data_path_abs = project_root / DATA_PATH
sb3_model_save_path_abs = project_root / SB3_MODEL_SAVE_PATH
sb3_log_path_abs = project_root / SB3_LOG_PATH

os.makedirs(os.path.dirname(sb3_model_save_path_abs), exist_ok=True)
os.makedirs(sb3_log_path_abs, exist_ok=True)

# Vérifier si l'encodeur doit être désactivé
skip_encoder = args.skip_encoder or config["rl_training"].get("skip_encoder", False)
logger.info(f"Mode encodeur {'désactivé' if skip_encoder else 'activé'}")

# Créer l'environnement pour la collecte de données ou l'entraînement
def make_env_func():  # Renommé pour éviter conflit avec variable 'make_env' de SB3
    env = MultiAssetEnv(
        data_path=data_path_abs,
        encoder_model=None,  # L'encodeur sera chargé depuis config.yaml
        initial_capital=config["rl_training"]["initial_capital"],
        transaction_cost_pct=config["rl_training"]["transaction_cost_pct"],
        verbose_env=True,  # Activer les logs pour le debug (renommé de verbose à verbose_env)
        mode="train",
        skip_encoder=skip_encoder  # Utiliser l'option skip_encoder définie plus haut
    )
    # Adapter pour SB3/Gymnasium
    if SB3_AVAILABLE:
        env = Monitor(env)  # Pour wrapper et logger les récompenses, longueurs d'épisodes, etc.
    return env

# Ignorer l'entraînement si demandé ou si SB3 n'est pas disponible
if args.skip_training or not SB3_AVAILABLE:
    # Créer un seul environnement pour la collecte de données
    logger.info("Mode collecte de données uniquement (sans entraînement)")
    single_env = make_env_func()
    
    # Si l'utilisateur a demandé des statistiques, les collecter manuellement
    if args.log_history:
        # Collecter des données en avançant dans l'environnement avec des actions aléatoires
        obs = single_env.reset()
        for _ in range(min(1000, total_timesteps)):  # Limiter à 1000 steps ou le nombre demandé
            action = single_env.action_space.sample()  # Action aléatoire
            obs, reward, done, info = single_env.step(action)
            if done:
                obs = single_env.reset()
        
        # Afficher les statistiques collectées
        console.print("\n[bold green]═════════ STATISTIQUES DE COLLECTE ═════════[/bold green]")
        if hasattr(single_env, 'history') and len(single_env.history) > 0:
            # Afficher le tableau des pas
            table = Table(title="Historique des étapes de trading")
            # Ajouter des colonnes similaires à celles dans TradeHistoryCallback
            # [Code d'affichage ici]
        
        single_env.close()
        exit(0)
else:
    # Procéder à l'entraînement normal avec SB3
    n_envs = args.n_envs if args.n_envs else 1
    logger.info(f"Création de {n_envs} environnements parallèles")
    train_env = make_vec_env(make_env_func, n_envs=n_envs)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,  # Réduit à 1e-4 pour stabilité
        n_steps=2048,  # Nombre d'étapes pour collecter les expériences (augmenté)
        batch_size=batch_size,
        gamma=config["rl_training"]["gamma"],  # Facteur de discount
        gae_lambda=0.95,  # Facteur pour Generalized Advantage Estimation
        clip_range=0.2,  # Paramètre de clipping pour PPO
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Coefficient d'entropie pour l'exploration
        vf_coef=0.5,  # Coefficient de la fonction de valeur
        max_grad_norm=0.5,  # Clipper le gradient pour stabiliser l'entraînement
        policy_kwargs=dict(net_arch=list(fc_layer_params)),  # Architecture du réseau (ex: [100, 50])
        verbose=1,  # Activer les warnings
        tensorboard_log=sb3_log_path_abs
    )

# Créer les callbacks
nan_fix_callback = NanInfFixCallback(verbose=1)
trade_history_callback = TradeHistoryCallback(verbose=1, log_history=args.log_history)

# Combiner les callbacks
if SB3_AVAILABLE:
    callbacks = CallbackList([nan_fix_callback, trade_history_callback])
else:
    callbacks = None

# Vérifier si Stable Baselines 3 est disponible avant de commencer l'entraînement
if not SB3_AVAILABLE:
    console.print("[bold red]ERREUR: Stable Baselines 3 ou PyTorch n'est pas disponible.[/bold red]")
    console.print("[yellow]Veuillez installer les packages requis avec:[/yellow]")
    console.print("pip install stable-baselines3 torch")
    console.print("\n[green]Pour utiliser uniquement les fonctionnalités de collecte de données sans SB3:[/green]")
    console.print("python train_rl_agent.py --log-history --skip-training")
    exit(1)

logger.info("Démarrage de l'entraînement de l'agent RL avec Stable-Baselines3...")
logger.info(f"Démarrage de l'entraînement sur {total_timesteps} timesteps avec {n_envs} environnements")
try:
    model.learn(
        total_timesteps=total_timesteps, 
        log_interval=log_interval_sb3,
        callback=callbacks,
        progress_bar=True
    )
    logger.info("Entraînement de l'agent RL terminé.")
    model.save(sb3_model_save_path_abs)
    logger.info(f"Modèle sauvegardé à l'emplacement : {sb3_model_save_path_abs}.zip")
    
    # Afficher les statistiques si demandé
    if args.log_history:
        console.print("\n[bold green]═════════ STATISTIQUES D'ENTRAÎNEMENT ═════════[/bold green]")
        
        # Récupérer l'environnement de base (premier environnement vectorisé)
        env = train_env.envs[0] if hasattr(train_env, 'envs') else train_env
        
        # Préparer les DataFrames
        trade_history_callback.prepare_dataframes(env)
        
        # Afficher les statistiques
        trade_history_callback.display_step_history()
        console.print("\n[bold green]═════════ RÉSUMÉ PAR PALIER ═════════[/bold green]")
        trade_history_callback.display_tier_statistics()
        
        # Générer et afficher les graphiques
        console.print("\n[bold green]═════════ VISUALISATIONS ═════════[/bold green]")
        
        # Créer les répertoires pour les exports si nécessaire
        if args.export_charts or args.export_csv or args.export_all:
            import os
            # Utiliser le répertoire personnalisé s'il est fourni
            base_export_dir = args.export_dir if args.export_dir else sb3_log_path_abs
            
            # Créer un sous-dossier avec la date et l'heure
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(base_export_dir, f"trading_stats_{timestamp}")
            
            charts_dir = os.path.join(export_dir, "charts")
            data_dir = os.path.join(export_dir, "data")
            
            os.makedirs(export_dir, exist_ok=True)
            os.makedirs(charts_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            console.print(f"[green]Dossier d'export créé: {export_dir}[/green]")
        
        # Tracer les graphiques si matplotlib est disponible
        if MATPLOTLIB_AVAILABLE:
            if args.export_charts or args.export_all:
                portfolio_chart_path = os.path.join(charts_dir, "portfolio_evolution.png")
                reward_chart_path = os.path.join(charts_dir, "reward_evolution.png")
                tier_stats_path = os.path.join(charts_dir, "tier_statistics.png")
                
                # Graphiques supplémentaires
                drawdown_chart_path = os.path.join(charts_dir, "drawdown_evolution.png")
                cumulative_pnl_path = os.path.join(charts_dir, "cumulative_pnl.png")
                
                trade_history_callback.plot_portfolio_evolution(portfolio_chart_path)
                trade_history_callback.plot_reward_evolution(reward_chart_path)
                trade_history_callback.plot_tier_statistics(tier_stats_path)
                
                # Graphique de drawdown si disponible
                if trade_history_callback.history_df is not None and 'drawdown' in trade_history_callback.history_df.columns:
                    plt.figure(figsize=(12, 6))
                    plt.plot(trade_history_callback.history_df['step'], 
                             trade_history_callback.history_df['drawdown'], 
                             color='red', label='Drawdown (%)')
                    plt.title('Évolution du drawdown')
                    plt.xlabel('Étape')
                    plt.ylabel('Drawdown (%)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(drawdown_chart_path, dpi=300, bbox_inches='tight')
                    console.print(f"[green]Graphique de drawdown sauvegardé: {drawdown_chart_path}[/green]")
                
                # Graphique du PnL cumulatif si disponible
                if trade_history_callback.trades_df is not None and 'cumulative_pnl' in trade_history_callback.trades_df.columns:
                    plt.figure(figsize=(12, 6))
                    plt.plot(trade_history_callback.trades_df.index, 
                             trade_history_callback.trades_df['cumulative_pnl'], 
                             color='green', label='PnL cumulatif ($)')
                    plt.title('Évolution du PnL cumulatif')
                    plt.xlabel('Trade #')
                    plt.ylabel('PnL cumulatif ($)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(cumulative_pnl_path, dpi=300, bbox_inches='tight')
                    console.print(f"[green]Graphique du PnL cumulatif sauvegardé: {cumulative_pnl_path}[/green]")
            else:
                trade_history_callback.plot_portfolio_evolution()
                trade_history_callback.plot_reward_evolution()
                trade_history_callback.plot_tier_statistics()
        else:
            console.print("[yellow]matplotlib n'est pas installé. Les visualisations graphiques ne sont pas disponibles.[/yellow]")
            console.print("[green]Pour générer des graphiques, installez matplotlib avec: pip install matplotlib[/green]")
        
        # Exporter les données en CSV si demandé
        if (args.export_csv or args.export_all) and trade_history_callback.trades_df is not None:
            trades_csv_path = os.path.join(data_dir, "trades_history.csv")
            history_csv_path = os.path.join(data_dir, "portfolio_history.csv")
            tier_csv_path = os.path.join(data_dir, "tier_statistics.csv")
            
            trade_history_callback.trades_df.to_csv(trades_csv_path, index=False)
            console.print(f"[green]Données de trades exportées: {trades_csv_path}[/green]")
            
            if trade_history_callback.history_df is not None:
                trade_history_callback.history_df.to_csv(history_csv_path, index=False)
                console.print(f"[green]Historique du portefeuille exporté: {history_csv_path}[/green]")
                
            if trade_history_callback.tier_df is not None:
                trade_history_callback.tier_df.to_csv(tier_csv_path, index=False)
                console.print(f"[green]Statistiques par palier exportées: {tier_csv_path}[/green]")
                
            # Exporter également les données brutes au format JSON
            import json
            raw_trades_path = os.path.join(data_dir, "raw_trades.json")
            raw_history_path = os.path.join(data_dir, "raw_history.json")
            
            # Sauvegarder les données brutes de l'environnement
            if hasattr(env, 'trade_log') and len(env.trade_log) > 0:
                with open(raw_trades_path, 'w') as f:
                    # Conversion des données pour JSON (timestamp, etc.)
                    trades_json = []
                    for trade in env.trade_log:
                        trade_dict = {k: str(v) if isinstance(v, (pd.Timestamp, datetime.datetime)) else v 
                                    for k, v in trade.items()}
                        trades_json.append(trade_dict)
                    json.dump(trades_json, f, indent=2)
                console.print(f"[green]Données brutes des trades exportées: {raw_trades_path}[/green]")
                
            if hasattr(env, 'history') and len(env.history) > 0:
                with open(raw_history_path, 'w') as f:
                    # Conversion des données pour JSON
                    history_json = []
                    for record in env.history:
                        record_dict = {k: str(v) if isinstance(v, (pd.Timestamp, datetime.datetime)) else 
                                    v.tolist() if isinstance(v, np.ndarray) else v 
                                    for k, v in record.items()}
                        history_json.append(record_dict)
                    json.dump(history_json, f, indent=2)
                console.print(f"[green]Données brutes de l'historique exportées: {raw_history_path}[/green]")
        
    # Évaluer le modèle si demandé
    if args.evaluate:
        console.print("\n[bold green]═════════ ÉVALUATION DU MODÈLE ═════════[/bold green]")
        logger.info("Évaluation du modèle...")
        
        # Créer un environnement d'évaluation
        eval_env = MultiAssetEnv(
            data_path=data_path_abs,
            encoder_model=None,
            initial_capital=config["rl_training"]["initial_capital"],
            transaction_cost_pct=config["rl_training"]["transaction_cost_pct"],
            verbose_env=True,  # Renommé de verbose à verbose_env
            mode="eval",  # Mode évaluation
            skip_encoder=skip_encoder
        )
        
        # Exécuter quelques épisodes d'évaluation
        obs = eval_env.reset()
        done = False
        eval_rewards = []
        current_reward = 0
        
        for _ in range(1000):  # Limiter à 1000 steps maximum
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            current_reward += reward
            
            if done:
                eval_rewards.append(current_reward)
                current_reward = 0
                obs = eval_env.reset()
                
        # Afficher les résultats d'évaluation
        if eval_rewards:
            console.print(f"Récompenses moyennes en évaluation: {np.mean(eval_rewards):.4f}")
            console.print(f"Écart-type des récompenses: {np.std(eval_rewards):.4f}")
            console.print(f"Min/Max des récompenses: {min(eval_rewards):.4f}/{max(eval_rewards):.4f}")
        
        # Fermer l'environnement d'évaluation
        eval_env.close()
        
except Exception as e:
    logger.error(f"Erreur pendant l'entraînement: {e}")
    # Sauvegarder quand même le modèle partiel en cas d'erreur
    try:
        model.save(sb3_model_save_path_abs)
        logger.info(f"Modèle partiel sauvegardé à : {sb3_model_save_path_abs}.zip")
        
        # Afficher les statistiques partielles si demandé
        if args.log_history:
            console.print("\n[bold yellow]═════════ STATISTIQUES PARTIELLES ═════════[/bold yellow]")
            trade_history_callback.display_step_history()
            trade_history_callback.display_tier_statistics()
            
    except Exception as save_error:
        logger.error(f"Impossible de sauvegarder le modèle partiel: {save_error}")

# Fermer l'environnement à la fin
train_env.close()

# Afficher un message de fin
console.print("\n[bold green]Exécution terminée![/bold green]")
