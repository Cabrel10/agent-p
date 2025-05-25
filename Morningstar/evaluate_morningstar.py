import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

# Charge l'historique des trades et la courbe de valeur du portefeuille
# (supposé sauvegardé par l'env ou le script de training)
trade_history = pd.read_csv('trade_history.csv')  # à adapter selon ton env
portfolio_curve = pd.read_csv('portfolio_curve.csv')  # idem

console = Console()

# 1. Statistiques de performance
roi = (portfolio_curve['value'].iloc[-1] / portfolio_curve['value'].iloc[0]) - 1
returns = portfolio_curve['value'].pct_change().dropna()
sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
cummax = portfolio_curve['value'].cummax()
drawdown = (portfolio_curve['value'] - cummax) / cummax
max_drawdown = drawdown.min()

# 2. Stats de trading
n_trades = len(trade_history)
n_win = (trade_history['pnl'] > 0).sum()
n_loss = (trade_history['pnl'] < 0).sum()
win_rate = n_win / n_trades if n_trades > 0 else 0

# 3. Périodes extrêmes
max_gain_period = returns.idxmax()
max_loss_period = returns.idxmin()

# 4. Profil du trader
freq = n_trades / len(portfolio_curve)
aggressiveness = trade_history['qty'].abs().mean() / portfolio_curve['value'].mean()

# 5. Affichage
stat_table = Table(title="Statistiques Morningstar", border_style="bright_green")
stat_table.add_column("Métrique", style="bold")
stat_table.add_column("Valeur", justify="right")
stat_table.add_row("ROI", f"{roi:.2%}")
stat_table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
stat_table.add_row("Max Drawdown", f"{max_drawdown:.2%}")
stat_table.add_row("Nb trades", str(n_trades))
stat_table.add_row("Nb gagnants", str(n_win))
stat_table.add_row("Nb perdants", str(n_loss))
stat_table.add_row("Win rate", f"{win_rate:.2%}")
stat_table.add_row("Max gain période", str(max_gain_period))
stat_table.add_row("Max perte période", str(max_loss_period))
stat_table.add_row("Fréquence trading", f"{freq:.2f}")
stat_table.add_row("Agressivité", f"{aggressiveness:.2f}")
console.print(stat_table)

# 6. Profil heuristique
if freq < 0.01:
    profil = "Ultra patient / buy & hold"
elif freq < 0.05:
    profil = "Swing trader / opportuniste"
else:
    profil = "Scalpeur / très actif"
console.print(f"\nProfil détecté : [bold cyan]{profil}[/bold cyan]")
