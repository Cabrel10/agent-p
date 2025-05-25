import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze_backtest")

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis : {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {config_path}: {e}")
        return None

def calculate_max_drawdown(portfolio_values):
    """Calcule le Max Drawdown."""
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_sharpe_ratio(returns, periods_per_year=252, risk_free_rate=0.0):
    """Calcule le Sharpe Ratio annualisé."""
    if len(returns) < 2:
        return np.nan
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe_ratio * np.sqrt(periods_per_year)

def analyze_report(report_path_csv, initial_capital=10000):
    logger.info(f"Chargement du rapport de backtest depuis : {report_path_csv}")
    if not os.path.exists(report_path_csv):
        logger.error(f"Le fichier rapport {report_path_csv} n'a pas été trouvé !")
        return

    df = pd.read_csv(report_path_csv)

    if 'timestamp' not in df.columns:
        logger.error("La colonne 'timestamp' est manquante dans le rapport.")
        # Si pas de timestamp, on utilise l'index pour les graphiques
        df['timestamp_idx'] = df.index
        time_col = 'timestamp_idx'
        is_datetime_index = False
    else:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp').set_index('timestamp')
            time_col = 'timestamp'
            is_datetime_index = True
        except Exception as e:
            logger.warning(f"Impossible de convertir 'timestamp' en datetime: {e}. Utilisation de l'index.")
            df['timestamp_idx'] = df.index # Fallback si la conversion échoue
            time_col = 'timestamp_idx'
            is_datetime_index = False

    logger.info(f"Rapport chargé : {len(df)} pas de temps.")

    # 1. Évolution du capital et P&L cumulé
    final_capital = df['capital'].iloc[-1]
    total_pnl = final_capital - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

    logger.info(f"\n--- Performance Générale ---")
    logger.info(f"Capital Initial: {initial_capital:.2f}")
    logger.info(f"Capital Final: {final_capital:.2f}")
    logger.info(f"P&L Cumulé: {total_pnl:.2f}")
    logger.info(f"Rendement Total: {total_return_pct:.2f}%")

    # 2. Métriques Clés
    portfolio_values = pd.Series([initial_capital] + df['capital'].tolist()) # Ajoute le capital initial au début
    returns = portfolio_values.pct_change().dropna()

    max_drawdown = calculate_max_drawdown(portfolio_values)
    periods_per_year_estimate = 252 # À AJUSTER selon la fréquence réelle
    sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year_estimate)

    logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"Sharpe Ratio (annualisé, estimé): {sharpe_ratio:.2f} (basé sur {periods_per_year_estimate} périodes/an)")

    # 3. Répartition des actions
    action_counts = df['action'].value_counts(normalize=True).sort_index()
    action_labels = {0: 'ACHAT', 1: 'VENTE', 2: 'TENIR'}
    logger.info(f"\n--- Répartition des Actions ---")
    for action_code, percentage in action_counts.items():
        logger.info(f"Action {action_labels.get(action_code, action_code)}: {percentage*100:.2f}%")

    # 4. Visualisations
    output_dir = os.path.dirname(report_path_csv)

    # Courbe de valeur du portefeuille
    plt.figure(figsize=(12, 6))
    if is_datetime_index:
        plt.plot(df.index, df['capital'], label="Valeur du Portefeuille")
    else:
        plt.plot(df[time_col], df['capital'], label="Valeur du Portefeuille")
    plt.title("Évolution de la Valeur du Portefeuille (Backtest)")
    plt.xlabel("Temps" if is_datetime_index else "Pas de Temps")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_portfolio_value_detailed.png"))
    logger.info(f"Graphique de la valeur du portefeuille sauvegardé dans {output_dir}")

    # Courbe de Drawdown
    portfolio_series_for_plot = pd.Series(portfolio_values.values, index=pd.Index([pd.Timestamp("1970-01-01")] + list(df.index if is_datetime_index else df[time_col])))
    peak_plot = portfolio_series_for_plot.cummax()
    drawdown_plot = (portfolio_series_for_plot - peak_plot) / peak_plot

    plt.figure(figsize=(12, 6))
    if is_datetime_index:
        plt.plot(drawdown_plot.index, drawdown_plot * 100, label="Drawdown (%)", color='red')
    else:
        plt.plot(drawdown_plot.index, drawdown_plot * 100, label="Drawdown (%)", color='red')
    plt.title("Drawdown du Portefeuille (Backtest)")
    plt.xlabel("Temps" if is_datetime_index else "Pas de Temps")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_drawdown.png"))
    logger.info(f"Graphique du drawdown sauvegardé dans {output_dir}")

    # Heatmap des actions (barres)
    plt.figure(figsize=(8, 5))
    action_counts.rename(index=action_labels).plot(kind='bar', color=['green', 'red', 'grey'])
    plt.title("Distribution des Actions")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_action_distribution.png"))
    logger.info(f"Graphique de distribution des actions sauvegardé dans {output_dir}")

    logger.info("\nAnalyse terminée. Vérifiez les graphiques et les logs pour les détails.")

if __name__ == "__main__":
    # Charger la configuration
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    config_path = script_dir / "config.yaml"
    
    config = load_config(config_path)
    if not config:
        logger.critical("Impossible de charger la configuration. Arrêt du script.")
        exit(1)
    
    # Obtenir les valeurs par défaut depuis la configuration
    project_root = Path(config["project_root"])
    default_report_path = project_root / config["paths"]["backtest_reports_dir"] / config["scripts"]["backtest_rl_agent"]["default_report_name"]
    default_initial_capital = config["scripts"]["backtest_rl_agent"]["default_initial_capital"]
    
    parser = argparse.ArgumentParser(description="Analyse des résultats d'un backtest RL.")
    parser.add_argument("--report-path", type=str, default=str(default_report_path), 
                        help=f"Chemin vers le rapport CSV du backtest (défaut: {default_report_path}).")
    parser.add_argument("--initial-capital", type=float, default=default_initial_capital, 
                        help=f"Capital initial utilisé pour le backtest (défaut: {default_initial_capital}).")
    
    args = parser.parse_args()

    analyze_report(report_path_csv=args.report_path, initial_capital=args.initial_capital)
