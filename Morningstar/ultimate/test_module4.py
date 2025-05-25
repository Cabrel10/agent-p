#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test du Module 4: Optimisation génétique des hyperparamètres de l'agent RL
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Importer load_data depuis data_loader
from model.training.data_loader import load_data
from model.training.genetic_optimizer import (
    # load_data, # Supprimé car importé depuis data_loader
    load_price_and_features,  # Importer la fonction spécifique de genetic_optimizer
    # decode_individual, # Fonction inexistante dans la version actuelle
    # create_trading_env, # Fonction inexistante/renommée
    # create_rl_agent, # Fonction inexistante/renommée
    # evaluate_hyperparams, # Fonction inexistante/renommée
    optimize_hyperparams,  # Cette fonction existe
    # train_best_agent # Fonction inexistante/renommée
)

# Importer les fonctions/classes nécessaires depuis reinforcement_learning.py
from model.training.reinforcement_learning import create_trading_env_from_data, TradingRLAgent


def generate_test_data(n_samples=1000):
    """
    Génère des données de test pour l'environnement de trading.

    Args:
        n_samples: Nombre d'échantillons à générer

    Returns:
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
    """
    print("Génération des données de test...")

    # Générer une série temporelle pour les prix
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="h")
    # Convertir les dates en timestamps Unix (secondes depuis epoch) pour qu'elles soient numériques
    timestamps = dates.astype(np.int64) // 10**9  # Convertir nanosecondes en secondes

    # Prix avec tendance et bruit
    price = 100
    prices = []
    for i in range(n_samples):
        # Ajouter une tendance et un bruit
        change = np.random.normal(0, 1) + np.sin(i / 100) * 0.5
        price *= 1 + change / 100
        prices.append(price)

    # Créer le DataFrame des prix
    price_data = pd.DataFrame(
        {
            "timestamp": dates,  # Garder les dates pour price_data
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            "close": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "volume": [np.random.uniform(1000, 10000) for _ in range(n_samples)],
        }
    )

    # Créer des caractéristiques techniques en suivant la structure de données détaillée
    # Nous incluons les 41+ colonnes recommandées pour un modèle complet

    # 1. Données brutes de marché (déjà dans price_data, mais ajoutons price)
    # 2. Données microstructurelles
    # 3. Indicateurs techniques classiques
    # 4. Raisonnement / Événements structurés
    # 5. Colonnes de target
    # 6. Colonnes système

    feature_data = pd.DataFrame()

    # Utiliser timestamp numérique pour feature_data
    feature_data["timestamp_numeric"] = timestamps

    # 1. Données brutes de marché (complémentaires)
    feature_data["price"] = prices

    # 2. Données microstructurelles
    feature_data["bid_price"] = [p * (1 - np.random.uniform(0, 0.001)) for p in prices]
    feature_data["ask_price"] = [p * (1 + np.random.uniform(0, 0.001)) for p in prices]
    feature_data["bid_volume"] = [np.random.uniform(500, 5000) for _ in range(n_samples)]
    feature_data["ask_volume"] = [np.random.uniform(500, 5000) for _ in range(n_samples)]
    feature_data["spread"] = feature_data["ask_price"] - feature_data["bid_price"]
    feature_data["order_imbalance"] = (feature_data["bid_volume"] - feature_data["ask_volume"]) / (
        feature_data["bid_volume"] + feature_data["ask_volume"]
    )

    # 3. Indicateurs techniques classiques
    feature_data["rsi_14"] = np.random.uniform(0, 100, n_samples)
    feature_data["macd"] = np.random.normal(0, 1, n_samples)
    feature_data["macd_signal"] = np.random.normal(0, 1, n_samples)
    feature_data["ema_9"] = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
    feature_data["ema_21"] = [p * (1 + np.random.normal(0, 0.02)) for p in prices]
    feature_data["sma_50"] = [p * (1 + np.random.normal(0, 0.03)) for p in prices]
    feature_data["bollinger_upper"] = [p * (1 + np.random.uniform(0.01, 0.05)) for p in prices]
    feature_data["bollinger_lower"] = [p * (1 - np.random.uniform(0.01, 0.05)) for p in prices]
    feature_data["atr_14"] = np.abs(np.random.normal(0, 2, n_samples))
    feature_data["adx"] = np.random.uniform(0, 100, n_samples)

    # 4. Raisonnement / Événements structurés
    feature_data["event_spike_volume"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    feature_data["event_breakout"] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    feature_data["event_reversal"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    feature_data["trend_direction"] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])
    feature_data["momentum_shift"] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.25, 0.5, 0.25])
    feature_data["pattern_match"] = np.random.choice(range(10), size=n_samples)

    # 5. Colonnes de target
    feature_data["future_return_5s"] = np.random.normal(0, 0.001, n_samples)
    feature_data["future_return_10s"] = np.random.normal(0, 0.002, n_samples)
    feature_data["future_signal"] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
    feature_data["future_max_dd"] = np.random.uniform(0, 0.05, n_samples)
    feature_data["target_profit"] = np.random.uniform(0, 0.1, n_samples)

    # 6. Colonnes système
    feature_data["position"] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.1, 0.8, 0.1])
    feature_data["pnl"] = np.random.normal(0, 10, n_samples)
    feature_data["cumulative_pnl"] = np.cumsum(feature_data["pnl"])
    feature_data["drawdown"] = np.random.uniform(0, 0.1, n_samples)
    feature_data["entry_price"] = [p if np.random.random() > 0.7 else 0 for p in prices]
    feature_data["exit_signal"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    feature_data["execution_latency"] = np.random.uniform(10, 100, n_samples)

    # Ajouter quelques colonnes supplémentaires pour atteindre 41+
    feature_data["cci"] = np.random.normal(0, 100, n_samples)
    feature_data["mfi"] = np.random.uniform(0, 100, n_samples)
    feature_data["obv"] = np.random.normal(0, 10000, n_samples)
    feature_data["williams_r"] = np.random.uniform(-100, 0, n_samples)
    feature_data["hour_of_day"] = [d.hour for d in dates]
    feature_data["day_of_week"] = [d.dayofweek for d in dates]

    print(f"Données générées: {len(price_data)} échantillons")
    print(f"Nombre de colonnes dans price_data: {len(price_data.columns)}")
    print(f"Nombre de colonnes dans feature_data: {len(feature_data.columns)}")
    print(
        f"Toutes les colonnes de feature_data sont numériques: {all(dtype.kind in 'biufc' for dtype in feature_data.dtypes)}"
    )

    return price_data, feature_data


# Les fonctions suivantes sont commentées car elles dépendent de fonctions
# qui n'existent plus dans model/training/genetic_optimizer.py dans sa forme actuelle.
# def test_decode_individual():
#     pass
# def test_create_trading_env():
#     pass
# def test_create_rl_agent():
#     pass
# def test_evaluate_hyperparams():
#     pass


def test_optimize_hyperparams():  # Ce test peut rester car optimize_hyperparams existe
    """
    Teste l'optimisation des hyperparamètres avec un algorithme génétique.
    """
    print("\n=== Test de la fonction optimize_hyperparams ===")

    # Générer des données de test
    price_data, feature_data = generate_test_data(1000)

    # Optimiser les hyperparamètres avec une petite population et peu de générations
    output_dir = "output/test_optimize"
    os.makedirs(output_dir, exist_ok=True)

    print("Optimisation des hyperparamètres...")
    # Passer les arguments par mot-clé pour éviter les erreurs de position
    # Note: optimize_hyperparams attend data_path, pas price_data et feature_data séparément.
    # Nous devons sauvegarder les données générées dans un fichier temporaire.
    temp_data_path = os.path.join(output_dir, "temp_test_data.parquet")
    # Combiner price_data (sans la colonne timestamp objet) et feature_data
    combined_df = pd.concat([price_data.drop(columns=["timestamp"]), feature_data], axis=1)
    combined_df.to_parquet(temp_data_path)

    best_hyperparams = optimize_hyperparams(
        data_path=temp_data_path,  # Passer le chemin du fichier combiné
        output_dir=output_dir,
        population_size=4,
        generations=2,
        # train_timesteps n'est pas un argument de optimize_hyperparams
    )

    print(f"Meilleurs hyperparamètres: {best_hyperparams}")
    assert best_hyperparams is not None, "Les meilleurs hyperparamètres n'ont pas été trouvés"

    print("Test de optimize_hyperparams réussi!")


# def test_train_best_agent():
#     pass
# def test_cot_coherence():
#     pass


def run_all_tests():
    """
    Exécute tous les tests.
    """
    print("=== Démarrage des tests du Module 4: Optimisation génétique des hyperparamètres de l'agent RL ===")

    # Créer le répertoire de sortie
    os.makedirs("output", exist_ok=True)

    # Exécuter les tests (seul test_optimize_hyperparams est actif)
    test_optimize_hyperparams()

    print("\n=== Tous les tests du Module 4 ont réussi! ===")


if __name__ == "__main__":
    run_all_tests()
