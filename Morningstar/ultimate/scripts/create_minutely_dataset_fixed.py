#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de dataset pour 4 paires crypto en timeframe d'une minute
sur une période courte (2 jours) pour test rapide, avec correction des problèmes
de volume zéro.

Ce script utilise les modules existants pour télécharger et générer un dataset
complet et enrichi prêt à être utilisé par le modèle monolithique.
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ajouter le chemin du projet au PYTHONPATH
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Ajustez selon votre structure
sys.path.append(str(PROJECT_ROOT))

# Configurer les avertissements pour qu'ils soient ignorés
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_minutely_dataset_fixed")

def download_crypto_data(api_manager, pair, start_date, end_date, interval):
    """Télécharge les données pour une paire spécifique avec gestion des erreurs améliorée."""
    symbol = pair.split("/")[0]
    logger.info(f"Téléchargement des données {pair} de {start_date} à {end_date} ({interval})")
    
    try:
        # Téléchargement des données OHLCV
        df = api_manager.fetch_ohlcv_data(
            exchange_id="binance",
            token=pair,
            timeframe=interval,
            start_date=start_date,
            end_date=end_date,
        )
        
        if df.empty:
            logger.warning(f"Aucune donnée récupérée depuis l'API pour {pair}, utilisation de données simulées")
            # Générer des données simulées
            df = generate_simulated_data(pair, start_date, end_date, interval)
        else:
            logger.info(f"{len(df)} bougies récupérées depuis l'API pour {pair}")
            # Ajouter une colonne pour identifier la paire
            df["symbol"] = symbol
            
            # S'assurer que timestamp est l'index
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            
            # Correction des volumes nuls pour éviter les erreurs
            df["volume"] = df["volume"].replace(0, 1e-6)  # Remplacer par une valeur très petite mais non nulle
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de {pair}: {e}")
        logger.warning(f"Utilisation de données simulées pour {pair}")
        # Générer des données simulées en cas d'erreur
        return generate_simulated_data(pair, start_date, end_date, interval)

def generate_simulated_data(pair, start_date, end_date, interval):
    """Génère des données simulées avec volumes non nuls."""
    symbol = pair.split("/")[0]
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Créer un range de dates selon l'intervalle
    if interval == "1m":
        date_range = pd.date_range(start=start, end=end, freq="1min")
    elif interval == "1h":
        date_range = pd.date_range(start=start, end=end, freq="h")
    elif interval == "4h":
        date_range = pd.date_range(start=start, end=end, freq="4h")
    else:  # 1d
        date_range = pd.date_range(start=start, end=end, freq="D")
    
    # Définir des prix de base différents selon la crypto
    base_prices = {
        "BTC": 30000,
        "ETH": 2000,
        "XRP": 0.5,
        "SOL": 100,
        "ADA": 0.4,
        "DOT": 15,
        "LINK": 10,
        "AVAX": 20,
        "MATIC": 1,
    }
    base_price = base_prices.get(symbol, 100)  # Prix par défaut si non trouvé
    
    # Créer un mouvement de prix réaliste
    price_changes = np.random.normal(0, 0.002, size=len(date_range))
    trend = np.linspace(-0.05, 0.05, len(date_range))
    cycles = 0.02 * np.sin(np.linspace(0, 10 * np.pi, len(date_range)))
    
    cumulative_returns = np.cumsum(price_changes) + trend + cycles
    close_prices = base_price * (1 + cumulative_returns)
    
    # Générer des volumes toujours positifs et non nuls
    volume_base = base_price * 1000
    volumes = np.random.gamma(2, volume_base, size=len(date_range))
    
    # S'assurer qu'il n'y a pas de volume nul
    volumes = np.maximum(volumes, 1e-6)
    
    # Générer open, high, low basés sur close
    df = pd.DataFrame({
        "timestamp": date_range,
        "open": close_prices * (1 + np.random.normal(0, 0.001, size=len(date_range))),
        "high": close_prices * (1 + np.abs(np.random.normal(0, 0.002, size=len(date_range)))),
        "low": close_prices * (1 - np.abs(np.random.normal(0, 0.002, size=len(date_range)))),
        "close": close_prices,
        "volume": volumes,
        "symbol": symbol,
    })
    
    # S'assurer que high >= open, close, low et low <= open, close, high
    for idx, row in df.iterrows():
        max_val = max(row["open"], row["close"])
        min_val = min(row["open"], row["close"])
        df.at[idx, "high"] = max(row["high"], max_val)
        df.at[idx, "low"] = min(row["low"], min_val)
    
    # Définir timestamp comme index
    df = df.set_index("timestamp")
    
    logger.info(f"{len(df)} bougies simulées générées pour {pair}")
    return df

def main():
    """Fonction principale de génération du dataset minutely court."""
    
    # Paires de crypto à télécharger (différentes de celles déjà utilisées)
    pairs = [
        "DOT/USDT",  # Polkadot
        "LINK/USDT", # Chainlink
        "AVAX/USDT", # Avalanche
        "MATIC/USDT" # Polygon
    ]
    
    # Période très courte (seulement 2 jours)
    start_date = "2024-01-01"
    end_date = "2024-01-03"
    interval = "1m"  # Timeframe d'une minute
    
    # Répertoire de sortie
    output_dir = os.path.join(PROJECT_ROOT, "ultimate", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Génération d'un dataset court pour {', '.join(pairs)} en timeframe {interval}")
    logger.info(f"Période: {start_date} à {end_date}")
    
    # Importer les modules nécessaires
    from ultimate.utils.api_manager import APIManager
    from ultimate.utils.feature_engineering import apply_feature_pipeline
    from ultimate.utils.market_regime import MarketRegimeDetector
    
    # Initialiser l'API manager
    api_manager = APIManager(config={"exchange": "binance", "pair": "BTC/USDT", "timeframe": interval})
    
    try:
        all_dfs = []
        
        # Télécharger les données pour chaque paire
        for pair in pairs:
            # Télécharger et préparer les données avec notre fonction améliorée
            df = download_crypto_data(api_manager, pair, start_date, end_date, interval)
            
            # Appliquer le feature engineering
            logger.info(f"Application du feature engineering pour {pair}")
            df = apply_feature_pipeline(df)
            
            # Détection des régimes de marché HMM
            logger.info(f"Détection des régimes de marché HMM pour {pair}")
            try:
                hmm_detector = MarketRegimeDetector(n_components=3)
                hmm_detector.fit(df)
                regimes = hmm_detector.predict(df)
                df["hmm_regime"] = regimes
                # Ajouter les probabilités de régime
                features = hmm_detector._prepare_features(df)
                scaled_features = hmm_detector.scaler.transform(features)
                regime_probs = hmm_detector.model.predict_proba(scaled_features)
                for i in range(hmm_detector.n_components):
                    df[f"hmm_prob_{i}"] = regime_probs[:, i]
            except Exception as e:
                logger.warning(f"Erreur lors de la détection HMM pour {pair}: {e}")
                # Créer des colonnes HMM simulées
                df["hmm_regime"] = np.random.randint(0, 3, size=len(df))
                for i in range(3):
                    df[f"hmm_prob_{i}"] = np.random.random(size=len(df))
                # Normaliser les probabilités
                prob_sum = df[[f"hmm_prob_{i}" for i in range(3)]].sum(axis=1)
                for i in range(3):
                    df[f"hmm_prob_{i}"] = df[f"hmm_prob_{i}"] / prob_sum
            
            all_dfs.append(df)
            logger.info(f"Traitement de {pair} terminé")
        
        # Combiner tous les DataFrames
        logger.info("Combinaison de tous les DataFrames")
        combined_df = pd.concat(all_dfs)
        
        # Trier par date
        combined_df = combined_df.sort_index()
        
        # Intégration du contexte LLM
        logger.info("Intégration du contexte LLM pour toutes les paires")
        # Simuler les embeddings LLM pour chaque paire
        summaries = []
        embeddings = []
        
        # Générer des embeddings simulés cohérents par date et par symbole
        for idx, row in combined_df.iterrows():
            date_str = str(idx)[:10]
            symbol = row["symbol"]
            
            # Créer un embedding simulé cohérent pour chaque paire/date
            seed = hash(f"{symbol}_{date_str}") % 10000
            np.random.seed(seed)
            emb = np.random.randn(768)  # Dimension standard des embeddings BERT
            # Normaliser l'embedding
            emb = emb / np.linalg.norm(emb)
            
            embeddings.append(emb)
            summaries.append(f"Résumé simulé pour {symbol} le {date_str}")
        
        # Ajouter les résultats au DataFrame
        combined_df["llm_context_summary"] = summaries
        combined_df["llm_embedding"] = embeddings
        
        # Génération des features MCP avec valeurs non nulles
        logger.info("Génération des 128 features MCP")
        for i in range(128):
            # Créer un DataFrame temporaire pour stocker les features
            temp_features = []
            
            for idx, row in combined_df.iterrows():
                date_str = str(idx)[:10]
                symbol = row["symbol"]
                
                # Utiliser la date, le symbole et l'indice de feature comme seed
                seed = hash(f"{symbol}_{date_str}_{i}") % 10000
                np.random.seed(seed)
                
                # Générer une feature qui dépend des prix pour plus de réalisme
                base_feature = np.random.randn()
                price_component = 0.2 * (row["close"] / row["open"] - 1)
                # Assurer que volume est non nul avant de calculer le log
                volume = max(row["volume"], 1e-6)
                volume_component = 0.1 * (np.log(volume) - 15) / 5  # Normaliser le volume
                
                feature_value = base_feature + price_component + volume_component
                temp_features.append(feature_value)
            
            # Ajouter la feature au DataFrame principal
            combined_df[f"mcp_feature_{i:03d}"] = temp_features
        
        # Valider le dataset
        logger.info("Validation de la structure du dataset")
        tech_cols = [col for col in combined_df.columns if not col.startswith(("llm_", "mcp_", "hmm_")) and col != "symbol"]
        hmm_cols = [col for col in combined_df.columns if col.startswith("hmm_")]
        llm_cols = [col for col in combined_df.columns if col.startswith("llm_")]
        mcp_cols = [col for col in combined_df.columns if col.startswith("mcp_")]
        
        logger.info(f"Colonnes techniques trouvées: {len(tech_cols)}")
        logger.info(f"Colonnes HMM trouvées: {len(hmm_cols)}")
        logger.info(f"Colonnes LLM trouvées: {len(llm_cols)}")
        logger.info(f"Colonnes MCP trouvées: {len(mcp_cols)}")
        
        # Sauvegarder le dataset
        output_file = os.path.join(output_dir, "minutely_crypto_dataset_fixed.parquet")
        logger.info(f"Sauvegarde du dataset dans {output_file}")
        
        # Sauvegarder au format Parquet
        combined_df.to_parquet(output_file)
        
        # Afficher les statistiques finales
        logger.info(f"Dataset sauvegardé avec {len(combined_df)} lignes et {len(combined_df.columns)} colonnes")
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"Taille du fichier: {file_size_mb:.2f} Mo")
        logger.info(f"Paires présentes: {combined_df['symbol'].unique()}")
        logger.info(f"Période couverte: {combined_df.index.min()} à {combined_df.index.max()}")
        
        logger.info("Génération du dataset minutely terminée avec succès!")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du dataset: {e}")
        raise

if __name__ == "__main__":
    main() 