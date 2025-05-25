#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de dataset pour 4 paires crypto en timeframe d'une minute
sur la période de janvier à mars 2024.

Ce script utilise les modules existants pour télécharger et générer un dataset
complet et enrichi prêt à être utilisé par le modèle monolithique.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ajouter le chemin du projet au PYTHONPATH
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Ajustez selon votre structure
sys.path.append(str(PROJECT_ROOT))

# Imports des modules du projet
from ultimate.utils.api_manager import APIManager
from ultimate.utils.feature_engineering import apply_feature_pipeline
from ultimate.utils.market_regime import MarketRegimeDetector
from ultimate.scripts.datasetbiuld import MultiCryptoDatasetGenerator

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_minutely_dataset")

def main():
    """Fonction principale de génération du dataset minutely."""
    
    # Paires de crypto à télécharger (différentes de celles déjà utilisées)
    pairs = [
        "DOT/USDT",  # Polkadot
        "LINK/USDT", # Chainlink
        "AVAX/USDT", # Avalanche
        "MATIC/USDT" # Polygon
    ]
    
    # Période
    start_date = "2024-01-01"
    end_date = "2024-03-31"
    interval = "1m"  # Timeframe d'une minute
    
    # Répertoire de sortie
    output_dir = os.path.join(PROJECT_ROOT, "ultimate", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Génération d'un dataset pour {', '.join(pairs)} en timeframe {interval}")
    logger.info(f"Période: {start_date} à {end_date}")
    
    # Utiliser le générateur existant
    generator = MultiCryptoDatasetGenerator(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        output_dir=output_dir,
        pairs=pairs
    )
    
    # Générer le dataset
    try:
        generator.generate()
        
        # Renommer le fichier de sortie pour éviter d'écraser le précédent
        output_file = os.path.join(output_dir, "multi_crypto_dataset.parquet")
        minutely_output_file = os.path.join(output_dir, "minutely_crypto_dataset.parquet")
        
        if os.path.exists(output_file):
            os.rename(output_file, minutely_output_file)
            logger.info(f"Dataset renommé en {minutely_output_file}")
        
        logger.info("Génération du dataset minutely terminée avec succès!")
        
        # Vérifier la taille du fichier
        if os.path.exists(minutely_output_file):
            file_size_mb = os.path.getsize(minutely_output_file) / (1024 * 1024)
            logger.info(f"Taille du fichier: {file_size_mb:.2f} Mo")
            
            # Afficher un aperçu du dataset
            df = pd.read_parquet(minutely_output_file)
            logger.info(f"Dataset généré: {len(df)} lignes, {len(df.columns)} colonnes")
            logger.info(f"Paires présentes: {df['symbol'].unique()}")
            logger.info(f"Période couverte: {df.index.min()} à {df.index.max()}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du dataset: {e}")
        raise

if __name__ == "__main__":
    main() 