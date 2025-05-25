#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour mettre en cache les données brutes des signaux potentiels.
Ce script lit un fichier de données, assigne un ID unique à chaque ligne (signal potentiel),
et sauvegarde les données pertinentes pour une explication future.
"""

import argparse
import pandas as pd
import uuid
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def cache_signal_data(input_file_path: str, output_cache_path: str):
    """
    Lit les données d'un fichier, génère des IDs de signal, et sauvegarde dans un cache Parquet.

    Args:
        input_file_path (str): Chemin vers le fichier de données d'entrée (CSV ou Parquet).
        output_cache_path (str): Chemin où sauvegarder le fichier Parquet du cache de signaux.
    """
    logger.info(f"Lecture des données d'entrée depuis : {input_file_path}")
    input_path = Path(input_file_path)
    output_path = Path(output_cache_path)

    if not input_path.exists():
        logger.error(f"Le fichier d'entrée {input_file_path} n'existe pas.")
        return

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        logger.error(f"Format de fichier non supporté pour l'entrée : {input_path.suffix}")
        return

    logger.info(f"{len(df)} lignes lues depuis le fichier d'entrée.")

    # Assurer que l'index est unique si on l'utilise comme base pour signal_id,
    # ou générer des UUIDs pour garantir l'unicité.
    # Pour cette version, nous allons utiliser un index simple si l'index est unique,
    # sinon, nous générons des UUIDs.
    # Une approche plus simple pour un cache statique est de juste utiliser l'index existant
    # si le dataset d'entrée est stable.
    # Si l'index n'est pas nommé ou est un simple RangeIndex, on le transforme en colonne.
    
    if df.index.name is None or isinstance(df.index, pd.RangeIndex):
        df = df.reset_index() # Crée une colonne 'index' si RangeIndex, ou utilise l'index existant
        # Si la nouvelle colonne est 'index', on la renomme pour éviter confusion
        if 'index' in df.columns and 'signal_id' not in df.columns:
             df.rename(columns={'index': 'original_index'}, inplace=True)


    # Générer des signal_id uniques. Pour cet exemple, on utilise un UUID.
    # Dans un système réel, cela pourrait être lié à un timestamp de signal, un ID de trade, etc.
    # Si on veut pouvoir récupérer par un index simple (0, 1, 2...), on peut utiliser df.index.
    # Pour la robustesse, UUID est bien.
    if 'signal_id' not in df.columns:
        df['signal_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        logger.info("Colonne 'signal_id' générée avec des UUIDs.")
    else:
        logger.info("La colonne 'signal_id' existe déjà. Utilisation des IDs existants.")
        df['signal_id'] = df['signal_id'].astype(str)


    # Sélectionner les colonnes à sauvegarder. Pour l'instant, on sauvegarde tout,
    # car `preprocess_data` dans `predict_with_reasoning` s'attend à toutes les colonnes.
    # On s'assure juste que signal_id est la première colonne pour la facilité de lecture.
    cols_to_save = ['signal_id'] + [col for col in df.columns if col != 'signal_id']
    df_cache = df[cols_to_save]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cache.to_parquet(output_path, index=False)
        logger.info(f"Cache de signaux sauvegardé avec succès dans : {output_path} ({len(df_cache)} signaux)")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du cache de signaux : {e}")

def main():
    parser = argparse.ArgumentParser(description="Créer un cache de données de signaux pour explication.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Chemin vers le fichier de données d'entrée (Parquet ou CSV)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Chemin vers le fichier Parquet de sortie pour le cache des signaux."
    )
    args = parser.parse_args()

    cache_signal_data(args.input, args.output)

if __name__ == "__main__":
    # Exemple d'utilisation:
    # python scripts/cache_signals.py --input tests/fixtures/small_test.parquet --output data/cache/signal_cache.parquet
    main()
