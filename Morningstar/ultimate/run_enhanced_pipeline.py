#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour exu00e9cuter le pipeline complet de Morningstar avec capacitu00e9 de raisonnement.
Ce script orchestre toutes les u00e9tapes du processus, de la collecte des donnu00e9es u00e0 l'entrau00eenement
du modu00e8le et u00e0 l'u00e9valuation des performances.
"""

import os
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Arguments parsu00e9s
    """
    parser = argparse.ArgumentParser(description="Pipeline complet de Morningstar")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT",  # Limité aux 5 premières paires
        help="Liste des symboles de crypto-monnaies su00e9paru00e9s par des virgules",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d", help="Timeframe pour les donnu00e9es OHLCV (1m, 5m, 15m, 1h, 4h, 1d)"
    )
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Date de du00e9but au format YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2025-05-01", help="Date de fin au format YYYY-MM-DD")
    parser.add_argument(
        "--skip-data-collection", action="store_true", help="Sauter l'u00e9tape de collecte des donnu00e9es"
    )
    parser.add_argument(
        "--skip-data-preparation", action="store_true", help="Sauter l'u00e9tape de pru00e9paration des donnu00e9es"
    )
    parser.add_argument(
        "--skip-model-training", action="store_true", help="Sauter l'u00e9tape d'entrau00eenement du modu00e8le"
    )
    parser.add_argument("--skip-backtesting", action="store_true", help="Sauter l'u00e9tape de backtesting")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'u00e9poques d'entrau00eenement")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille du batch")
    parser.add_argument(
        "--use-market-info", action="store_true", help="Utiliser les informations de marchu00e9 de CoinMarketCap"
    )
    parser.add_argument("--use-sentiment", action="store_true", help="Utiliser l'analyse de sentiment")
    parser.add_argument("--use-hmm", action="store_true", help="Utiliser la du00e9tection de ru00e9gime HMM")
    parser.add_argument("--use-cryptobert", action="store_true", help="Utiliser les embeddings CryptoBERT")
    parser.add_argument("--live", action="store_true", help="Lancer le mode de trading en direct")
    parser.add_argument(
        "--dry-run-live",
        dest="dry_run_live",
        action="store_true",
        help="Exécuter le trading en direct en mode dry_run (pas d'ordres réels)",
    )
    parser.add_argument(
        "--no-dry-run-live",
        dest="dry_run_live",
        action="store_false",
        help="Désactiver le dry_run pour le trading en direct (ordres réels)",
    )
    parser.set_defaults(dry_run_live=True)  # Par défaut, le trading live est en dry_run

    return parser.parse_args()


def setup_environment():
    """
    Configure l'environnement en cru00e9ant les ru00e9pertoires nu00e9cessaires.
    """
    logger.info("Configuration de l'environnement")

    # Cru00e9er les ru00e9pertoires nu00e9cessaires
    directories = [
        "data/enriched",
        "data/processed/normalized",
        "data/processed/normalized/scalers",
        "model/enhanced_reasoning_model",
        "model/enhanced_reasoning_model/logs",
        "results/backtesting",
        "results/predictions",
        "results/explanations",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ru00e9pertoire cru00e9u00e9: {directory}")

    # Vu00e9rifier si le fichier .env existe
    if not os.path.exists(".env"):
        logger.info("Fichier .env non trouvé, exécution de setup_api_keys.py")
        subprocess.run(["python", "setup_api_keys.py", "--use-mock-exchange"], check=True)


def collect_enriched_data(args):
    """
    Collecte des donnu00e9es enrichies.

    Args:
        args: Arguments parsu00e9s
    """
    logger.info("u00c9tape 1: Collecte des donnu00e9es enrichies")

    # Construire la commande
    cmd = [
        "python",
        "scripts/collect_enriched_data.py",
        "--symbols",
        args.symbols,
        "--timeframe",
        args.timeframe,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--output-dir",
        "data/enriched",
    ]

    # Ajouter les options pour les sources de donnu00e9es
    if args.use_market_info:
        cmd.append("--use-market-info")

    if args.use_sentiment:
        cmd.append("--use-sentiment")

    if args.use_hmm:
        cmd.append("--use-hmm")

    if args.use_cryptobert:
        cmd.append("--use-cryptobert")

    # Exu00e9cuter la commande
    logger.info(f"Exu00e9cution de la commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("Collecte des donnu00e9es enrichies terminu00e9e")


def prepare_dataset():
    """
    Pru00e9pare le dataset pour l'entrau00eenement.
    """
    logger.info("u00c9tape 2: Pru00e9paration du dataset")

    # Construire la commande
    cmd = [
        "python",
        "scripts/prepare_improved_dataset.py",
        "--input",
        "data/enriched/enriched_dataset.parquet",
        "--output",
        "data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv",
    ]

    # Exu00e9cuter la commande
    logger.info(f"Exu00e9cution de la commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("Pru00e9paration du dataset terminu00e9e")


def train_model(args):
    """
    Entrau00eene le modu00e8le avec capacitu00e9 de raisonnement.

    Args:
        args: Arguments parsu00e9s
    """
    logger.info("u00c9tape 3: Entrau00eenement du modu00e8le avec capacitu00e9 de raisonnement")

    # Construire la commande
    # Le script enhanced_reasoning_training.py lit les paramètres depuis config.yaml
    # Ne passer que les arguments obligatoires ou ceux non présents dans config.yaml
    cmd = [
        "python",
        "model/training/enhanced_reasoning_training.py",
        "--data-path",
        "data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv",
        "--output-dir",
        "model/enhanced_reasoning_model",
        # Les autres arguments comme --epochs, --batch-size etc. sont lus depuis config.yaml par le script lui-même
    ]

    # Ajouter l'option pour la normalisation par batch si elle n'est pas dans config.yaml
    # (Supposons qu'elle n'y est pas pour cet exemple, mais à vérifier)
    # cmd.append("--use-batch-norm") # Commenté car non reconnu par le script cible

    # Exu00e9cuter la commande
    logger.info(f"Exu00e9cution de la commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("Entrau00eenement du modu00e8le terminu00e9")


def run_backtesting():
    """
    Exu00e9cute le backtesting sur le modu00e8le entrau00eenu00e9.
    """
    logger.info("u00c9tape 4: Backtesting du modu00e8le")

    # Construire la commande
    cmd = [
        "python",
        "predict_with_reasoning.py",
        "--data-path",
        "data/enriched/enriched_dataset.parquet",
        "--output-dir",
        "results/predictions",
    ]

    # Exu00e9cuter la commande
    logger.info(f"Exu00e9cution de la commande: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    logger.info("Backtesting terminu00e9")


def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()

    # Configurer l'environnement
    setup_environment()

    # 1. Collecte des donnu00e9es enrichies (u00e9tape 3 de votre ordre de prioritu00e9)
    if not args.skip_data_collection:
        collect_enriched_data(args)
    else:
        logger.info("u00c9tape 1 (Collecte des donnu00e9es) ignoru00e9e")

    # 2. Pru00e9paration du dataset (u00e9tape 1 de votre ordre de prioritu00e9)
    if not args.skip_data_preparation:
        prepare_dataset()
    else:
        logger.info("u00c9tape 2 (Pru00e9paration du dataset) ignoru00e9e")

    # 3. Entrau00eenement du modu00e8le (u00e9tape 2 de votre ordre de prioritu00e9)
    if not args.skip_model_training:
        train_model(args)
    else:
        logger.info("u00c9tape 3 (Entrau00eenement du modu00e8le) ignoru00e9e")

    # 4. Backtesting (u00e9tape 5 de votre ordre de prioritu00e9)
    if not args.skip_backtesting:
        run_backtesting()
    else:
        logger.info("u00c9tape 4 (Backtesting) ignoru00e9e")

    logger.info("Pipeline complet exu00e9cutu00e9 avec succu00e8s")
    logger.info("Vous pouvez maintenant consulter les ru00e9sultats dans le ru00e9pertoire 'results/'")


async def run_live_trading_session(args):
    """Lance une session de trading en direct avec LiveExecutor."""
    logger.info("Initialisation du mode de trading en direct...")
    try:
        # Importer LiveExecutor ici pour éviter les imports circulaires potentiels au niveau du module
        from live.executor import LiveExecutor
        from utils.config_loader import load_config  # S'assurer que load_config est accessible

        config_path = Path(os.getcwd()) / "config" / "config.yaml"  # Assumer que config.yaml est à la racine

        # Utiliser l'exchange_id de la config par défaut, ou un override si nécessaire
        # Pour l'instant, on se base sur la config par défaut de LiveExecutor
        executor = LiveExecutor(
            config_path=str(config_path), dry_run=args.dry_run_live
        )  # Ajouter un arg --dry-run-live

        logger.info("Démarrage de la session LiveExecutor...")
        await executor.run()  # Exécute la boucle principale de trading
    except ImportError as e:
        logger.error(f"Erreur d'importation pour le trading en direct (LiveExecutor ou config_loader): {e}")
        logger.error("Assurez-vous que les modules sont correctement installés et accessibles.")
    except Exception as e:
        logger.exception(f"Erreur inattendue lors du lancement du trading en direct: {e}")
    finally:
        if "executor" in locals() and hasattr(executor, "close"):
            logger.info("Fermeture de la session LiveExecutor...")
            await executor.close()


def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()

    # Ajouter un argument pour le mode live et dry_run_live s'ils n'existent pas
    if not hasattr(args, "live"):
        args.live = False  # Par défaut, ne pas lancer en mode live
    if not hasattr(args, "dry_run_live"):
        args.dry_run_live = True  # Par défaut, dry_run pour le mode live

    if args.live:
        asyncio.run(run_live_trading_session(args))
    else:
        # Configurer l'environnement
        setup_environment()

        # 1. Collecte des donnu00e9es enrichies (u00e9tape 3 de votre ordre de prioritu00e9)
        if not args.skip_data_collection:
            collect_enriched_data(args)
        else:
            logger.info("u00c9tape 1 (Collecte des donnu00e9es) ignoru00e9e")

        # 2. Pru00e9paration du dataset (u00e9tape 1 de votre ordre de prioritu00e9)
        if not args.skip_data_preparation:
            prepare_dataset()
        else:
            logger.info("u00c9tape 2 (Pru00e9paration du dataset) ignoru00e9e")

        # 3. Entrau00eenement du modu00e8le (u00e9tape 2 de votre ordre de prioritu00e9)
        if not args.skip_model_training:
            train_model(args)
        else:
            logger.info("u00c9tape 3 (Entrau00eenement du modu00e8le) ignoru00e9e")

        # 4. Backtesting (u00e9tape 5 de votre ordre de prioritu00e9)
        if not args.skip_backtesting:
            run_backtesting()
        else:
            logger.info("u00c9tape 4 (Backtesting) ignoru00e9e")

        logger.info("Pipeline complet exu00e9cutu00e9 avec succu00e8s")
        logger.info("Vous pouvez maintenant consulter les ru00e9sultats dans le ru00e9pertoire 'results/'")


if __name__ == "__main__":
    # Pour exécuter le pipeline standard : python run_enhanced_pipeline.py
    # Pour exécuter le trading live (simulé par défaut) : python run_enhanced_pipeline.py --live
    # Pour exécuter le trading live réel : python run_enhanced_pipeline.py --live --no-dry-run-live (nécessite d'ajouter --no-dry-run-live à parse_args)

    # Ajout de l'argument --live et --dry-run-live au parser
    # Cela devrait être fait dans parse_args, mais pour l'exemple, on le fait ici si ce n'est pas déjà fait
    # de manière plus propre dans parse_args.

    # Modifier parse_args pour inclure ces options:
    # parser.add_argument("--live", action="store_true", help="Lancer le mode de trading en direct")
    # parser.add_argument("--dry-run-live", action="store_true", default=True, help="Exécuter le trading en direct en mode dry_run (pas d'ordres réels)")
    # parser.add_argument("--no-dry-run-live", dest='dry_run_live', action='store_false', help="Désactiver le dry_run pour le trading en direct (ordres réels)")

    # Pour que cela fonctionne, il faut modifier la fonction parse_args pour inclure ces arguments.
    # Je vais supposer pour l'instant que l'utilisateur modifiera parse_args ou passera ces arguments.
    # Une solution plus propre serait d'ajouter ces arguments dans la fonction parse_args.

    # Pour l'instant, je vais ajouter un import asyncio au début du fichier.
    import asyncio  # S'assurer qu'asyncio est importé

    # La logique pour vérifier current_args.live est maintenant dans la fonction main()
    # qui reçoit l'objet args correctement parsé.
    main()
