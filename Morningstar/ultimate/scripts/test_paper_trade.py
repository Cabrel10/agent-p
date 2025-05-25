import asyncio
import os
from pathlib import Path
import sys

# Ajouter le répertoire racine du projet au PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from live.executor import LiveExecutor
from utils.config_loader import load_config
import logging
import ccxt

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main_test_paper_trade():
    """
    Teste la configuration des clés API et la capacité à placer un ordre "papier"
    sur un compte de testnet.
    """
    config_path = BASE_DIR / "config" / "config.yaml"
    main_config = load_config(config_path)

    # IMPORTANT: L'utilisateur doit configurer ces variables d'environnement
    # pour l'exchange de test (par exemple, Binance Testnet).
    # Exemple pour Binance Testnet:
    # export BINANCE_TESTNET_API_KEY="votre_clé_api_testnet"
    # export BINANCE_TESTNET_API_SECRET="votre_secret_api_testnet"

    # Surcharger l'exchange_id pour utiliser le testnet.
    # Assurez-vous que 'binance_testnet' (ou l'équivalent pour votre exchange)
    # est un ID d'exchange valide que ccxt peut reconnaître et qui est configuré
    # pour utiliser les clés API de testnet.
    # Souvent, les exchanges de testnet ont des ID spécifiques ou nécessitent
    # une option dans les paramètres de l'exchange.

    # Pour Binance, le mode testnet est activé via une option dans les paramètres de l'exchange.
    # L'ID de l'exchange reste 'binance'.
    exchange_id_for_test = "binance"  # Reste 'binance'

    # Mettre à jour la configuration pour le testnet
    # Cela suppose que votre config.yaml a une section pour 'binance_testnet'
    # ou que vous allez surcharger les options directement.
    # Pour Binance, on active le mode 'testnet'.

    # Créer une copie de la configuration pour la modifier sans affecter l'original
    test_config = main_config.copy()
    if "exchange_params" not in test_config:
        test_config["exchange_params"] = {}
    if exchange_id_for_test not in test_config["exchange_params"]:
        test_config["exchange_params"][exchange_id_for_test] = {}

    # Activer le mode testnet pour Binance
    # La clé 'testnet' ou 'options': {'defaultType': 'future', 'testnet': True} dépend de ccxt et de l'exchange.
    # Pour Binance Spot Testnet, c'est souvent géré par l'URL de base.
    # Pour ccxt, il faut parfois explicitement dire à l'instance d'utiliser le sandbox.
    # Le plus simple est de s'assurer que les clés API sont celles du testnet.
    # Et que l'instance ccxt est configurée pour le testnet si nécessaire.

    # Note: LiveExecutor essaie de charger les clés API basées sur l'exchange_id.
    # Donc, si exchange_id_for_test est 'binance', il cherchera BINANCE_API_KEY.
    # Assurez-vous que ces variables pointent vers vos clés TESTNET.

    logger.info(f"Configuration pour le test papier sur {exchange_id_for_test.upper()} Testnet.")
    logger.info("Assurez-vous que les variables d'environnement pour les clés API TESTNET sont définies:")
    logger.info(f"  - {exchange_id_for_test.upper()}_API_KEY")
    logger.info(f"  - {exchange_id_for_test.upper()}_API_SECRET")
    if exchange_id_for_test.upper() in ["KUCOIN", "BITGET"]:  # Exchanges nécessitant une passphrase
        logger.info(f"  - {exchange_id_for_test.upper()}_PASSPHRASE")

    try:
        # Initialiser LiveExecutor en mode NON dry_run pour interagir avec le testnet.
        # L'exchange_id est surchargé pour utiliser celui du testnet.
        executor = LiveExecutor(config_path=config_path, exchange_id_override=exchange_id_for_test, dry_run=False)

        # Activer le mode testnet sur l'instance ccxt si ce n'est pas déjà fait par la config
        if hasattr(executor.client, "set_sandbox_mode"):
            executor.client.set_sandbox_mode(True)
            logger.info("Mode Sandbox activé sur le client CCXT.")
        elif "test" in executor.client.urls:  # Autre méthode pour certains exchanges
            executor.client.urls["api"] = executor.client.urls["test"]
            logger.info(f"URL API client CCXT basculée vers Testnet: {executor.client.urls['api']}")
        else:
            logger.warning(
                "Impossible de confirmer l'activation du mode testnet via set_sandbox_mode ou changement d'URL. "
                "Assurez-vous que les clés API sont celles du Testnet."
            )

        # Récupérer le solde (devrait fonctionner avec les clés testnet)
        await executor._update_balance_sync()  # Utiliser la version interne pour le test
        logger.info(f"Solde Testnet récupéré: {executor.last_known_balance}")

        # Tenter de placer un petit ordre d'achat MARKET "papier" (réel sur testnet)
        symbol_to_test = executor.symbol  # Utiliser le symbole de la config
        # Définir une petite quantité pour le test, conforme aux minimums de l'exchange
        # Vous devrez peut-être ajuster 'amount_to_buy' en fonction de l'exchange et de la paire
        amount_to_buy = 0.001  # Exemple pour BTC/USDT ou ETH/USDT sur certains testnets

        if "BTC" in symbol_to_test:  # Ajuster pour BTC
            markets = await executor.client.load_markets()
            market = markets[symbol_to_test]
            amount_to_buy = market["limits"]["amount"]["min"] if market["limits"]["amount"]["min"] else 0.0001
            logger.info(
                f"Ajustement de la quantité d'achat pour {symbol_to_test} à {amount_to_buy} (min: {market['limits']['amount']['min']})"
            )

        logger.info(
            f"Tentative de placement d'un ordre MARKET BUY de {amount_to_buy} {symbol_to_test.split('/')[0]} sur {exchange_id_for_test.upper()} Testnet..."
        )

        order_info = await executor.client.create_market_buy_order(symbol_to_test, amount_to_buy)

        logger.info(f"Ordre 'papier' (réel sur testnet) placé avec succès !")
        logger.info(f"Détails de l'ordre: {order_info}")

    except ValueError as e:
        logger.error(f"Erreur de configuration ou de valeur: {e}")
    except ccxt.AuthenticationError as e:
        logger.error(f"Erreur d'authentification CCXT. Vérifiez vos clés API Testnet et permissions: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Erreur réseau CCXT: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Erreur d'exchange CCXT: {e}")
    except Exception as e:
        logger.exception(f"Erreur inattendue lors du test de paper trading: {e}")
    finally:
        if "executor" in locals() and executor.client:
            # Fermer la connexion WebSocket si elle a été ouverte par LiveExecutor (bien que non utilisée ici directement)
            await executor.close()  # S'assure que le client ws est fermé s'il a été initialisé


if __name__ == "__main__":
    asyncio.run(main_test_paper_trade())
