import telegram
import asyncio
import os
from datetime import datetime  # Ajout de l'import manquant
import logging  # Ajout de l'import manquant
from dotenv import load_dotenv
import json  # Ajout pour lire le statut
from pathlib import Path  # Ajout pour gérer le chemin du fichier statut

# Charger les variables d'environnement si .env existe
load_dotenv()

# Configuration du logger pour ce module
logger = logging.getLogger(__name__)
if not logger.handlers:  # Éviter d'ajouter plusieurs handlers si le module est rechargé
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")  # Peut être un ID de groupe ou d'utilisateur

if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
    print(
        "WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. Please set them in your .env file or environment variables."
    )

bot = None
if TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    except Exception as e:
        print(f"Error initializing Telegram bot: {e}")
        bot = None


async def send_telegram_message(chat_id: str, message: str):
    """
    Envoie un message à un chat ID Telegram spécifié.
    """
    if not bot:
        print("Telegram bot not initialized. Message not sent.")
        print(f"Intended message to {chat_id}: {message}")
        return False
    if not chat_id or chat_id == "YOUR_TELEGRAM_CHAT_ID":
        print(f"Invalid TELEGRAM_CHAT_ID: {chat_id}. Message not sent.")
        print(f"Intended message: {message}")
        return False

    try:
        await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        print(f"Message sent to Telegram chat ID {chat_id}: {message}")
        return True
    except Exception as e:
        print(f"Error sending Telegram message to {chat_id}: {e}")
        return False


def notify_trade_sync(signal: str, price: float, reasoning: str, chat_id: str = TELEGRAM_CHAT_ID):
    """
    Wrapper synchrone pour envoyer une notification de trade.
    Utilise asyncio.run pour appeler la fonction asynchrone.
    """
    message = f"*Trade Alert - Morningstar*\n\n*Signal:* {signal}\n*Price:* ${price:.2f}\n*Reasoning:* {reasoning}"
    try:
        asyncio.run(send_telegram_message(chat_id, message))
    except RuntimeError as e:
        # Si une boucle d'événements est déjà en cours (par exemple, dans un notebook Jupyter ou un autre framework async)
        # essayer de créer une nouvelle boucle ou d'utiliser nest_asyncio si disponible.
        # Pour une simple CLI, cela devrait être moins problématique.
        print(f"RuntimeError calling asyncio.run (possibly due to existing event loop): {e}")
        # Tentative de contournement simple pour les environnements non-async stricts
        # Dans un environnement de production, une meilleure gestion de la boucle d'événements serait nécessaire.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(send_telegram_message(chat_id, message))
        print("Telegram message sent using a new event loop.")


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Testing Telegram bot notification...")
    if (
        TELEGRAM_BOT_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN"
        and TELEGRAM_CHAT_ID != "YOUR_TELEGRAM_CHAT_ID"
        and bot is not None
    ):
        test_signal = "BUY"
        test_price = 28500.50
        test_reasoning = "RSI oversold, MACD crossover, strong support at $28k."
        # notify_trade_sync(test_signal, test_price, test_reasoning) # Commenté pour éviter l'envoi auto au lancement

        # Test avec un message simple
        # asyncio.run(send_telegram_message(TELEGRAM_CHAT_ID, "Hello from Morningstar Bot! Test message."))
        print("Telegram bot initialized. Run with command handlers to interact.")
    else:
        print("Skipping test notification as bot token or chat ID is not configured, or bot failed to initialize.")

# --- Command Handlers ---
# Pour utiliser les handlers, ce script devrait être exécuté en continu.
# L'intégration avec predict_with_reasoning.py est pour l'envoi de notifications sortantes.
# Pour les commandes entrantes, le bot doit tourner.

from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters as Filters
from telegram import Update

# Importer les dépendances nécessaires pour /explain
from predict_with_reasoning import explain_signal, preprocess_data # Fonction utilitaire que nous avons préparée
from config.config import Config # Pour passer cfg à explain_signal
import pandas as pd # Pour créer un DataFrame factice pour fetch_signal_data
from typing import Optional, List # Ajout pour les type hints

# Placeholder pour le ReasoningModule - N'est plus directement utilisé par les commandes
# L'interaction avec le modèle de raisonnement se fait via explain_signal
# from model.reasoning.reasoning_module import ReasoningModule
# reasoning_module = None # Supprimé car explain_signal gère le modèle

bot_state = {
    "last_signal_time": None,
    "current_balance": 10000.00,  # USD factice
    "daily_pnl": 0.0,  # USD factice
    "active_trades": {},  # ex: {"BTC/USDT": {"entry_price": 30000, "size": 0.1}}
}
# Charger les IDs de chat autorisés depuis les variables d'environnement
# Exemple: ALLOWED_TELEGRAM_CHAT_IDS="12345678,98765432"
ALLOWED_CHAT_IDS_STR = os.getenv("ALLOWED_TELEGRAM_CHAT_IDS", "")
ALLOWED_CHAT_IDS = [int(chat_id.strip()) for chat_id in ALLOWED_CHAT_IDS_STR.split(",") if chat_id.strip().isdigit()]

if not ALLOWED_CHAT_IDS:
    print(
        "WARNING: ALLOWED_TELEGRAM_CHAT_IDS is not set or empty. The bot will respond to all users if chat_id specific checks are not implemented per command."
    )

# --- Décorateur pour vérifier les utilisateurs autorisés ---
from functools import wraps


def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id  # Peut aussi être un ID de groupe

        # Autoriser si l'ID de l'utilisateur OU l'ID du chat est dans la liste blanche
        # ou si la liste blanche est vide (mode "ouvert" - non recommandé en production)
        if not ALLOWED_CHAT_IDS or user_id in ALLOWED_CHAT_IDS or chat_id in ALLOWED_CHAT_IDS:
            return await func(update, context, *args, **kwargs)
        else:
            print(f"Accès non autorisé refusé pour user_id: {user_id}, chat_id: {chat_id}")
            await update.message.reply_text("Désolé, vous n'êtes pas autorisé à utiliser cette commande.")
            return

    return wrapped


@restricted
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envoie un message lorsque la commande /start est émise."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Salut {user.mention_html()}! Je suis le bot Morningstar. Envoyez /help pour voir les commandes.",
    )


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envoie un message lorsque la commande /help est émise."""
    help_text = (
        "Commandes disponibles:\n"
        "/start - Démarrer le bot\n"
        "/help - Afficher ce message d'aide\n"
        "/status - Afficher le statut actuel du robot et les performances (factice)\n"
        "/explain <ID_TRADE_OU_SIGNAL> - Expliquer un trade ou signal (factice)\n"
        "/override <BUY|SELL|HOLD> <SYMBOL> [PRIX] - Forcer une action de trading (factice)"
    )
    await update.message.reply_text(help_text)


@restricted
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Affiche le statut du robot en lisant le fichier live_status.json."""
    status_file_path_str = os.getenv("LIVE_STATUS_FILE_PATH", "live_status.json")  # Lire depuis env ou défaut
    status_file_path = Path(status_file_path_str)
    current_time_req = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_message = f"*Statut Morningstar Bot - {current_time_req}*\n\n"

    if status_file_path.exists():
        try:
            with open(status_file_path, "r") as f:
                live_status = json.load(f)

            last_update_ts = live_status.get("timestamp")
            last_update_str = (
                datetime.fromtimestamp(last_update_ts).strftime("%Y-%m-%d %H:%M:%S") if last_update_ts else "N/A"
            )

            symbol = live_status.get("symbol", "N/A")
            position_side = live_status.get("position_side", "FLAT")
            entry_price = live_status.get("entry_price", 0.0)
            current_pos_size = live_status.get("current_position_size", 0.0)

            balance_data = live_status.get("last_known_balance", {})
            quote_currency = symbol.split("/")[-1] if symbol != "N/A" else "USDT"
            free_balance = balance_data.get(quote_currency, {}).get("free", "N/A")
            total_balance = balance_data.get(quote_currency, {}).get("total", "N/A")

            trading_active = "✅ Actif" if live_status.get("trading_active", False) else "❌ Inactif"
            consecutive_errors = live_status.get("consecutive_errors", 0)
            current_pnl = live_status.get("current_pnl", 0.0)

            status_message += (
                f"🕒 *Dernière mise à jour du statut:* {last_update_str}\n"
                f"⚙️ *Trading:* {trading_active}\n"
                f"📉 *Symbole:* {symbol}\n"
                f"🧭 *Position:* {position_side if position_side else 'FLAT'}\n"
                f" Entrée *Prix d'entrée:* {entry_price:.4f} (si en position)\n"  # Corrigé " एंट्री " en " Entrée "
                f"📏 *Taille actuelle:* {current_pos_size:.6f}\n"
                f"💹 *PnL Cumulé:* {current_pnl:.2f} {quote_currency}\n"  # Ajout du PnL
                f"💰 *Solde ({quote_currency}):* Libre={free_balance}, Total={total_balance}\n"
                f"⚠️ *Erreurs consécutives:* {consecutive_errors}\n"
            )
            if live_status.get("error_pause_active", False):
                status_message += "⏸️ *PAUSE ERREUR ACTIVE*\n"

        except json.JSONDecodeError:
            status_message += "Impossible de lire le fichier de statut (format JSON invalide).\n"
        except Exception as e:
            status_message += f"Erreur lors de la lecture du statut: {str(e)}\n"
    else:
        status_message += "Fichier de statut `live_status.json` non trouvé. Le moteur de trading n'est peut-être pas en cours d'exécution ou le chemin est incorrect.\n"
        status_message += f"Chemin attendu: {status_file_path.resolve()}\n"

    status_message += "\nRobot en attente de signaux (si actif)."
    await update.message.reply_text(status_message, parse_mode="Markdown")


@restricted
async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explique un trade ou un signal (logique factice)."""
    args = context.args
    if not args:
        await update.message.reply_text(
            "Veuillez fournir un ID de trade/signal. Usage: `/explain <ID_DU_SIGNAL>`", parse_mode="Markdown"
        )
        return
    signal_id = args[0]

    # Fonctions pour charger les données nécessaires à /explain
    def fetch_signal_data(s_id: str, cache_file_path: str = "data/cache/signal_cache.parquet") -> Optional[pd.DataFrame]:
        """
        Récupère les données brutes pour un signal_id depuis le cache.
        Retourne un DataFrame d'une ligne ou None si non trouvé.
        """
        cache_path = Path(cache_file_path)
        if not cache_path.exists():
            logger.error(f"Fichier cache de signaux non trouvé: {cache_path}")
            return None
        try:
            df_cache = pd.read_parquet(cache_path)
            signal_row_df = df_cache[df_cache['signal_id'] == s_id]
            if signal_row_df.empty:
                logger.warning(f"Signal ID {s_id} non trouvé dans le cache {cache_path}.")
                return None
            # Retourner un DataFrame contenant la ligne (ou les lignes si plusieurs correspondances, bien que signal_id soit supposé unique)
            return signal_row_df.copy() # Retourner une copie pour éviter SettingWithCopyWarning
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du cache de signaux pour {s_id}: {e}")
            return None

    def load_feature_names(names_file_path: str = "outputs/enhanced/feature_names.json") -> List[str]:
        """
        Charge la liste des noms de features depuis un fichier JSON.
        """
        names_path = Path(names_file_path)
        if not names_path.exists():
            logger.error(f"Fichier des noms de features non trouvé: {names_path}")
            return [] # Retourner une liste vide en cas d'erreur
        try:
            with open(names_path, 'r') as f:
                feature_names = json.load(f)
            if not isinstance(feature_names, list):
                logger.error(f"Le contenu de {names_path} n'est pas une liste.")
                return []
            return feature_names
        except Exception as e:
            logger.error(f"Erreur lors du chargement des noms de features: {e}")
            return []

    try:
        logger.info(f"Commande /explain reçue pour signal_id: {signal_id}")
        cfg_instance = Config()
        
        signal_cache_file = cfg_instance.get_config("paths.signal_cache_file", "data/cache/signal_cache.parquet")
        feature_names_file = cfg_instance.get_config("paths.feature_names_file", "outputs/enhanced/feature_names.json")
        logger.info(f"Utilisation du cache de signaux: {signal_cache_file}")
        logger.info(f"Utilisation du fichier de noms de features: {feature_names_file}")

        signal_data_df = fetch_signal_data(signal_id, signal_cache_file)
        
        if signal_data_df is None or signal_data_df.empty:
            logger.warning(f"Aucune donnée trouvée pour signal_id {signal_id} dans {signal_cache_file}.")
            await update.message.reply_text(f"Désolé, impossible de trouver les données pour le signal ID: {signal_id}")
            return
        logger.info(f"Données pour signal_id {signal_id} chargées, {len(signal_data_df)} ligne(s).")

        feature_names = load_feature_names(feature_names_file)
        if not feature_names:
            logger.error(f"La liste des noms de features de {feature_names_file} est vide ou n'a pas pu être chargée.")
            await update.message.reply_text("Erreur: La liste des noms de features n'a pas pu être chargée. Explication impossible.")
            return
        logger.info(f"{len(feature_names)} noms de features chargés depuis {feature_names_file}.")
            
        logger.info(f"Appel de explain_signal pour signal_id: {signal_id}")
        explanation_text = await asyncio.to_thread(
            explain_signal, # C'est une fonction synchrone
            signal_data_df, 
            cfg_instance, 
            feature_names 
        )
        logger.info(f"Explication générée pour {signal_id}: {explanation_text[:200]}...") # Loggue le début de l'explication
        
        # Tronquer si trop long pour Telegram
        max_length = 4090 # Un peu moins que 4096 pour la marge
        if len(explanation_text) > max_length:
            explanation_text = explanation_text[:max_length] + "\n\n[...message tronqué...]"
            
        await update.message.reply_text(explanation_text, parse_mode="Markdown")

    except FileNotFoundError as e: # Si le modèle ou les données ne sont pas trouvés par explain_signal
        logger.error(f"Erreur de fichier dans /explain pour {signal_id}: {e}")
        await update.message.reply_text(f"Erreur: Fichier nécessaire non trouvé pour générer l'explication.")
    except Exception as e:
        logger.error(f"Erreur inattendue dans /explain pour {signal_id}: {e}", exc_info=True)
        await update.message.reply_text(f"Une erreur est survenue lors de la génération de l'explication.")


@restricted
async def override_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forcer une action de trading (logique factice)."""
    args = context.args
    if len(args) < 2 or args[0].upper() not in ["BUY", "SELL", "HOLD"]:
        await update.message.reply_text(
            "Usage: `/override <BUY|SELL|HOLD> <SYMBOLE> [PRIX_OPTIONNEL]`", parse_mode="Markdown"
        )
        return

    action = args[0].upper()
    symbol = args[1].upper()
    price_str = args[2] if len(args) > 2 else None

    # Validation basique (à étendre)
    # Exemple: vérifier si le symbole est connu, si le prix est valide
    valid_symbols = ["BTC/USDT", "ETH/USDT"]  # Exemple
    if symbol not in valid_symbols:
        await update.message.reply_text(f"Symbole inconnu: {symbol}. Symboles valides: {', '.join(valid_symbols)}")
        return

    price = None
    if price_str:
        try:
            price = float(price_str)
            if price <= 0:
                raise ValueError("Le prix doit être positif.")
        except ValueError:
            await update.message.reply_text(f"Prix invalide: {price_str}. Doit être un nombre positif.")
            return

    # Logique factice pour traiter l'override
    # Dans une vraie application, cela interagirait avec le moteur de trading.
    log_message = f"Override reçu: {action} {symbol}"
    if price:
        log_message += f" @ {price:.2f}"

    # Simuler l'enregistrement de la commande
    # db_store_override_command(user_id=update.effective_user.id, command_details=log_message)

    bot_state["last_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Mettre à jour pour /status

    # --- Logique d'appel à LiveExecutor (nécessite un mécanisme IPC) ---
    # Ceci est un placeholder. Dans une application réelle, il faudrait un moyen
    # pour que le bot (qui tourne dans son propre processus/thread) communique
    # avec l'instance de LiveExecutor (qui tourne dans un autre).
    # Options: API REST locale, RPC, file de messages (RabbitMQ, Redis), signaux OS, etc.

    # Exemple de ce que l'on voudrait faire (si executor était accessible directement):
    # if context.bot_data.get('live_executor_instance'):
    #     executor = context.bot_data['live_executor_instance']
    #     success, message = await executor.process_override_command(action, symbol, price)
    #     if success:
    #         response_text = f"✅ Commande d'override traitée par LiveExecutor: {message}"
    #     else:
    #         response_text = f"⚠️ Erreur lors du traitement de l'override par LiveExecutor: {message}"
    # else:
    #     response_text = f"⚠️ LiveExecutor non accessible. Commande d'override enregistrée : {log_message}\nElle sera appliquée manuellement ou au prochain cycle (simulation)."

    # Pour l'instant, on simule une réponse positive.
    response_text = f"✅ Commande d'override '{log_message}' reçue et transmise (simulation).\nLe moteur de trading tentera de l'appliquer."

    await update.message.reply_text(response_text, parse_mode="Markdown")


# --- Gestionnaire d'erreurs global (optionnel mais recommandé) ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Logge les erreurs causées par les Updates."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    # Optionnel: Notifier l'utilisateur ou un admin d'une erreur
    if isinstance(update, Update) and update.effective_chat:
        try:
            await update.effective_chat.send_message(
                "Oups! Une erreur est survenue lors du traitement de votre demande. L'équipe technique a été notifiée."
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du message d'erreur à l'utilisateur: {e}")
    # Notifier un admin (si configuré)
    # ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
    # if ADMIN_CHAT_ID:
    #     await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Erreur dans le bot: {context.error}")


def run_bot():
    """Démarre le bot Telegram pour écouter les commandes."""
    if not bot:  # bot est l'instance de telegram.Bot
        print(
            "Le bot Telegram n'a pas pu être initialisé (token manquant ou invalide). Impossible de démarrer les handlers."
        )
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Enregistrer les handlers de commandes
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("explain", explain_command))
    application.add_handler(CommandHandler("override", override_command))

    print("Bot Telegram démarré et à l'écoute des commandes...")
    # Démarrer le bot (bloquant jusqu'à Ctrl-C)
    application.run_polling()


if __name__ == "__main__":
    # L'exemple d'utilisation pour notify_trade_sync est conservé ci-dessus.
    # Pour démarrer le bot en mode écoute, décommentez la ligne suivante:
    # run_bot()
    # Note: Assurez-vous que TELEGRAM_BOT_TOKEN est correctement configuré.
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("Le token du bot Telegram n'est pas configuré. Le bot ne démarrera pas en mode écoute.")
        print("Pour tester l'envoi de notifications, configurez TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID.")
    else:
        # Si on exécute ce fichier directement, on peut vouloir démarrer le bot en écoute.
        # Cependant, pour l'intégration avec d'autres scripts, seul notify_trade_sync sera importé.
        print(
            "Ce script peut être exécuté pour démarrer le bot en mode écoute (décommentez run_bot() dans if __name__ == '__main__')."
        )
        print("Pour l'instant, il ne fait que tester l'initialisation.")
        # Pour un test rapide de l'envoi (nécessite aussi TELEGRAM_CHAT_ID):
        # if TELEGRAM_CHAT_ID != "YOUR_TELEGRAM_CHAT_ID":
        #     asyncio.run(send_telegram_message(TELEGRAM_CHAT_ID, "Bot script started and can send messages."))

        # Décommentez pour lancer le bot en mode écoute lors de l'exécution directe du fichier
        # print("Pour démarrer le bot en mode écoute, décommentez 'run_bot()' dans la section __main__ de telegram_bot.py")
        run_bot()  # Décommentez pour tester les handlers
        # pass  # Ne rien faire par défaut lors de l'exécution directe, sauf si run_bot() est décommenté.
