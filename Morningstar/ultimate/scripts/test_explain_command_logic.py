import asyncio
import logging
from unittest.mock import MagicMock
import pandas as pd

# Configure logging to see output from the modules
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__) # Initialisation du logger pour ce script

# Suppress TensorFlow warnings for this test script
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# Import the command from telegram_bot
# Ensure telegram_bot can be imported (sys.path or PYTHONPATH might be needed if run from a different dir)
try:
    from telegram_bot import explain_command
except ImportError:
    # Adjust path if necessary, assuming script is run from project root
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from telegram_bot import explain_command

# Dummy Update and Context classes (simplified from pytest-telegram-bot or similar)
class DummyUser:
    def __init__(self, user_id, first_name="TestUser"):
        self.id = user_id
        self.first_name = first_name
    def mention_html(self):
        return f"<a href='tg://user?id={self.id}'>{self.first_name}</a>"

class DummyChat:
    def __init__(self, chat_id):
        self.id = chat_id

class DummyMessage:
    def __init__(self, text, chat_id, user_id):
        self.text = text
        self.chat = DummyChat(chat_id)
        self._reply_text = "" # To store what reply_text was called with

    async def reply_text(self, text, parse_mode=None, **kwargs):
        logger.info(f"DummyMessage.reply_text called with: {text}")
        self._reply_text = text
        # In a real test, you might want to store parse_mode and kwargs too

class DummyUpdate:
    def __init__(self, message_text, chat_id, user_id, command_args=None):
        self.message = DummyMessage(message_text, chat_id, user_id)
        self.effective_user = DummyUser(user_id)
        self.effective_chat = self.message.chat # For convenience
        # context.args is derived from message.text by the CommandHandler
        # For direct call, we simulate it in DummyContext
        self._command_args = command_args if command_args is not None else []

class DummyContext:
    def __init__(self, command_args=None):
        self.args = command_args if command_args is not None else []
        self.bot = MagicMock() # Mock the bot object if any methods are called on it

async def test_explain_logic():
    logger.info("--- Début du test logique pour explain_command ---")
    
    # Utiliser le signal_id récupéré
    test_signal_id = "cfd7e823-d8c2-4cbf-9541-a701c303bbc9" 
    
    # Simuler un utilisateur et un chat
    test_user_id = 12345
    test_chat_id = 67890

    # Créer les objets Update et Context factices
    # Le message text n'est pas directement utilisé par explain_command, mais args oui.
    dummy_update = DummyUpdate(
        message_text=f"/explain {test_signal_id}",
        chat_id=test_chat_id,
        user_id=test_user_id,
        command_args=[test_signal_id] # Simule ce que CommandHandler ferait
    )
    dummy_context = DummyContext(command_args=[test_signal_id])

    # S'assurer que les fichiers nécessaires existent (cache, feature_names)
    # data/cache/signal_cache.parquet (créé)
    # outputs/enhanced/feature_names.json (créé par la dernière exécution de predict_with_reasoning)

    logger.info(f"Appel de explain_command avec signal_id: {test_signal_id}")
    await explain_command(dummy_update, dummy_context)

    logger.info("--- Fin du test logique pour explain_command ---")
    logger.info(f"Réponse simulée envoyée par le bot: {dummy_update.message._reply_text}")

if __name__ == "__main__":
    # Assurer que les variables d'environnement pour le bot ne sont PAS celles de production
    # pour éviter d'envoyer de vrais messages si send_telegram_message était appelé directement.
    # Cependant, explain_command appelle reply_text sur l'objet Update mocké.
    os.environ["TELEGRAM_BOT_TOKEN"] = "YOUR_TEST_TOKEN" # Placeholder
    os.environ["TELEGRAM_CHAT_ID"] = "YOUR_TEST_CHAT_ID"   # Placeholder
    
    asyncio.run(test_explain_logic())
