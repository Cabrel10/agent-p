import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import keras
from monolith_implementation.monolith_model import TransformerBlock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resave_encoder")

def resave_model():
    original_model_path = "models/sprint2_contrastive_encoder/contrastive_encoder_model.keras"
    new_model_path = "models/sprint2_contrastive_encoder/contrastive_encoder_model_v2.keras"

    custom_objects = {'TransformerBlock': TransformerBlock}
    
    logger.info(f"Chargement du modèle original depuis {original_model_path}...")
    try:
        model = keras.models.load_model(original_model_path, custom_objects=custom_objects, compile=False)
        logger.info("Modèle original chargé avec succès.")
    except Exception as e:
        logger.exception(f"Erreur lors du chargement du modèle original: {e}")
        return

    logger.info(f"Re-sauvegarde du modèle vers {new_model_path}...")
    try:
        model.save(new_model_path)
        logger.info(f"Modèle re-sauvegardé avec succès à {new_model_path}.")
    except Exception as e:
        logger.exception(f"Erreur lors de la re-sauvegarde du modèle: {e}")
        return

if __name__ == "__main__":
    resave_model()
