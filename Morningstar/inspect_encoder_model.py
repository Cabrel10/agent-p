import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Peut aider pour le chargement initial
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

# Assure-toi que le chemin vers monolith_implementation est correct
try:
    from monolith_implementation.monolith_model import TransformerBlock
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from monolith_implementation.monolith_model import TransformerBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inspect_encoder")

def inspect_model(model_path):
    logger.info(f"Tentative de chargement du modèle depuis: {model_path}")
    custom_objects = {'TransformerBlock': TransformerBlock}
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        logger.info("Modèle chargé avec succès.")
    except Exception as e:
        logger.exception(f"Erreur lors du chargement du modèle: {e}")
        return

    logger.info("\n--- Résumé du Modèle ---")
    model.summary(print_fn=logger.info)

    logger.info("\n--- Attributs de Sortie ---")
    logger.info(f"model.output_names: {getattr(model, 'output_names', 'Non trouvé')}")
    
    outputs_attr = getattr(model, 'outputs', 'Non trouvé')
    logger.info(f"model.outputs (type): {type(outputs_attr)}")
    if isinstance(outputs_attr, list):
        logger.info(f"model.outputs (longueur): {len(outputs_attr)}")
        for i, out_tensor in enumerate(outputs_attr):
            logger.info(f"  Sortie {i}: name='{getattr(out_tensor, 'name', 'N/A')}', shape={getattr(out_tensor, 'shape', 'N/A')}")
    else:
        logger.info(f"model.outputs: {outputs_attr}")
        
    logger.info(f"model.output_shape: {getattr(model, 'output_shape', 'Non trouvé')}")

    # Optionnel : tentative de prédiction factice
    # try:
    #     dummy_input = ... # créer des données factices correspondant aux entrées du modèle
    #     predictions = model.predict(dummy_input)
    #     logger.info(f"Prédiction factice réussie. Type de sortie: {type(predictions)}")
    # except Exception as e:
    #     logger.exception(f"Erreur lors de la prédiction factice: {e}")

if __name__ == "__main__":
    # Utilise le chemin vers le modèle que tu veux inspecter
    encoder_model_to_inspect = "models/sprint2_contrastive_encoder/contrastive_encoder_model_v2.keras"
    inspect_model(encoder_model_to_inspect)
