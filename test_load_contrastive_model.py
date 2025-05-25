from tensorflow import keras
from monolith_implementation.monolith_model import TransformerBlock

def l2_normalize_fn(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis=1)

if __name__ == "__main__":
    try:
        model = keras.models.load_model(
            "models/sprint2_contrastive_encoder/contrastive_encoder_model.keras",
            custom_objects={"l2_normalize_fn": l2_normalize_fn, "TransformerBlock": TransformerBlock},
            safe_mode=False
        )
        print("Modèle chargé avec succès !")
        model.summary()
    except Exception as e:
        import traceback
        print("Erreur lors du chargement du modèle :")
        traceback.print_exc()
