import keras
import tensorflow as tf
from monolith_implementation.monolith_model import build_autoencoder_monolith_model, TransformerBlock, projection_zero, l2_normalize_fn

# 1) Construire un modèle dummy
def main():
    model = build_autoencoder_monolith_model(
        tech_input_shape=(43,),
        latent_dim=64,
        reconstruction_target_dim=43,
        # Ajoutez d'autres paramètres si nécessaire
    )
    model.compile(loss={
        "projection": projection_zero,
        "reconstruction_output": "mse"
    }, optimizer="adam")

    # 2) Sauvegarder
    model.save("tmp_encoder.keras")

    # 3) Recharger en précisant custom_objects
    loaded = keras.models.load_model(
        "tmp_encoder.keras",
        custom_objects={
            "Custom>TransformerBlock": TransformerBlock,
            "TransformerBlock": TransformerBlock,
            "projection_zero": projection_zero,
            "Custom>l2_normalize_fn": l2_normalize_fn,
            "l2_normalize_fn": l2_normalize_fn
        }
    )

    # 4) Vérifier un résumé
    loaded.summary()
    print("✅ Test save/load réussi.")

if __name__ == "__main__":
    main()
