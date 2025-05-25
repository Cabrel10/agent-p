import tensorflow as tf
import numpy as np
from pathlib import Path
import os


def create_dummy_model(output_path_str: str):
    output_path = Path(output_path_str)
    # Définir une architecture de modèle minimale
    # Ajuster les shapes si nécessaire en fonction de ce que `prepare_features` produit.
    # D'après golden_backtest.parquet, nous avons 908 colonnes.
    # OHLCV + symbol = 6 colonnes.
    # feature_SMA_10, feature_RSI_14, feature_MACD_12_26_9, feature_MACD_signal_12_26_9, feature_MACD_hist_12_26_9 = 5 colonnes
    # bert_0 à bert_767 = 768 colonnes
    # mcp_0 à mcp_127 = 128 colonnes
    # Total features techniques = 5.
    # Total features = 6 (base) + 5 (tech) + 768 (llm) + 128 (mcp) = 907. Le script generate_golden_fixture.py dit 908 colonnes.
    # La colonne 'timestamp' est l'index et n'est pas une feature.
    # Les features techniques dans `prepare_features` sont calculées par `apply_feature_pipeline`.
    # Pour un test E2E simple, on va supposer que `technical_features.values` aura une certaine shape.
    # Le script `generate_golden_fixture.py` crée 5 features techniques + 768 LLM + 128 MCP.
    # `prepare_features` dans `run_backtest.py` utilise `apply_feature_pipeline` qui pourrait changer le nombre de features techniques.
    # Pour un modèle *factice*, la shape exacte des inputs techniques est moins critique tant qu'elle est cohérente.
    # Le `golden_backtest.parquet` a 908 colonnes. `timestamp` est une colonne, pas l'index dans le fichier.
    # `symbol` est une colonne.
    # `open, high, low, close, volume` = 5 colonnes.
    # `feature_SMA_10`, `feature_RSI_14`, `feature_MACD_12_26_9`, `feature_MACD_signal_12_26_9`, `feature_MACD_hist_12_26_9` = 5 colonnes.
    # `bert_0` ... `bert_767` = 768 colonnes.
    # `mcp_0` ... `mcp_127` = 128 colonnes.
    # Total = 1+1+5+5+768+128 = 908.
    # `prepare_features` dans `run_backtest.py` fait:
    #   `technical_features = df_features.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')`
    #   Donc, les features techniques pour le modèle seraient les 5 indicateurs + 768 LLM + 128 MCP + symbol (si pas drop).
    #   Si `apply_feature_pipeline` ne modifie pas `symbol` et ne la drop pas, elle sera incluse.
    #   Cependant, `morningstar_model.py` s'attend à des inputs séparés.
    #   Le modèle dans `run_backtest.py` est `morningstar_final.h5` qui a des inputs nommés.

    # Inputs du modèle Morningstar typique
    technical_input = tf.keras.Input(shape=(38,), name="technical_input")  # Shape typique pour les 38 indicateurs
    llm_input = tf.keras.Input(shape=(768,), name="llm_input")
    # instrument_input = tf.keras.Input(shape=(1,), name="instrument_input") # Si utilisé

    # Simuler une combinaison simple des inputs
    combined_tech_llm = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Dense(8, activation="relu")(technical_input),
            tf.keras.layers.Dense(8, activation="relu")(llm_input),
        ]
    )
    # combined_all = tf.keras.layers.concatenate([combined_tech_llm, instrument_input])
    # current_combined = combined_all
    current_combined = combined_tech_llm

    # Sorties (simulant signal et SL/TP)
    # Signal: -1 (Vente), 0 (Hold), 1 (Achat) -> 3 classes (Buy, Sell, Hold)
    # Dans le modèle original, c'est (Buy, Sell, Hold, SL, TP) ou des têtes séparées.
    # Ici, on simule les sorties attendues par `generate_signals` dans `run_backtest.py`
    # `signal_pred = predictions[0]` et `sl_tp_pred = predictions[1]`

    output_signal = tf.keras.layers.Dense(3, activation="softmax", name="signal_output")(current_combined)
    output_sl_tp = tf.keras.layers.Dense(2, activation="linear", name="sl_tp_output")(current_combined)  # SL, TP levels

    model = tf.keras.Model(
        inputs=[technical_input, llm_input],  # Ajouter instrument_input si utilisé
        outputs=[output_signal, output_sl_tp],
    )
    model.compile(optimizer="adam", loss="mse")  # Compiler est nécessaire pour sauvegarder

    # Sauvegarder le modèle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path_str)  # Keras préfère un str pour le chemin parfois
    print(f"Modèle factice sauvegardé dans : {output_path_str}")


if __name__ == "__main__":
    # Exemple d'utilisation (non exécuté lors de l'import)
    # Créer un répertoire temporaire pour le test du script lui-même
    temp_model_dir = Path("temp_dummy_model_output")
    temp_model_path = temp_model_dir / "dummy_model.h5"

    # Supprimer l'ancien modèle s'il existe pour éviter les erreurs de HDF5
    if temp_model_path.exists():
        os.remove(temp_model_path)

    create_dummy_model(str(temp_model_path))

    # Vérifier que le modèle peut être chargé
    try:
        loaded_model = tf.keras.models.load_model(str(temp_model_path))
        print("Modèle factice chargé avec succès pour vérification.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle factice pour vérification: {e}")
