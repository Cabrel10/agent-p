import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from monolith_implementation.monolith_model import MonolithModel
import logging

# Configuration CPU/threads
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-only', action='store_true', help='Forcer l\'utilisation du CPU uniquement, même si un GPU est disponible')
args, unknown = parser.parse_known_args()

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
import tensorflow as tf
import numpy as np
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
np.set_printoptions(threshold=10000)
if args.cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("[INFO] GPU désactivé, calcul forcé sur CPU.")

# Configuration du modèle (à compléter si besoin avec d'autres hyperparams d'Optuna)
CONFIG = {
    "learning_rate": 0.00039497810376922887,
    "dropout_rate": 0.1173889599996675,
    "l2_reg": 1.866186463803556e-06,
    "dense_units": 64,
    "lstm_units": 128,
    "transformer_blocks": 2,
    "transformer_heads": 2,
    "transformer_ff_dim_factor": 2,
    "batch_size": 32,
    "use_batch_norm": False,
    "latent_dim": 112,
    "tech_input_shape": (43,),
    "reconstruction_target_dim": 43
}

# Définition des colonnes (adapter selon optuna_monolith_optimization.py)
BASE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'pair', 'timeframe', 'symbol']

# --- DÉTECTION DYNAMIQUE DES FEATURES TECHNIQUES ---
# À placer après le chargement du DataFrame df
# technical_cols = [c for c in df.columns if c not in BASE_COLS]
# (à utiliser dans la logique principale)

def load_data(data_path: str):
    """Charge le dataset complet."""
    df = pd.read_parquet(data_path)
    return df

def prepare_data(df: pd.DataFrame, technical_cols: list, mcp_cols: list, embedding_cols: list, instrument_col: str):
    """Prépare les données pour l'entraînement autoencodeur MonolithModel."""
    # Split chronologique 80/20
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()

    # Scalers
    tech_scaler = StandardScaler()
    df_train[technical_cols] = tech_scaler.fit_transform(df_train[technical_cols])
    df_val[technical_cols] = tech_scaler.transform(df_val[technical_cols])

    mcp_scaler = None
    if mcp_cols and len(mcp_cols) > 0:
        mcp_scaler = StandardScaler()
        df_train[mcp_cols] = mcp_scaler.fit_transform(df_train[mcp_cols])
        df_val[mcp_cols] = mcp_scaler.transform(df_val[mcp_cols])

    # Instrument map
    instrument_map = {instrument: i for i, instrument in enumerate(df_train[instrument_col].unique())}

    def create_X_y_dict(df_subset, instrument_map_subset, tech_cols_subset, mcp_cols_subset, embedding_cols_subset, instrument_col_subset):
        X_dict = {}
        y_dict = {}

        X_dict['technical_input'] = df_subset[tech_cols_subset].values

        # Embeddings : gérer JSON ou tableau
        if embedding_cols_subset and len(embedding_cols_subset) > 0:
            emb_col = embedding_cols_subset[0]
            if isinstance(df_subset[emb_col].iloc[0], str):
                # JSON string
                X_dict['embeddings_input'] = np.stack(df_subset[emb_col].apply(lambda x: np.array(json.loads(x))).values)
            else:
                X_dict['embeddings_input'] = np.stack(df_subset[emb_col].values)
        else:
            X_dict['embeddings_input'] = None

        if mcp_cols_subset and len(mcp_cols_subset) > 0:
            X_dict['mcp_input'] = df_subset[mcp_cols_subset].values
        else:
            X_dict['mcp_input'] = None

        X_dict['instrument_input'] = df_subset[instrument_col_subset].map(instrument_map_subset).fillna(0).astype(int).values

        y_dict['reconstruction_output'] = X_dict['technical_input']
        # Ajout d'une clé 'projection' factice pour satisfaire la structure attendue par Keras
        y_dict['projection'] = np.zeros((X_dict['technical_input'].shape[0], 64), dtype=np.float32)

        return X_dict, y_dict

    X_train_dict, y_train_dict = create_X_y_dict(df_train, instrument_map, technical_cols, mcp_cols, embedding_cols, instrument_col)
    X_val_dict, y_val_dict = create_X_y_dict(df_val, instrument_map, technical_cols, mcp_cols, embedding_cols, instrument_col)

    return X_train_dict, y_train_dict, X_val_dict, y_val_dict, tech_scaler, mcp_scaler, instrument_map

def train_model(monolith_ae_model, X_train_dict, y_train_dict, X_val_dict, y_val_dict):
    """Entraîne le MonolithModel autoencodeur"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    history = monolith_ae_model.train(
        X_train_dict, y_train_dict,
        validation_data=(X_val_dict, y_val_dict),
        epochs=100,
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks
    )
    return history

def save_artifacts(monolith_ae_model, tech_scaler, mcp_scaler, history, model_config, data_processing_metadata_for_model):
    """Sauvegarde le modèle, les scalers et les métadonnées"""
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    # Modèle
    monolith_ae_model.save(os.path.join(CONFIG['model_dir'], "autoencoder_monolith_model.keras"))
    # Config
    with open(os.path.join(CONFIG['model_dir'], "model_config.json"), "w") as f:
        json.dump(model_config, f)
    # Metadata
    with open(os.path.join(CONFIG['model_dir'], "data_processing_metadata.json"), "w") as f:
        json.dump(data_processing_metadata_for_model, f)
    # Scalers
    with open(os.path.join(CONFIG['model_dir'], "scalers.pkl"), "wb") as f:
        pickle.dump({'tech_scaler': tech_scaler, 'mcp_scaler': mcp_scaler}, f)
    # Historique d'entraînement
def main():
    import os
    import json
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from tensorflow.keras import layers

    # 1) Chargement du dataset
    DATA_PATH = os.path.expanduser("ultimate/data/raw/market/features/dataset_train_rl.parquet")
    df = pd.read_parquet(DATA_PATH)

    # 2) Sélection des features techniques uniquement
    BASE_COLS = ['timestamp','open','high','low','close','volume','pair','timeframe','symbol']
    tech_cols = [c for c in df.columns if c not in BASE_COLS]
    X = df[tech_cols].astype(np.float32).values  # shape (N, n_features)

    # 3) Construction de l’autoencodeur
    input_dim = X.shape[1]
    encoding_dim = 16
    input_layer = keras.Input(shape=(input_dim,), name="technical_input")
    encoded = layers.Dense(64, activation="relu")(input_layer)
    encoded = layers.Dense(encoding_dim, activation="relu")(encoded)
    decoded = layers.Dense(64, activation="relu")(encoded)
    decoded = layers.Dense(input_dim, activation="linear")(decoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name="AE_tech")
    encoder = keras.Model(inputs=input_layer, outputs=encoded, name="Encoder_tech")
    autoencoder.compile(optimizer="adam", loss="mse")

    print(f"Shape X for training: {X.shape}")
    autoencoder.fit(
        X, X,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
    )

    # 4) Sauvegarde du modèle et de la config
    MODEL_DIR = "models/tech_autoencoder"
    os.makedirs(MODEL_DIR, exist_ok=True)
    autoencoder.save(f"{MODEL_DIR}/autoencoder_model.keras", include_optimizer=False)
    encoder.save(f"{MODEL_DIR}/encoder_model.keras", include_optimizer=False)
    config = {
        "input_names": ["technical_input"],
        "technical_cols": tech_cols,
        "encoding_dim": encoding_dim,
        "input_dim": input_dim
    }
    with open(f"{MODEL_DIR}/model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Autoencodeur entraîné et sauvé dans {MODEL_DIR}")

    # Entraînement
    history = train_model(monolith_ae_model, X_train_dict, y_train_dict, X_val_dict, y_val_dict)


    # Sauvegarde des artefacts
    save_artifacts(autoencoder, encoder, input_dim, encoding_dim, tech_cols)

if __name__ == '__main__':
    main()
