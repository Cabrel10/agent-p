import os
import json
import pickle
import numpy as np
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_tech_autoencoder")

# Traitement des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Entraînement d'un autoencoder technique pour la compression des features")
parser.add_argument('--cpu-only', action='store_true', help='Forcer l\'utilisation du CPU uniquement')
parser.add_argument('--epochs', type=int, help='Nombre d\'époques d\'entraînement')
parser.add_argument('--batch-size', type=int, help='Taille du batch')
args = parser.parse_args()

# Gestion des ressources CPU/GPU
if args.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("GPU désactivé, calcul forcé sur CPU.")

# Chargement de la configuration
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis : {config_path}")
        return config
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {config_path}: {e}")
        return None

# Chemin vers la racine du projet et le fichier de configuration
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
config_path = script_dir / "config.yaml"

# Charger la configuration
config = load_config(config_path)
if not config:
    logger.critical("Impossible de charger la configuration. Arrêt du script.")
    exit(1)

# Récupérer les paramètres depuis la configuration
project_root = Path(config["project_root"])
DATA_PATH = project_root / config["paths"]["dataset_train_rl_legacy"]
MODEL_DIR = project_root / config["paths"]["tech_autoencoder_dir"]

# Récupérer les hyperparamètres d'entraînement
epochs = args.epochs if args.epochs else config["tech_autoencoder_train"]["epochs"]
batch_size = args.batch_size if args.batch_size else config["tech_autoencoder_train"]["batch_size"]
encoding_dim = config["tech_autoencoder_train"]["encoding_dim"]
validation_split = config["tech_autoencoder_train"]["validation_split"]

logger.info(f"Chargement des données depuis: {DATA_PATH}")
try:
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Données chargées avec succès: {df.shape} lignes, {df.columns.size} colonnes")
except FileNotFoundError:
    logger.warning(f"Fichier {DATA_PATH} non trouvé. Tentative avec les données fusionnées...")
    try:
        # Essayer avec le fichier de données fusionnées à la place
        DATA_PATH = project_root / config["paths"]["merged_features_file"]
        logger.info(f"Tentative avec le fichier: {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"Données chargées avec succès: {df.shape} lignes, {df.columns.size} colonnes")
    except FileNotFoundError:
        logger.critical(f"Aucun fichier de données trouvé. Vérifiez les chemins dans config.yaml")
        exit(1)

# Limiter la taille du dataset pour accélérer l'entraînement
max_rows = 50000  # Utiliser seulement les 50000 premières lignes
if len(df) > max_rows:
    logger.info(f"Réduction du dataset à {max_rows} lignes pour accélérer l'entraînement")
    df = df.head(max_rows)

# 2) Sélection des features techniques uniquement
BASE_COLS = ['timestamp','open','high','low','close','volume','pair','timeframe','symbol']
tech_cols = [c for c in df.columns if c not in BASE_COLS]
logger.info(f"Nombre de colonnes techniques: {len(tech_cols)}")
logger.info(f"Exemples de colonnes techniques: {tech_cols[:5]}")
X = df[tech_cols].astype(np.float32).values  # shape (N, n_features)

# 2b) Normalisation des features techniques
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 3) Construction de l'autoencodeur
input_dim = X.shape[1]  # nombre de features techniques
logger.info(f"Dimension d'entrée: {input_dim}, dimension encodée: {encoding_dim}")

input_layer = keras.Input(shape=(input_dim,), name="technical_input")
encoded = layers.Dense(64, activation="relu")(input_layer)
encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name="AE_tech")
encoder = keras.Model(inputs=input_layer, outputs=encoded, name="Encoder_tech")

autoencoder.compile(optimizer="adam", loss="mse")

# 4) Entraînement
logger.info(f"Shape X for training: {X_scaled.shape}")
logger.info(f"Démarrage de l'entraînement: {epochs} époques, batch size: {batch_size}")

# Limiter le nombre d'étapes pour les tests
steps_per_epoch = min(500, X_scaled.shape[0] // batch_size)
validation_steps = min(100, int(X_scaled.shape[0] * validation_split) // batch_size)
logger.info(f"Nombre d'étapes par époque limité à: {steps_per_epoch}, validation steps: {validation_steps}")

# Callbacks pour l'entraînement
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=validation_split,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Log des résultats d'entraînement
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
logger.info(f"Entraînement terminé. Loss final: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")

# 5) Sauvegarde du modèle, du scaler, et de la config
os.makedirs(MODEL_DIR, exist_ok=True)
logger.info(f"Sauvegarde des modèles dans: {MODEL_DIR}")

autoencoder_path = os.path.join(MODEL_DIR, "autoencoder_model.keras")
encoder_path = os.path.join(MODEL_DIR, "encoder_model.keras")

autoencoder.save(autoencoder_path, include_optimizer=False)
encoder.save(encoder_path, include_optimizer=False)
logger.info(f"Modèles sauvegardés: {autoencoder_path}, {encoder_path}")

# Sauvegarde du scaler
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_technical_light.pkl")
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
logger.info(f"Scaler sauvegardé: {SCALER_PATH}")

# Sauvegarde de l'historique d'entraînement
history_path = os.path.join(MODEL_DIR, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

# 6) Génération de model_config.json
model_config = {
    "input_names": ["technical_input"],
    "technical_cols": tech_cols,
    "encoding_dim": encoding_dim,
    "input_dim": input_dim,
    "scaler_path": config["paths"]["tech_autoencoder_scaler"],
    "date_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_params": {
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "final_loss": float(final_loss),
        "final_val_loss": float(final_val_loss)
    }
}

config_path = os.path.join(MODEL_DIR, "model_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)
logger.info(f"Configuration du modèle sauvegardée: {config_path}")

logger.info(f"✅ Autoencodeur entraîné, scaler sauvegardé, et config générée dans {MODEL_DIR}")
