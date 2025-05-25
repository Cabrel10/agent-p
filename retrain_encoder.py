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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retrain_encoder")

# Traitement des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Réentraînement d'un encodeur technique compatible avec l'agent RL")
parser.add_argument('--cpu-only', action='store_true', help='Forcer l\'utilisation du CPU uniquement')
parser.add_argument('--epochs', type=int, default=25, help='Nombre d\'époques d\'entraînement (recommandé: 20-30)')
parser.add_argument('--batch-size', type=int, default=64, help='Taille du batch')
parser.add_argument('--save-dir', type=str, default=None, help='Répertoire de sauvegarde personnalisé')
parser.add_argument('--encoding-dim', type=int, default=16, help='Dimension de l\'espace latent de l\'encodeur')
parser.add_argument('--samples', type=int, default=150000, help='Nombre d\'échantillons à utiliser (recommandé: 100000-200000)')
args = parser.parse_args()

# Gestion des ressources CPU/GPU
if args.cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("GPU désactivé, calcul forcé sur CPU.")

# Configuration de l'environnement Keras
# Désactivation de l'environnement legacy pour assurer la compatibilité avec TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "0"
logger.info("Configuration de l'environnement Keras pour compatibilité.")

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
DATA_PATH = project_root / config["paths"]["merged_features_file"]
MODEL_DIR = args.save_dir if args.save_dir else project_root / "models/retrained_encoder"

# Créer le répertoire de sauvegarde s'il n'existe pas
os.makedirs(MODEL_DIR, exist_ok=True)

# Récupérer les hyperparamètres d'entraînement
epochs = args.epochs
batch_size = args.batch_size
encoding_dim = args.encoding_dim
validation_split = config["tech_autoencoder_train"]["validation_split"]
max_samples = args.samples

logger.info(f"Chargement des données depuis: {DATA_PATH}")
try:
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Données chargées avec succès: {df.shape} lignes, {df.columns.size} colonnes")
except Exception as e:
    logger.critical(f"Erreur lors du chargement des données: {e}")
    exit(1)

# Limiter la taille du dataset pour accélérer l'entraînement
if len(df) > max_samples:
    logger.info(f"Échantillonnage de {max_samples} lignes parmi {len(df)} pour équilibrer diversité et temps d'entraînement")
    df = df.sample(n=max_samples, random_state=42)
    logger.info(f"Échantillonnage terminé: {df.shape}")

# Sélection des features techniques uniquement
BASE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
tech_cols = [c for c in df.columns if c not in BASE_COLS]
logger.info(f"Nombre de colonnes techniques: {len(tech_cols)}")
logger.info(f"Exemples de colonnes techniques: {tech_cols[:5]}")
X = df[tech_cols].astype(np.float32).values  # shape (N, n_features)

# Normalisation des features techniques
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
logger.info(f"Données normalisées: min={X_scaled.min()}, max={X_scaled.max()}")

# Construction de l'autoencodeur
input_dim = X.shape[1]  # nombre de features techniques
logger.info(f"Dimension d'entrée: {input_dim}, dimension encodée: {encoding_dim}")

input_layer = keras.Input(shape=(input_dim,), name="technical_input")
encoded = layers.Dense(64, activation="relu", name="encoder_dense1")(input_layer)
encoded = layers.Dense(encoding_dim, activation="relu", name="encoder_dense2")(encoded)

decoded = layers.Dense(64, activation="relu", name="decoder_dense1")(encoded)
decoded = layers.Dense(input_dim, activation="linear", name="decoder_output")(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name="AE_tech")
encoder = keras.Model(inputs=input_layer, outputs=encoded, name="Encoder_tech")

# Compile avec Adam
autoencoder.compile(optimizer="adam", loss="mse")

# Entraînement
logger.info(f"Shape X for training: {X_scaled.shape}")
logger.info(f"Démarrage de l'entraînement: {epochs} époques, batch size: {batch_size}")

# Callbacks pour l'entraînement
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_encoder_checkpoint.h5"), 
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    ),
]

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=validation_split,
    callbacks=callbacks,
    verbose=1
)

# Log des résultats d'entraînement
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
logger.info(f"Entraînement terminé. Loss final: {final_loss:.6f}, Val loss: {final_val_loss:.6f}")

# Sauvegarde du modèle, du scaler, et de la config
logger.info(f"Sauvegarde des modèles dans: {MODEL_DIR}")

autoencoder_path = os.path.join(MODEL_DIR, "autoencoder_model.h5")
encoder_path = os.path.join(MODEL_DIR, "encoder_model.h5")

try:
    # Sauvegarde au format HDF5 qui est plus robuste aux changements de version
    autoencoder.save(autoencoder_path, save_format='h5')
    encoder.save(encoder_path, save_format='h5')
    
    # Sauvegarde également au format keras pour compatibilité
    autoencoder_keras_path = os.path.join(MODEL_DIR, "autoencoder_model.keras")
    encoder_keras_path = os.path.join(MODEL_DIR, "encoder_model.keras")
    autoencoder.save(autoencoder_keras_path)
    encoder.save(encoder_keras_path)
    logger.info(f"Modèles sauvegardés en format dual (H5 et Keras)")
except Exception as e:
    logger.warning(f"Erreur lors de la sauvegarde standard: {e}")
    # Alternative: sauvegarder uniquement les poids
    autoencoder.save_weights(os.path.join(MODEL_DIR, "autoencoder_weights.h5"))
    encoder.save_weights(os.path.join(MODEL_DIR, "encoder_weights.h5"))
    logger.info(f"Sauvegarde des poids seulement: {os.path.join(MODEL_DIR, 'autoencoder_weights.h5')}")
logger.info(f"Modèles sauvegardés: {autoencoder_path}, {encoder_path}")

# Sauvegarde du scaler
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_technical.pkl")
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
logger.info(f"Scaler sauvegardé: {SCALER_PATH}")

# Sauvegarde de l'historique d'entraînement
history_path = os.path.join(MODEL_DIR, "training_history.json")
with open(history_path, "w") as f:
    json.dump(history.history, f)

# Génération de model_config.json
model_config = {
    "input_names": ["technical_input"],
    "technical_cols": tech_cols,
    "encoding_dim": encoding_dim,
    "input_dim": input_dim,
    "scaler_path": os.path.join(MODEL_DIR, "scaler_technical.pkl"),
    "model_path_h5": encoder_path,  # Chemin vers la version HDF5
    "model_path_keras": os.path.join(MODEL_DIR, "encoder_model.keras"),  # Chemin keras pour compatibilité
    "date_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "keras_compatible_mode": True,
    "use_h5_format": True,  # Indiquer qu'on utilise le format HDF5
    "samples_used": max_samples,
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

logger.info(f"✅ Autoencodeur réentraîné sur {max_samples} échantillons, {epochs} époques")
logger.info(f"✅ Performances finales: loss={final_loss:.6f}, val_loss={final_val_loss:.6f}")
logger.info(f"✅ Modèle, scaler et config sauvegardés dans {MODEL_DIR}")
logger.info(f"✅ Pour utiliser cet encodeur, assurez-vous que retrained_encoder_model est configuré dans config.yaml")
logger.info(f"✅ L'encodeur est maintenant sauvegardé au format HDF5 (.h5) qui est plus robuste aux changements de version")