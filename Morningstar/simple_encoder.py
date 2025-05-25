import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Force l'utilisation de TF_USE_LEGACY_KERAS pour compatibilité
os.environ["TF_USE_LEGACY_KERAS"] = "1"

class SimpleEncoder:
    """
    Classe pour créer et gérer un encodeur simple sans avoir besoin de charger des modèles externes.
    Peut être utilisée directement dans l'environnement RL.
    """
    
    def __init__(self, input_dim=42, encoding_dim=16, tech_cols=None):
        """
        Initialise un encodeur simple.
        
        Args:
            input_dim: Dimension d'entrée (nombre de features techniques)
            encoding_dim: Dimension du vecteur encodé
            tech_cols: Liste des noms des colonnes techniques
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.tech_cols = tech_cols
        self.scaler = MinMaxScaler()
        self.trained = False
        self.logger = logging.getLogger("SimpleEncoder")
        
        # Création du modèle
        self._build_model()
    
    def _build_model(self):
        """Construit le modèle d'encodeur."""
        # Définition de l'architecture
        input_layer = keras.Input(shape=(self.input_dim,), name="technical_input")
        x = layers.Dense(64, activation="relu", name="encoder_dense1")(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation="relu", name="encoder_dense2")(x)
        
        # Création du modèle
        self.model = keras.Model(inputs=input_layer, outputs=encoded, name="SimpleEncoder")
        self.model.compile(optimizer="adam", loss="mse")
        
        self.logger.info(f"Modèle encodeur simple créé avec input_dim={self.input_dim}, encoding_dim={self.encoding_dim}")
    
    def fit_scaler(self, X):
        """
        Entraîne le scaler sur les données.
        
        Args:
            X: Données d'entrée (numpy array)
        """
        self.scaler.fit(X)
        self.logger.info(f"Scaler ajusté: min={X.min()}, max={X.max()}")
    
    def predict(self, X):
        """
        Normalise les données et prédit l'encodage.
        
        Args:
            X: Données d'entrée (numpy array)
        
        Returns:
            Encodage des données
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Nettoyage des données (remplacer NaN, Inf par 0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalisation
        X_scaled = self.scaler.transform(X)
        
        # Prédiction
        encoding = self.model.predict(X_scaled, verbose=0)
        return encoding
    
    def train_dummy(self, X, epochs=5, batch_size=32):
        """
        Entraîne l'encodeur avec un objectif factice (reconstruction identité).
        Utile pour initialiser les poids de manière sensée.
        
        Args:
            X: Données d'entrée (numpy array)
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
        """
        # Ajustement du scaler
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Création d'un décodeur factice pour l'entraînement
        input_layer = self.model.input
        encoded = self.model.output
        
        decoded = layers.Dense(64, activation="relu")(encoded)
        decoded = layers.Dense(self.input_dim, activation="linear")(decoded)
        
        autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        
        # Entraînement
        self.logger.info(f"Entraînement factice de l'encodeur sur {len(X)} échantillons...")
        autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Marquage comme entraîné
        self.trained = True
        self.logger.info(f"Encodeur entraîné (poids initialisés)")
    
    def save(self, filepath):
        """
        Sauvegarde le modèle, le scaler et la configuration.
        
        Args:
            filepath: Répertoire de sauvegarde
        """
        os.makedirs(filepath, exist_ok=True)
        
        # Sauvegarde du modèle
        model_path = os.path.join(filepath, "simple_encoder.keras")
        self.model.save(model_path)
        
        # Sauvegarde du scaler
        scaler_path = os.path.join(filepath, "simple_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Sauvegarde de la configuration
        config = {
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
            "technical_cols": self.tech_cols,
            "trained": self.trained
        }
        
        config_path = os.path.join(filepath, "simple_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Modèle, scaler et configuration sauvegardés dans {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Charge un encodeur sauvegardé.
        
        Args:
            filepath: Répertoire contenant les fichiers sauvegardés
        
        Returns:
            Instance de SimpleEncoder chargée
        """
        # Chargement de la configuration
        config_path = os.path.join(filepath, "simple_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Création de l'instance
        instance = cls(
            input_dim=config["input_dim"],
            encoding_dim=config["encoding_dim"],
            tech_cols=config["technical_cols"]
        )
        
        # Chargement du modèle
        model_path = os.path.join(filepath, "simple_encoder.keras")
        instance.model = keras.models.load_model(model_path)
        
        # Chargement du scaler
        scaler_path = os.path.join(filepath, "simple_scaler.pkl")
        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)
        
        instance.trained = config["trained"]
        instance.logger.info(f"Encodeur chargé depuis {filepath}")
        
        return instance

def create_encoder_from_data(data, tech_cols=None, encoding_dim=16, train=True):
    """
    Fonction utilitaire pour créer un encodeur à partir d'un DataFrame.
    
    Args:
        data: DataFrame pandas contenant les données
        tech_cols: Liste des colonnes techniques à utiliser (si None, toutes sauf timestamp/symbol/open/high/low/close/volume)
        encoding_dim: Dimension de l'encodage
        train: Si True, entraîne l'encodeur sur les données
    
    Returns:
        Instance de SimpleEncoder
    """
    logger = logging.getLogger("create_encoder")
    
    # Déterminer les colonnes techniques si non spécifiées
    if tech_cols is None:
        BASE_COLS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        tech_cols = [c for c in data.columns if c not in BASE_COLS]
    
    # Créer le modèle
    input_dim = len(tech_cols)
    encoder = SimpleEncoder(input_dim=input_dim, encoding_dim=encoding_dim, tech_cols=tech_cols)
    
    # Entraîner si demandé
    if train and len(data) > 0:
        # Extraire les données
        X = data[tech_cols].values
        
        # Nettoyer (remplacer NaN/Inf par 0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Entraîner
        encoder.train_dummy(X, epochs=3, batch_size=64)
    
    return encoder

if __name__ == "__main__":
    # Exemple d'utilisation
    import pandas as pd
    
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Chargement de données
    data_path = "ultimate/data/processed/market_features/all_assets_features_merged.parquet"
    
    try:
        df = pd.read_parquet(data_path)
        print(f"Données chargées: {df.shape}")
        
        # Sous-échantillonnage pour démonstration
        df_sample = df.sample(10000)
        
        # Création et entraînement de l'encodeur
        encoder = create_encoder_from_data(df_sample)
        
        # Test de prédiction
        sample_row = df.iloc[0][encoder.tech_cols].values
        encoding = encoder.predict(sample_row)
        print(f"Exemple d'encodage: {encoding.shape}")
        
        # Sauvegarde
        encoder.save("models/simple_encoder")
        print("Encodeur sauvegardé dans models/simple_encoder")
        
    except Exception as e:
        print(f"Erreur: {e}")