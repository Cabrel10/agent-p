import os
import sys
import argparse
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from monolith_implementation.monolith_model import MonolithModel
from monolith_implementation.contrastive_utils import jitter, scaling, time_masking
import logging

# Configuration CPU/threads
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-only', action='store_true', help='Forcer l\'utilisation du CPU uniquement, même si un GPU est disponible')
args, unknown = parser.parse_known_args()

os.environ['OMP_NUM_THREADS'] = '6'
os.environ['TF_NUM_INTRAOP_THREADS'] = '6'
os.environ['TF_NUM_INTEROP_THREADS'] = '6'
import tensorflow as tf
import numpy as np
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)
np.set_printoptions(threshold=10000)
if args.cpu_only:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("[INFO] GPU désactivé, calcul forcé sur CPU.")

# Configuration du modèle sera chargée depuis config.yaml
CONFIG = {}

# Colonnes à adapter selon optuna_monolith_optimization.py
technical_cols = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'RSI', 'MACD', 'MACDs', 'MACDh',
    'BBU', 'BBM', 'BBL', 'ATR', 'STOCHk', 'STOCHd', 'ADX', 'CCI', 'Momentum', 'ROC',
    'Williams_%R', 'TRIX', 'Ultimate_Osc', 'DPO', 'OBV', 'VWMA', 'CMF', 'MFI', 'Parabolic_SAR',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SenkouA', 'Ichimoku_SenkouB', 'Ichimoku_Chikou',
    'KAMA', 'VWAP', 'STOCHRSIk', 'CMO', 'PPO', 'FISHERt'
]
embedding_cols = ['llm_embedding']
mcp_cols = [f'mcp_feature_{i:03d}' for i in range(128)]
instrument_col = 'symbol'
datetime_col = 'timestamp'

def get_augmentation_fn(augmentation_type):
    if augmentation_type == "jitter":
        return jitter
    elif augmentation_type == "scaling":
        return scaling
    elif augmentation_type == "time_masking":
        return time_masking
    else:
        raise ValueError(f"Augmentation inconnue: {augmentation_type}")

def load_data(data_path: str):
    df = pd.read_parquet(data_path)
    logging.info(f"Dataset chargé: {df.shape} lignes, {len(df.columns)} colonnes")
    logging.info(f"Exemples de colonnes: {list(df.columns)[:10]}")
    
    if datetime_col in df.columns:
        df = df.sort_values(datetime_col)
        logging.info(f"Données triées par {datetime_col}")
    else:
        logging.warning(f"Colonne {datetime_col} non trouvée. Données non triées.")
    
    return df

def prepare_data(df: pd.DataFrame, technical_cols: list, mcp_cols: list, embedding_cols: list, instrument_col: str):
    # Vérifier les colonnes disponibles dans le DataFrame
    available_tech_cols = [col for col in technical_cols if col in df.columns]
    available_mcp_cols = [col for col in mcp_cols if col in df.columns]
    available_embedding_cols = [col for col in embedding_cols if col in df.columns]
    
    if instrument_col not in df.columns:
        logging.warning(f"Colonne d'instrument '{instrument_col}' non trouvée. Utilisation de 'symbol' ou création d'une colonne factice.")
        if 'symbol' in df.columns:
            instrument_col = 'symbol'
        else:
            df[instrument_col] = 'default'
    
    logging.info(f"Colonnes techniques disponibles: {len(available_tech_cols)}/{len(technical_cols)}")
    logging.info(f"Colonnes MCP disponibles: {len(available_mcp_cols)}/{len(mcp_cols)}")
    logging.info(f"Colonnes d'embedding disponibles: {len(available_embedding_cols)}/{len(embedding_cols)}")
    
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()

    tech_scaler = StandardScaler()
    if available_tech_cols:
        df_train[available_tech_cols] = tech_scaler.fit_transform(df_train[available_tech_cols])
        df_val[available_tech_cols] = tech_scaler.transform(df_val[available_tech_cols])

    mcp_scaler = None
    if available_mcp_cols and len(available_mcp_cols) > 0:
        mcp_scaler = StandardScaler()
        df_train[available_mcp_cols] = mcp_scaler.fit_transform(df_train[available_mcp_cols])
        df_val[available_mcp_cols] = mcp_scaler.transform(df_val[available_mcp_cols])

    instrument_map = {instrument: i for i, instrument in enumerate(df_train[instrument_col].unique())}

    def create_X_dict(df_subset, instrument_map_subset, tech_cols_subset, mcp_cols_subset, embedding_cols_subset, instrument_col_subset):
        X_dict = {}
        
        # Colonnes techniques
        available_tech_cols = [col for col in tech_cols_subset if col in df_subset.columns]
        if available_tech_cols:
            X_dict['technical_input'] = df_subset[available_tech_cols].values
        else:
            # Créer un tableau de zéros si aucune colonne technique n'est disponible
            X_dict['technical_input'] = np.zeros((len(df_subset), 1))
            logging.warning("Aucune colonne technique disponible. Utilisation d'un tableau de zéros.")

        # Colonnes d'embedding
        if embedding_cols_subset and len(embedding_cols_subset) > 0:
            available_emb_cols = [col for col in embedding_cols_subset if col in df_subset.columns]
            if available_emb_cols:
                emb_col = available_emb_cols[0]
                try:
                    if isinstance(df_subset[emb_col].iloc[0], str):
                        X_dict['embeddings_input'] = np.stack(df_subset[emb_col].apply(lambda x: np.array(json.loads(x))).values)
                    else:
                        X_dict['embeddings_input'] = np.stack(df_subset[emb_col].values)
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    logging.warning(f"Erreur lors du traitement des embeddings: {e}")
                    X_dict['embeddings_input'] = None
            else:
                X_dict['embeddings_input'] = None
        else:
            X_dict['embeddings_input'] = None

        # Colonnes MCP
        available_mcp_cols = [col for col in mcp_cols_subset if col in df_subset.columns]
        if available_mcp_cols and len(available_mcp_cols) > 0:
            X_dict['mcp_input'] = df_subset[available_mcp_cols].values
        else:
            X_dict['mcp_input'] = None

        # Colonne d'instrument
        if instrument_col_subset in df_subset.columns:
            X_dict['instrument_input'] = df_subset[instrument_col_subset].map(instrument_map_subset).fillna(0).astype(int).values
        else:
            X_dict['instrument_input'] = np.zeros(len(df_subset), dtype=int)
            logging.warning(f"Colonne d'instrument '{instrument_col_subset}' non trouvée. Utilisation de zéros.")

        return X_dict

    # Utiliser les colonnes disponibles pour créer les dictionnaires d'entrée
    available_tech_cols = [col for col in technical_cols if col in df.columns]
    available_mcp_cols = [col for col in mcp_cols if col in df.columns]
    available_embedding_cols = [col for col in embedding_cols if col in df.columns]
    
    X_train_dict = create_X_dict(df_train, instrument_map, available_tech_cols, available_mcp_cols, available_embedding_cols, instrument_col)
    X_val_dict = create_X_dict(df_val, instrument_map, available_tech_cols, available_mcp_cols, available_embedding_cols, instrument_col)

    return X_train_dict, X_val_dict, tech_scaler, mcp_scaler, instrument_map

def train_contrastive_model(monolith_model, X_train_dict, X_val_dict, augmentation_fn, temperature):
    # Limiter encore plus le nombre d'étapes pour l'entraînement
    steps_per_epoch = min(50, len(X_train_dict['technical_input']) // CONFIG['batch_size'])
    validation_steps = min(10, len(X_val_dict['technical_input']) // CONFIG['batch_size'])
    
    logging.info(f"Entraînement avec {steps_per_epoch} étapes par époque, {validation_steps} étapes de validation")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    # Réduire la taille du batch pour éviter les problèmes de mémoire
    actual_batch_size = min(32, CONFIG['batch_size'])
    logging.info(f"Utilisation d'une taille de batch de {actual_batch_size} (configuré: {CONFIG['batch_size']})")
    
    try:
        history = monolith_model.train(
            X_train_dict, None,
            validation_data=(X_val_dict, None),
            epochs=CONFIG.get('epochs', 3),  # Réduire à 3 époques par défaut
            batch_size=actual_batch_size,
            callbacks=callbacks,
            contrastive_training=True,
            contrastive_augmentation_fn=augmentation_fn,
            contrastive_temperature=temperature,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2  # Réduire la verbosité
        )
        return history
    except Exception as e:
        logging.error(f"Erreur pendant l'entraînement: {e}")
        return None

def save_artifacts(monolith_model, tech_scaler, mcp_scaler, history, model_config, data_processing_metadata):
    try:
        os.makedirs(CONFIG['model_dir'], exist_ok=True)
        logging.info(f"Sauvegarde du modèle dans {CONFIG['model_dir']}")
        
        # Sauvegarde du modèle
        try:
            model_path = os.path.join(CONFIG['model_dir'], "contrastive_encoder_model.keras")
            monolith_model.save(model_path)
            logging.info(f"Modèle sauvegardé avec succès: {model_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du modèle: {e}")
        
        # Sauvegarde des configurations
        try:
            config_path = os.path.join(CONFIG['model_dir'], "model_config.json")
            with open(config_path, "w") as f:
                json.dump(model_config, f)
            logging.info(f"Configuration sauvegardée: {config_path}")
            
            metadata_path = os.path.join(CONFIG['model_dir'], "data_processing_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(data_processing_metadata, f)
            logging.info(f"Métadonnées sauvegardées: {metadata_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des configurations: {e}")
        
        # Sauvegarde des scalers
        try:
            scalers_path = os.path.join(CONFIG['model_dir'], "scalers.pkl")
            with open(scalers_path, "wb") as f:
                pickle.dump({'tech_scaler': tech_scaler, 'mcp_scaler': mcp_scaler}, f)
            logging.info(f"Scalers sauvegardés: {scalers_path}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des scalers: {e}")
        
        # Sauvegarde de l'historique d'entraînement
        if history is not None and hasattr(history, 'history'):
            try:
                import pandas as pd
                history_path = os.path.join(CONFIG['model_dir'], 'training_history.csv')
                pd.DataFrame(history.history).to_csv(history_path, index=False)
                logging.info(f"Historique d'entraînement sauvegardé: {history_path}")
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde de l'historique: {e}")
        
        logging.info(f"Artefacts sauvegardés dans: {CONFIG['model_dir']}")
    except Exception as e:
        logging.error(f"Erreur générale lors de la sauvegarde des artefacts: {e}")

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration chargée depuis : {config_path}")
        return config
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {config_path}: {e}")
        return None

def main():
    # Configurer le logging
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    try:
        # Charger la configuration
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        config_path = script_dir / "config.yaml"
        
        global_config = load_config(config_path)
        if not global_config:
            logging.critical("Impossible de charger la configuration. Arrêt du script.")
            sys.exit(1)
        
        # Mettre à jour CONFIG avec les paramètres du fichier de configuration
        global CONFIG
        CONFIG.update(global_config["contrastive_encoder_train"])
        CONFIG["tech_input_shape"] = (43,)  # sera mis à jour plus tard
        CONFIG["reconstruction_target_dim"] = 43  # sera mis à jour plus tard
        CONFIG["use_batch_norm"] = True
        
        # Réduire la taille du modèle et la complexité
        CONFIG["transformer_blocks"] = min(2, CONFIG.get("transformer_blocks", 3))
        CONFIG["transformer_heads"] = min(4, CONFIG.get("transformer_heads", 8))
        CONFIG["latent_dim"] = min(32, CONFIG.get("latent_dim", 80))
        CONFIG["dense_units"] = min(32, CONFIG.get("dense_units", 32))
        CONFIG["lstm_units"] = min(32, CONFIG.get("lstm_units", 64))
        
        # Définir les chemins à partir de la configuration
        project_root = Path(global_config["project_root"])
        CONFIG['model_dir'] = str(project_root / global_config["paths"]["contrastive_encoder_dir"])
        
        # Préparation des données
        try:
            data_path = str(project_root / global_config["paths"]["merged_features_file"])
            logging.info(f"Chargement des données depuis: {data_path}")
            df = load_data(data_path)
            
            # Sous-échantillonner les données pour réduire la taille
            if len(df) > 50000:
                logging.info(f"Sous-échantillonnage des données: {len(df)} -> 50000 lignes")
                df = df.sample(n=50000, random_state=42)
            
            X_train_dict, X_val_dict, tech_scaler, mcp_scaler, instrument_map = prepare_data(
                df, technical_cols, mcp_cols, embedding_cols, instrument_col
            )
        except Exception as e:
            logging.error(f"Erreur lors de la préparation des données: {e}")
            sys.exit(1)

        # Configuration du modèle
        try:
            model_config = CONFIG.copy()
            model_config["tech_input_shape"] = (len(technical_cols),)
            model_config["reconstruction_target_dim"] = len(technical_cols)
            
            # Gestion sécurisée des embeddings
            if 'embeddings_input' in X_train_dict and X_train_dict['embeddings_input'] is not None:
                model_config["embeddings_input_dim"] = X_train_dict['embeddings_input'].shape[1]
            else:
                model_config["embeddings_input_dim"] = 0
                logging.warning("Pas d'embeddings disponibles, utilisation de dimension 0")
            
            # Nombre d'instruments et caractéristiques MCP
            available_mcp_cols = [col for col in mcp_cols if col in df.columns]
            model_config["mcp_input_dim"] = len(available_mcp_cols)
            model_config["instrument_vocab_size"] = max(1, len(instrument_map))  # Au moins 1
            model_config["instrument_embedding_dim"] = 8
            
            data_processing_metadata = {
                'technical_cols': technical_cols,
                'embedding_cols': embedding_cols,
                'mcp_cols': available_mcp_cols,
                'instrument_col': instrument_col,
                'instrument_map': instrument_map,
                'num_instruments': len(instrument_map),
                'embeddings_input_dim': model_config["embeddings_input_dim"],
                'mcp_input_dim': model_config["mcp_input_dim"],
                'technical_input_dim': len(technical_cols)
            }
            model_config.update(data_processing_metadata)
        except Exception as e:
            logging.error(f"Erreur lors de la configuration du modèle: {e}")
            sys.exit(1)

        # Création et entraînement du modèle
        try:
            monolith_model = MonolithModel(config=model_config)
            monolith_model.model.summary()
            
            augmentation_fn = get_augmentation_fn(CONFIG["augmentation_type"])
            temperature = CONFIG["contrastive_temperature"]
            
            logging.info(f"Début de l'entraînement contrastif avec {CONFIG['augmentation_type']} comme augmentation")
            history = train_contrastive_model(monolith_model, X_train_dict, X_val_dict, augmentation_fn, temperature)
            
            # Log des pertes finales
            if history and hasattr(history, 'history'):
                train_loss = history.history['loss'][-1] if 'loss' in history.history else None
                val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
                logging.info(f"Final Contrastive Train Loss: {train_loss}")
                logging.info(f"Final Contrastive Val Loss: {val_loss}")
            
            save_artifacts(monolith_model, tech_scaler, mcp_scaler, history, model_config, data_processing_metadata)
        except Exception as e:
            logging.error(f"Erreur lors de la création ou de l'entraînement du modèle: {e}")
            # Même en cas d'erreur, essayer de sauvegarder ce qui est disponible
            try:
                save_artifacts(monolith_model, tech_scaler, mcp_scaler, None, model_config, data_processing_metadata)
            except:
                logging.error("Impossible de sauvegarder les artefacts après erreur")
    except Exception as e:
        logging.error(f"Erreur générale dans la fonction main: {e}")

if __name__ == '__main__':
    try:
        main()
        logging.info("Script terminé avec succès")
    except Exception as e:
        logging.error(f"Erreur fatale dans l'exécution du script: {e}")
        import traceback
        logging.error(traceback.format_exc())
