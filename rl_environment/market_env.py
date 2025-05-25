import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"  # Force Keras 3 mode

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import logging
import traceback  # Pour logs détaillés

# Logger dédié pour MarketEnv
logger_env = logging.getLogger("MarketEnv")
if not logger_env.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_env.addHandler(handler)
    logger_env.setLevel(logging.DEBUG)  # DEBUG pour logs granulaires

# Import de la couche personnalisée TransformerBlock
from monolith_implementation.monolith_model import (
    TransformerBlock,
    projection_zero,
    l2_normalize_fn,
    build_autoencoder_monolith_model
)

class MarketEnv(gym.Env):
    """
    Environnement de simulation de marché pour agent RL.
    Compatible avec l'API Gymnasium.
    """

    def __init__(
        self,
        data_path="ultimate/data/processed/multi_crypto_dataset.parquet",
        encoder_model_path="models/sprint2_contrastive_encoder/contrastive_encoder_model.keras",
        artifacts_dir="models/sprint2_contrastive_encoder/",
        initial_capital=15,
        transaction_cost_pct=0.009,
        position_size=1.0,  # Taille fixe de la position pour chaque trade
        eval_mode=False,    # Mode évaluation/backtest
        eval_data_ratio=0.2, # Proportion des données à utiliser pour l'évaluation (les X derniers %)
        skip_encoder=False, # Option pour contourner l'utilisation de l'encodeur
        symbol=None # Nouvelle option pour filtrer une paire spécifique
    ):
        super().__init__()
        logger_env.info(f"Début de l'initialisation de MarketEnv (eval_mode={eval_mode}, eval_data_ratio={eval_data_ratio})")

        # Chargement des données historiques
        try:
            logger_env.debug(f"Chargement des données depuis: {data_path}")
            full_df = pd.read_parquet(data_path)
            logger_env.debug(f"Données chargées avec succès. Taille: {len(full_df)}, Colonnes: {full_df.columns.tolist()}")
        except Exception as e:
            logger_env.error(f"Erreur lors du chargement des données: {traceback.format_exc()}")
            raise

        # --- Filtrage par symbol (paire) ---
        if symbol is not None:
            assert 'symbol' in full_df.columns, "La colonne 'symbol' est requise pour le filtrage par paire."
            logger_env.info(f"Filtrage du DataFrame sur la paire '{symbol}'")
            filtered_df = full_df[full_df['symbol'] == symbol].reset_index(drop=True)
            if filtered_df.empty:
                logger_env.error(f"Aucune donnée trouvée pour la paire {symbol}. Symboles disponibles: {full_df['symbol'].unique().tolist()}")
                raise ValueError(f"Aucune donnée trouvée pour la paire {symbol}.")
            full_df = filtered_df
            self.symbol = symbol
        else:
            self.symbol = None

        try:
            if eval_mode:
                eval_start_index = int(len(full_df) * (1 - eval_data_ratio))
                logger_env.debug(f"Mode évaluation: Utilisation des données de l'index {eval_start_index} à {len(full_df)-1}.")
                if eval_start_index >= len(full_df) or len(full_df) == 0:
                    logger_env.warning(f"Le ratio d'évaluation ({eval_data_ratio}) ou la taille du DataFrame ({len(full_df)}) résulte en un DataFrame de test vide ou invalide. Utilisation du DataFrame complet.")
                    self.df = full_df.reset_index(drop=True)
                else:
                    self.df = full_df.iloc[eval_start_index:].reset_index(drop=True)
                if self.df.empty:
                    logger_env.error("Le DataFrame pour le mode évaluation est vide après le slicing.")
                    raise ValueError("Le DataFrame pour le mode évaluation est vide après le slicing alors que le DataFrame original n'était pas vide. Vérifiez eval_data_ratio.")
                logger_env.debug(f"DataFrame après slicing pour évaluation, taille: {len(self.df)}")
            else:
                self.df = full_df
                logger_env.debug(f"Mode entraînement: Utilisation de toutes les données. Taille: {len(self.df)}")
            # --- Resample toutes les 5 minutes si timestamp disponible ---
            if 'timestamp' in self.df.columns:
                try:
                    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                    self.df = self.df.set_index('timestamp').resample('5T').first().dropna().reset_index()
                    logger_env.info(f"DataFrame resamplé toutes les 5 minutes. Nouvelle taille: {len(self.df)}")
                except Exception as e:
                    logger_env.warning(f"Resampling 5min impossible: {e}. On continue sans resample.")
            self.n_steps = len(self.df)
            self.current_step = 0
            logger_env.debug(f"Nombre total de pas dans l'environnement: {self.n_steps}")
        except Exception as e:
            logger_env.error(f"Erreur lors du traitement du DataFrame: {traceback.format_exc()}")
            raise

        # Chargement de l'encodeur pré-entraîné (sauf si skip_encoder=True)
        if encoder_model_path is None:
            logger_env.error("encoder_model_path est None, impossible de charger l'encodeur.")
            raise ValueError("encoder_model_path doit être spécifié.")
        try:
            logger_env.debug(f"Chargement de l'encodeur depuis: {encoder_model_path}")
            self.encoder = keras.models.load_model(
                encoder_model_path,
                custom_objects={
                    "Custom>TransformerBlock": TransformerBlock,
                    "TransformerBlock": TransformerBlock,
                    "projection_zero": projection_zero,
                    "Custom>l2_normalize_fn": l2_normalize_fn,
                    "l2_normalize_fn": l2_normalize_fn,
                },
                safe_mode=False
            )
            logger_env.debug("Encodeur chargé avec succès.")
        except Exception as e:
            logger_env.error(f"Erreur lors du chargement de l'encodeur: {traceback.format_exc()}")
            raise

        # Chargement des artefacts (scalers, métadonnées)
        if artifacts_dir is None:
            logger_env.error("artifacts_dir est None, impossible de charger les artefacts.")
            raise ValueError("artifacts_dir doit être spécifié.")
        try:
            logger_env.debug(f"Chargement des artefacts depuis: {artifacts_dir}")
            scalers_path = os.path.join(artifacts_dir, "scalers.pkl")
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger_env.debug(f"Scalers chargés depuis: {scalers_path}")
            else:
                logger_env.warning(f"Fichier de scalers non trouvé à: {scalers_path}")
                self.scalers = {}
            metadata_path = os.path.join(artifacts_dir, "data_processing_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger_env.debug(f"Métadonnées chargées depuis: {metadata_path}")
            else:
                logger_env.warning(f"Fichier de métadonnées non trouvé à: {metadata_path}")
                self.metadata = {}
        except Exception as e:
            logger_env.error(f"Erreur lors du chargement des artefacts: {traceback.format_exc()}")
            self.scalers = {}
            self.metadata = {}

        # Initialisation des paramètres de trading
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0: pas de position, 1: en position
        self.entry_price = 0.0
        self.transaction_cost_pct = transaction_cost_pct
        self.position_size = position_size  # Quantité d'actif par trade
        logger_env.debug(f"Capital initial: {self.initial_capital}, Coût de transaction: {self.transaction_cost_pct}, Taille de position: {self.position_size}")

        # Définition de l'espace d'action (ACHETER, VENDRE, TENIR)
        self.action_space = spaces.Discrete(3)
        logger_env.debug("Espace d'action défini: Discrete(3)")

        # Définition dynamique de la dimension de projection pour l'observation
        try:
            self.projection_dim = self.encoder.output_shape[-1]
            logger_env.debug(f"Dimension de projection trouvée: {self.projection_dim}")
        except Exception as e:
            logger_env.warning(f"Impossible de déterminer la dimension de projection depuis le modèle: {e}. Essai avec latent_representation ou fallback.")
            try:
                self.projection_dim = self.encoder.get_layer("latent_representation").output_shape[-1]
                logger_env.debug(f"Dimension de projection trouvée via latent_representation: {self.projection_dim}")
            except Exception:
                self.projection_dim = 64  # Fallback codé en dur
                logger_env.warning(f"Fallback: projection_dim fixé à 64.")

        # Nouvelle version : on extrait l’entier de projection_dim avant d’ajouter 3
        # Si projection_dim = (None, D) on prend D via [-1], sinon (D,) via [0]
        latent_dim = (
            self.projection_dim[-1]
            if isinstance(self.projection_dim, tuple) and len(self.projection_dim) > 1
            else self.projection_dim[0]
        )
        total_dim = latent_dim + 3
        self.observation_space = spaces.Box(
            low  = np.full((total_dim,), -1.0, dtype=np.float32),
            high = np.full((total_dim,),  1.0, dtype=np.float32),
            dtype=np.float32
        )
        logger_env.info(f"Espace d'observation défini: {self.observation_space}")

        # Définition dynamique de la dimension de projection pour l'observation
        logger_env.info("Détermination de la dimension de projection...")
        # Logs d'inspection initiaux (rendus plus robustes)
        encoder_output_attr = getattr(self.encoder, 'output', 'Attribut "output" non trouvé')
        logger_env.info(f"self.encoder.output (type): {type(encoder_output_attr)}")
        if isinstance(encoder_output_attr, list):
            logger_env.info(f"self.encoder.output (longueur): {len(encoder_output_attr)}")

        encoder_outputs_attr = getattr(self.encoder, 'outputs', 'Attribut "outputs" non trouvé')
        logger_env.info(f"self.encoder.outputs (type): {type(encoder_outputs_attr)}")
        if isinstance(encoder_outputs_attr, list):
            logger_env.info(f"self.encoder.outputs (longueur): {len(encoder_outputs_attr)}")
            try:
                for i, out_tensor in enumerate(encoder_outputs_attr):
                    logger_env.info(f"  Sortie {i}: name='{getattr(out_tensor, 'name', 'N/A')}', shape={getattr(out_tensor, 'shape', 'N/A')}")
            except Exception as e_loop:
                logger_env.error(f"Erreur en bouclant sur self.encoder.outputs: {e_loop}")
        
        logger_env.info("self.encoder.output_shape: {}".format(getattr(self.encoder, 'output_shape', 'Attribut \"output_shape\" non trouvé')))
        
        output_names_val = getattr(self.encoder, 'output_names', 'Attribut "output_names" non trouvé')
        logger_env.info(f"self.encoder.output_names: {output_names_val}")

        if skip_encoder:
            # Si l'encodeur est désactivé, on utilise une dimension de projection fixe
            # Cette valeur doit correspondre à la taille de notre représentation simplifiée
            # dans _get_observation (tech_subset + llm_subset + mcp_subset)
            self.projection_dim = 32  # 10 (tech) + 10 (llm) + 12 (mcp)
            logger_env.info(f"Option skip_encoder activée: projection_dim fixée à {self.projection_dim}")
        else:
            self.projection_dim = None
            # --- BYPASS TEMPORAIRE ---
            # self.projection_dim = 64
            # logger_env.warning(f"BYPASS TEMPORAIRE: projection_dim forcé à {self.projection_dim}")
            # --- FIN BYPASS TEMPORAIRE ---
            logger_env.info("Début du bloc try pour la détermination de projection_dim.")
            try:
                output_names_attr = getattr(self.encoder, 'output_names', None)
                if output_names_attr and isinstance(output_names_attr, list):
                    output_names = output_names_attr
                    outputs_list = getattr(self.encoder, 'outputs', [])
                    logger_env.info(f"Dans try: output_names = {output_names}")
                    logger_env.info(f"Dans try: outputs_list (longueur) = {len(outputs_list)}")

                    if not outputs_list:
                        logger_env.error("self.encoder.outputs est une liste vide alors que output_names existe. Impossible d'indexer.")
                        self.projection_dim = 64
                        logger_env.info(f"outputs_list est vide, projection_dim mis à la valeur par défaut: {self.projection_dim}")
                    elif len(output_names) != len(outputs_list):
                        logger_env.error(f"Incohérence: len(output_names)={len(output_names)} mais len(outputs_list)={len(outputs_list)}. Utilisation de la valeur par défaut.")
                        self.projection_dim = 64
                        logger_env.info(f"Incohérence de longueur, projection_dim mis à la valeur par défaut: {self.projection_dim}")
                    else:
                        found_projection = False
                        for target_name in ["projection_l2", "projection"]:
                            if target_name in output_names:
                                try:
                                    idx = output_names.index(target_name)
                                    self.projection_dim = outputs_list[idx].shape[-1]
                                    logger_env.info(f"projection_dim trouvé via output_names ('{target_name}'): {self.projection_dim}")
                                    found_projection = True
                                    break
                                except Exception as e_idx:
                                    logger_env.exception(f"Erreur lors de l'accès à outputs_list[{idx}] pour '{target_name}': {e_idx}")
                        if not found_projection:
                            logger_env.warning(f"Noms 'projection_l2' ou 'projection' non trouvés dans {output_names}. Prise de la dernière sortie.")
                            self.projection_dim = outputs_list[-1].shape[-1]
                            logger_env.info(f"Fallback: projection_dim pris comme dernière sortie (index -1): {self.projection_dim}")
                elif hasattr(self.encoder, "output_shape"):
                    logger_env.info("Cas mono-sortie, utilisation de output_shape[-1]")
                    if isinstance(self.encoder.output_shape, tuple):
                        self.projection_dim = self.encoder.output_shape[-1]
                        logger_env.info(f"projection_dim via output_shape (mono-sortie, tuple): {self.projection_dim}")
                    else:
                        logger_env.warning(f"output_shape n'est pas un tuple: {self.encoder.output_shape}. Impossible de déterminer projection_dim.")
                        if isinstance(self.encoder.output_shape, dict):
                            if 'projection_l2' in self.encoder.output_shape:
                                self.projection_dim = self.encoder.output_shape['projection_l2'][-1]
                                logger_env.info(f"projection_dim via output_shape (dict, 'projection_l2'): {self.projection_dim}")
                            elif 'projection' in self.encoder.output_shape:
                                self.projection_dim = self.encoder.output_shape['projection'][-1]
                                logger_env.info(f"projection_dim via output_shape (dict, 'projection'): {self.projection_dim}")
                            else:
                                logger_env.warning(f"Noms 'projection_l2' ou 'projection' non trouvés dans output_shape (dict).")
                        else:
                            logger_env.warning(f"output_shape est de type inattendu: {type(self.encoder.output_shape)}.")
                else:
                    logger_env.error("Ni output_names ni output_shape trouvés pour déterminer projection_dim.")
                    raise ValueError("Impossible de déterminer projection_dim")
            except Exception as e:
                logger_env.exception(f"Erreur lors de la détermination de projection_dim: {e}")
        
            if self.projection_dim is None:
                logger_env.error("projection_dim n'a pas pu être déterminé. Utilisation de la valeur par défaut.")
                self.projection_dim = 64
                logger_env.info(f"Utilisation de la valeur par défaut pour projection_dim: {self.projection_dim}")
        obs_low_bound = -5.0
        obs_high_bound = 5.0
        self.observation_space = spaces.Box(
            low=obs_low_bound, high=obs_high_bound, shape=(self.projection_dim + 3,), dtype=np.float32
        )
        logger_env.info(f"Espace d'observation défini: {self.observation_space}")
        logger_env.info("Initialisation de MarketEnv terminée.")

    def reset(self, seed=None, options=None):
        logger_env.info(f"[reset] Entrée dans reset() - current_step avant: {self.current_step}, n_steps: {self.n_steps}, capital: {getattr(self, 'capital', None)}")
        if self.n_steps == 0:
            logger_env.error("Impossible de réinitialiser l'environnement: n_steps est 0 (DataFrame vide).")
            raise ValueError("Le DataFrame de l'environnement est vide, impossible de réinitialiser.")
        self.current_step = 0
        self.capital = self.initial_capital  # Toujours réinitialiser au capital de départ
        self.position = 0
        self.entry_price = 0.0
        logger_env.info(f"[reset] Paramètres réinitialisés: current_step={self.current_step}, capital={self.capital}, position={self.position}, entry_price={self.entry_price}")
        try:
            logger_env.info(f"[reset] Appel à _get_observation pour current_step={self.current_step}")
            obs = self._get_observation(self.current_step)
            logger_env.info("[reset] _get_observation exécuté avec succès")
        except Exception as e:
            logger_env.error(f"[reset] Exception lors de l'appel à _get_observation: {e}", exc_info=True)
            raise
        info = {}
        logger_env.info("[reset] Fin de reset() - retour de l'observation et info")
        return obs, info

    def _get_position_size(self):
        # Paliers dynamiques selon le capital
        tiers = [
            (0,    0.1),   # <100
            (100,  0.5),   # <1000
            (1000, 1.0),   # >=1000
        ]
        for seuil, taille in reversed(tiers):
            if self.capital >= seuil:
                return taille
        return tiers[0][1]

    def step(self, action):
        # --- Logs avant action ---
        prev_capital = self.capital
        prev_position = self.position
        current_price = float(self.df.iloc[self.current_step]['open'])
        logger_env.debug(f"[BEFORE] step={self.current_step} | pos={prev_position} | cap={prev_capital:.4f} | price={current_price:.4f}")

        # Paliers dynamiques
        self.position_size = self._get_position_size()

        # Contraindre le trading à toutes les 5 minutes
        BUY, SELL, HOLD = 2, 0, 1
        info = {}
        if self.current_step % 5 != 0:
            action = HOLD
            info['skipped'] = True

        # Logique de trading
        reward = 0.0
        entry_price = getattr(self, 'entry_price', 0.0)
        exit_price = None
        # BUY
        if action == BUY and self.position == 0:
            max_affordable = self.capital / ((1 + self.transaction_cost_pct) * current_price)
            qty = min(self.position_size, max_affordable)
            if qty <= 0:
                logger_env.info(f"[BUY] CAPITAL INSUFFISANT pour acheter (capital={self.capital}, prix={current_price}, position_size={self.position_size})")
            else:
                cost = qty * current_price
                fee = cost * self.transaction_cost_pct
                total_cost = cost + fee
                self.entry_price = current_price
                self.position = qty
                self.capital -= total_cost
                logger_env.info(f"[BUY] qty={qty:.4f}, cost={cost:.4f}, fee={fee:.4f}, cap_after={self.capital:.4f}")
        # SELL
        elif action == SELL and self.position > 0:
            exit_price = current_price
            gross_pnl = self.position * (exit_price - self.entry_price)
            fee = abs(gross_pnl) * self.transaction_cost_pct
            net_pnl = gross_pnl - fee
            self.capital += self.position * exit_price - fee
            logger_env.info(f"[SELL] entry={self.entry_price}, exit={exit_price}, qty={self.position}, gross_pnl={gross_pnl:.4f}, fee={fee:.4f}, cap_after={self.capital:.4f}")
            self.position = 0
            self.entry_price = 0.0
        # HOLD : rien à faire

        # Reward = delta capital
        reward = self.capital - prev_capital

        # Incrément step
        self.current_step += 1
        done = (self.current_step >= self.n_steps)
        truncated = False

        # Historique des trades
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        self.trade_history.append({
            'step': self.current_step,
            'action': action,
            'position': self.position,
            'price': current_price,
            'capital': self.capital,
            'reward': reward
        })

        # Logs après action
        logger_env.info(
            f"[AFTER] action={action} | pos: {prev_position}->{self.position} | "
            f"entry_price={getattr(self, 'entry_price', None)} | "
            f"exit_price={exit_price} | "
            f"capital: {prev_capital:.4f}->{self.capital:.4f} | reward={reward:.4f}"
        )

        # Terminaison si capital <= 0
        if self.capital <= 0:
            self.capital = 0
            done = True
            reward -= 100

        obs = self._get_observation(self.current_step)
        return obs, reward, done, truncated, info

    def _get_observation(self, idx=None):
        logger_env.info(f"[_get_observation] Entrée - idx={idx}, current_step={self.current_step}, n_steps={self.n_steps}")
        # Si idx n'est pas fourni, utiliser current_step
        if idx is None:
            idx = self.current_step
            logger_env.info(f"[_get_observation] idx non fourni, utilisation de current_step={idx}")
        if idx >= self.n_steps:
            logger_env.error(f"_get_observation: idx ({idx}) >= n_steps ({self.n_steps}).")
            raise IndexError(f"_get_observation: idx ({idx}) est hors des limites du DataFrame (n_steps: {self.n_steps}).")            
        logger_env.info(f"[_get_observation] Avant extraction de la ligne du DataFrame pour idx={idx}")
        current_data_df = self.df.iloc[[idx]].copy()
        logger_env.info(f"[_get_observation] current_data_df extrait, shape={current_data_df.shape}")

        technical_cols = self.metadata.get('technical_cols', [])
        mcp_cols = self.metadata.get('mcp_cols', [])
        embedding_cols = self.metadata.get('embedding_cols', [])
        instrument_col = self.metadata.get('instrument_col', 'symbol')
        instrument_map = self.metadata.get('instrument_map', {})

        try:
            if 'tech_scaler' in self.scalers and technical_cols and all(col in current_data_df.columns for col in technical_cols):
                current_data_df[technical_cols] = self.scalers['tech_scaler'].transform(current_data_df[technical_cols])
            elif technical_cols and not all(col in current_data_df.columns for col in technical_cols):
                logger_env.warning("Certaines colonnes techniques sont manquantes pour le scaling.")
            if 'mcp_scaler' in self.scalers and mcp_cols and all(col in current_data_df.columns for col in mcp_cols):
                current_data_df[mcp_cols] = self.scalers['mcp_scaler'].transform(current_data_df[mcp_cols])
            elif mcp_cols and not all(col in current_data_df.columns for col in mcp_cols):
                logger_env.warning("Certaines colonnes MCP sont manquantes pour le scaling.")
        except Exception as e:
            logger_env.error(f"Erreur lors de l'application des scalers: {e}")
            raise

        # Préparation des données pour l'encodeur ou pour la représentation simplifiée
        technical_features = None
        llm_embedding = None
        mcp_features = None

        if technical_cols:
            technical_features = current_data_df[technical_cols].values.astype(np.float32)[0]
        
        if embedding_cols and embedding_cols[0] in current_data_df:
            raw_embedding = current_data_df[embedding_cols[0]].iloc[0]
            if isinstance(raw_embedding, str):
                embedding_list = json.loads(raw_embedding)
            elif isinstance(raw_embedding, (list, np.ndarray)):
                embedding_list = raw_embedding
            else:
                logger_env.warning(f"Format d'embedding non reconnu pour {embedding_cols[0]} à l'étape {self.current_step}. Utilisation de zéros.")
                embedding_list = [0.0] * self.metadata.get('embeddings_input_dim', 768)
            llm_embedding = np.array(embedding_list, dtype=np.float32)
        else:
            llm_embedding = np.zeros(self.metadata.get('embeddings_input_dim', 768), dtype=np.float32)
        
        if mcp_cols:
            mcp_features = current_data_df[mcp_cols].values.astype(np.float32)[0]

        # Obtention de l'instrument actuel
        if instrument_col not in current_data_df.columns:
            logger_env.error(f"La colonne instrument '{instrument_col}' est manquante dans current_data_df à l'étape {self.current_step}.")
            current_instrument_name = "UNKNOWN"
        else:
            current_instrument_name = current_data_df[instrument_col].iloc[0]
        instrument_id = instrument_map.get(current_instrument_name, 0)

        # Génération de l'embedding selon que l'encodeur est disponible ou non
        if self.encoder is not None:
            # Utilisation de l'encodeur pour obtenir une représentation latente
            X_dict = {}
            if technical_features is not None:
                X_dict['technical_input'] = np.expand_dims(technical_features, axis=0)
            if llm_embedding is not None:
                X_dict['embeddings_input'] = np.expand_dims(llm_embedding, axis=0)
            if mcp_features is not None:
                X_dict['mcp_input'] = np.expand_dims(mcp_features, axis=0)
            X_dict['instrument_input'] = np.array([[instrument_id]], dtype=np.int32)
            
            model_output = self.encoder.predict(X_dict, verbose=0)
            if isinstance(model_output, dict) and 'projection' in model_output:
                embedding = model_output['projection'][0]
            elif isinstance(model_output, (list, tuple)) and len(model_output) > 1:
                embedding = model_output[1][0]
            else:
                logger_env.warning(f"Sortie de l'encodeur inattendue à l'étape {self.current_step}. Type: {type(model_output)}. Utilisation d'un embedding de zéros.")
                embedding = model_output[0] if hasattr(model_output, '__getitem__') else np.zeros(self.projection_dim, dtype=np.float32)
                
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger_env.warning(f"Embedding contient NaN/Inf à l'étape {self.current_step}. Remplacement par des zéros.")
                embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Si l'encodeur est désactivé, on utilise une représentation simplifiée
            logger_env.debug("Utilisation d'une représentation simplifiée sans encodeur")
            # Prendre un sous-ensemble des features pour réduire la dimensionnalité
            tech_subset = technical_features[:10] if technical_features is not None else np.zeros(10, dtype=np.float32)
            llm_subset = llm_embedding[:10] if llm_embedding is not None else np.zeros(10, dtype=np.float32)
            mcp_subset = mcp_features[:12] if mcp_features is not None else np.zeros(12, dtype=np.float32)
            
            # Créer une représentation de même dimension que celle de l'encodeur (32)
            embedding = np.concatenate([tech_subset, llm_subset, mcp_subset])

        # Obtention du prix actuel et calcul des métriques normalisées
        if "close" not in current_data_df.columns:
            logger_env.error(f"Colonne 'close' manquante dans current_data_df à l'étape {self.current_step} pour le calcul de norm_entry_price.")
            current_price = 0
        else:
            current_price = current_data_df["close"].iloc[0]

        norm_capital = self.capital / self.initial_capital if self.initial_capital > 0 else 0.0
        norm_entry_price = (self.entry_price / current_price) if self.position == 1 and current_price > 0 else 0.0

        # Création de l'observation finale
        obs = np.concatenate(
            [embedding, [float(self.position), norm_capital, norm_entry_price]]
        ).astype(np.float32)
        logger_env.debug(f"Observation générée pour l'étape {self.current_step}: shape={obs.shape}")
        return obs
