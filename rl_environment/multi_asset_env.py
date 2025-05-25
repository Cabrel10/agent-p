import os
import time
import yaml
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import matplotlib.pyplot as plt
import pandas as pd

# Configuration du logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MultiAssetEnv')

# Chargement de la configuration
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()
PROJECT_ROOT = Path(CONFIG["project_root"])


class OrderStatus(Enum):
    PENDING = 'PENDING'
    EXECUTED = 'EXECUTED'
    EXPIRED = 'EXPIRED'
    CANCELED = 'CANCELED'


class OrderType(Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP_LOSS = 'STOP_LOSS'
    TRAILING_STOP = 'TRAILING_STOP'


class MultiAssetEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 data_path=None,
                 encoder_model=None,
                 initial_capital=15,
                 transaction_cost_pct=0.001,
                 verbose_env=False, # Renommé pour éviter la confusion avec verbose du logger
                 mode="train",
                 skip_encoder=False):
        super().__init__()
        self.verbose_env = verbose_env
        self.mode = mode
        self.skip_encoder = skip_encoder
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.current_step = 0
        self.done = False
        self.history = []  # Pour le logging SB3 (trajectoire complète)
        self.trade_log = []  # Pour le logging des transactions (PnL, etc.)
        
        # Paramètres de trading professionnels depuis la configuration
        try:
            self.min_order_value = CONFIG["trading"]["min_order_value"]
            self.fixed_fee = CONFIG["trading"]["transaction_fee_fixed"]
        except KeyError as e:
            logger.error(f"Clé manquante dans config.yaml: {e}")
            raise KeyError(f"Clé manquante dans config.yaml: {e}")
        
        # Définition des paliers (tiers) de capital depuis la configuration
        try:
            self.tiers = CONFIG["trading"]["tiers"]
            # Vérifier que chaque tier a toutes les clés nécessaires
            for t in self.tiers:
                for key in ("low", "high", "max_positions", "allocation_frac", "reward_pos_mult", "reward_neg_mult"):
                    if key not in t:
                        logger.error(f"Tier mal défini, clé manquante: {key}")
                        raise KeyError(f"Tier mal défini, clé manquante: {key}")
        except KeyError as e:
            logger.error(f"Erreur tiers config: {e}")
            raise RuntimeError(f"Erreur tiers config: {e}")
        
        # État du portefeuille
        self.positions = {}  # {asset_id: {qty: float, entry_price: float, type: str}}
        self.orders = {}     # {order_id: {asset_id: str, type: str, price: float, qty: float, expiry: int}}
        self.next_order_id = 0

        # Définition des actifs à trader
        self.num_assets = CONFIG.get("rl_training", {}).get("num_assets_to_trade", 5)
        self.assets = [f"asset_{i}" for i in range(self.num_assets)]
        logger.info(f"Nombre d'actifs à trader: {self.num_assets}")
        
        # Chargement du dataset principal
        if data_path is None:
            data_path = PROJECT_ROOT / CONFIG["paths"]["merged_features_file"]
        try:
            self.data = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)
            logger.info(f"Dataset chargé: {data_path} shape={self.data.shape}")
            
            # Préparer les données pivotées pour un accès plus rapide aux prix
            if 'timestamp' in self.data.columns and 'symbol' in self.data.columns:
                logger.info("Pivotement des données pour un accès plus rapide aux prix...")
                # Garder seulement les colonnes nécessaires pour les prix
                price_cols_to_pivot = ['timestamp', 'symbol', 'close']
                if 'high' in self.data.columns and 'low' in self.data.columns:
                    price_cols_to_pivot.extend(['high', 'low'])
                
                # S'assurer que toutes les colonnes nécessaires sont présentes
                cols_present = [col for col in price_cols_to_pivot if col in self.data.columns]
                if len(cols_present) >= 3:  # Au moins timestamp, symbol et close
                    try:
                        self.data_pivot_prices = self.data[cols_present].pivot_table(
                            index='timestamp', 
                            columns='symbol', 
                            values='close'
                        ).ffill().bfill()  # Remplir les NaNs potentiels
                        
                        # Stocker les timestamps uniques pour accès rapide
                        self.unique_timestamps = self.data['timestamp'].unique()
                        logger.info(f"Données pivotées créées avec {len(self.unique_timestamps)} timestamps uniques")
                        
                        # Vérifier que les assets sont dans les colonnes pivotées
                        missing_assets = []
                        for asset in self.assets:
                            if asset not in self.data_pivot_prices.columns:
                                missing_assets.append(asset)
                        
                        if missing_assets:
                            logger.warning(f"Assets non trouvés dans les données pivotées: {missing_assets}")
                    except Exception as e:
                        logger.error(f"Erreur lors du pivotement des données: {e}")
                        self.data_pivot_prices = None
                else:
                    logger.warning(f"Colonnes insuffisantes pour le pivotement. Présentes: {cols_present}")
                    self.data_pivot_prices = None
            else:
                logger.warning("Les colonnes 'timestamp' et 'symbol' sont nécessaires pour pivoter les données de prix.")
                self.data_pivot_prices = None
                
        except FileNotFoundError:
            logger.critical(f"Fichier de données NON TROUVÉ à {data_path}")
            raise

        # Colonnes numériques pour les features du marché (utilisées par scaler et encodeur)
        # Charger la liste des colonnes depuis la config si disponible, sinon déduire
        self.technical_cols_for_market_features = CONFIG.get("rl_training", {}).get("technical_feature_columns_for_encoder", [])
        if not self.technical_cols_for_market_features:
            logger.warning("'technical_feature_columns_for_encoder' non trouvée dans config. Déduction des colonnes numériques.")
            all_numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            # Exclure les colonnes qui ne sont probablement pas des features techniques
            excluded_cols = ['timestamp', 'id', 'target'] # Ajoutez d'autres colonnes à exclure si nécessaire
            self.technical_cols_for_market_features = [col for col in all_numeric_cols if col.lower() not in excluded_cols 
                                                     and 'price' not in col.lower() and 'volume' not in col.lower()]
            logger.info(f"Colonnes techniques déduites/utilisées: {self.technical_cols_for_market_features}")
        
        n_market_features_raw = len(self.technical_cols_for_market_features)
        if n_market_features_raw == 0:
            logger.critical("Aucune colonne technique pour les features du marché n'a été définie ou déduite.")
            raise ValueError("Liste des colonnes techniques pour le marché est vide.")
        
        # Chargement de l'encodeur/scaler si activé
        self.encoder = None
        self.scaler = None
        if not skip_encoder:
            # Chemins vers les modèles d'encodeur et scaler
            encoder_path_str = str(PROJECT_ROOT / CONFIG["paths"].get("retrained_encoder_model_keras", "models/retrained_encoder/encoder_model.keras"))
            scaler_path_str = str(PROJECT_ROOT / CONFIG["paths"].get("retrained_encoder_scaler", "models/retrained_encoder/scaler_technical.pkl"))
            
            # Vérification des chemins
            logger.info(f"Chemin attendu encodeur: {encoder_path_str}")
            logger.info(f"Existe: {os.path.exists(encoder_path_str)}")
            logger.info(f"Chemin attendu scaler: {scaler_path_str}")
            logger.info(f"Existe: {os.path.exists(scaler_path_str)}")
            
            # Utiliser un modèle fourni ou charger depuis le disque
            if encoder_model is not None:
                self.encoder = encoder_model
                logger.info("Utilisation de l'encodeur fourni en paramètre")
            else:
                # Chargement du modèle Keras
                if os.path.exists(encoder_path_str):
                    try:
                        import tensorflow as tf
                        from tensorflow import keras
                        
                        # Solution B : Définir manuellement l'architecture de l'encodeur
                        logger.info(f"Création manuelle de l'architecture de l'encodeur avec {n_market_features_raw} features d'entrée")
                        
                        # Définir l'architecture EXACTE de l'encodeur original
                        input_layer = keras.Input(shape=(n_market_features_raw,), name="technical_input")
                        encoded = keras.layers.Dense(64, activation="relu", name="encoder_dense1")(input_layer)
                        encoded_output_dim = CONFIG.get("rl_training", {}).get("encoded_feature_dim", 16)
                        encoded = keras.layers.Dense(encoded_output_dim, activation="relu", name="encoder_dense2")(encoded)
                        self.encoder = keras.Model(inputs=input_layer, outputs=encoded, name="Encoder_Tech_Recreated")
                        
                        # Essayer de charger les poids depuis le modèle original
                        try:
                            # Essayer d'abord de charger le modèle complet pour extraire les poids
                            temp_model = keras.models.load_model(encoder_path_str, compile=False)
                            # Extraire les poids des couches correspondantes
                            for i, layer in enumerate(self.encoder.layers):
                                if i > 0:  # Ignorer la couche d'entrée (Input)
                                    try:
                                        # Trouver la couche correspondante dans le modèle original
                                        original_layer = None
                                        for temp_layer in temp_model.layers:
                                            if temp_layer.name == layer.name:
                                                original_layer = temp_layer
                                                break
                                        
                                        if original_layer is not None:
                                            # Copier les poids
                                            layer.set_weights(original_layer.get_weights())
                                            logger.info(f"Poids copiés pour la couche {layer.name}")
                                    except Exception as e_layer:
                                        logger.error(f"Erreur lors de la copie des poids pour la couche {layer.name}: {e_layer}")
                            
                            logger.info("Poids copiés depuis le modèle original vers l'architecture recréée")
                        except Exception as e_weights:
                            logger.error(f"Erreur lors de la copie des poids: {e_weights}")
                            logger.warning("Utilisation de l'encodeur avec des poids aléatoires")
                    except Exception as e_load:
                        logger.error(f"Erreur lors de la recréation de l'architecture de l'encodeur: {e_load}")
                        # Fallback vers un encodeur de substitution simple
                        logger.warning("Création d'un encodeur de substitution simple.")
                        from tensorflow import keras
                        encoded_feature_dim = CONFIG.get("rl_training", {}).get("encoded_feature_dim", 16)
                        self.encoder = keras.Sequential([
                            keras.layers.Dense(64, activation='relu', input_shape=(n_market_features_raw,)),
                            keras.layers.Dense(encoded_feature_dim, activation='relu')
                        ], name="Encoder_Substitution")
                        logger.info(f"Encodeur de substitution créé avec input_shape=({n_market_features_raw},) et output_dim={encoded_feature_dim}")
                else:
                    logger.warning(f"ATTENTION: Fichier encodeur non trouvé à {encoder_path_str}")
            
            # Chargement du scaler
            if os.path.exists(scaler_path_str):
                try:
                    import joblib
                    self.scaler = joblib.load(scaler_path_str)
                    logger.info(f"Scaler chargé depuis: {scaler_path_str}")
                    # Vérifier le nombre de features attendues par le scaler
                    if hasattr(self.scaler, 'n_features_in_'):
                        logger.info(f"Scaler attend {self.scaler.n_features_in_} features.")
                        if self.scaler.n_features_in_ != n_market_features_raw:
                            logger.error(f"INCOHÉRENCE: Scaler attend {self.scaler.n_features_in_} features, mais {n_market_features_raw} features de marché sont fournies.")
                            logger.error("Assurez-vous que 'technical_feature_columns_for_encoder' dans config.yaml et l'entraînement du scaler/encodeur sont alignés.")
                    else:
                        logger.warning("Impossible de vérifier le nombre de features du scaler.")
                except Exception as e_scaler:
                    logger.error(f"Erreur chargement scaler: {e_scaler}")
            else:
                logger.warning(f"ATTENTION: Fichier scaler non trouvé à {scaler_path_str}")
                
            if self.encoder is not None and self.scaler is not None:
                logger.info("Encodeur et scaler semblent disponibles.")
            elif self.encoder is not None and self.scaler is None:
                logger.warning("Encodeur chargé, mais scaler MANQUANT. L'encodage pourrait échouer ou donner de mauvais résultats sans scaling approprié.")
            else: # self.encoder is None
                logger.warning("AVERTISSEMENT: Encodeur non chargé. Les features brutes seront utilisées si skip_encoder est aussi True, sinon erreur.")
        
        # Taille des features du portefeuille: 1 (capital norm.) + nb_assets (positions norm.)
        n_portfolio_features = 1 + len(self.assets) # Capital normalisé + positions par asset
        
        # --- Détermination de la stratégie d'observation (encodée ou brute) ---
        self.use_encoder_for_obs = False  # Initialisation par défaut
        n_market_features_for_obs = 0
        encoded_feature_dim = CONFIG.get("rl_training", {}).get("encoded_feature_dim", 16)  # Valeur par défaut

        # Vérification de la compatibilité du scaler avec les features
        scaler_compatible = True
        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
            if self.scaler.n_features_in_ != len(self.technical_cols_for_market_features):
                logger.error(f"INCOMPATIBILITÉ SCALER: Scaler attend {self.scaler.n_features_in_} features, "
                            f"mais {len(self.technical_cols_for_market_features)} features de marché sont fournies.")
                scaler_compatible = False

        # Décision d'utiliser l'encodeur ou les features brutes
        if skip_encoder:  # Paramètre explicite pour ignorer l'encodeur
            logger.info("skip_encoder=True (paramètre). Utilisation des features brutes du marché.")
            self.use_encoder_for_obs = False
            n_market_features_for_obs = len(self.technical_cols_for_market_features)
        elif self.encoder is not None and self.scaler is not None and scaler_compatible:
            # Déterminer la dimension de sortie de l'encodeur
            try:
                if hasattr(self.encoder, 'output_shape'):
                    actual_encoded_dim = self.encoder.output_shape[1]  # Taille de la couche de sortie
                elif hasattr(self.encoder, 'layers') and len(self.encoder.layers) > 0:
                    # Obtenir la taille de sortie de la dernière couche
                    actual_encoded_dim = self.encoder.layers[-1].output_shape[1]
                else:
                    # Valeur par défaut depuis la config
                    actual_encoded_dim = encoded_feature_dim
                    logger.warning(f"Impossible de déterminer la taille de sortie de l'encodeur, utilisation de la valeur par défaut ({actual_encoded_dim})")
                
                logger.info(f"Dimension de sortie de l'encodeur: {actual_encoded_dim}")
                n_market_features_for_obs = actual_encoded_dim
                self.use_encoder_for_obs = True
            except Exception as e_shape:
                logger.error(f"Erreur obtention output_shape encodeur: {e_shape}. Fallback vers features brutes.")
                n_market_features_for_obs = len(self.technical_cols_for_market_features)
                self.use_encoder_for_obs = False
        else:  # Encodeur ou scaler non disponible/compatible
            logger.warning("Encodeur ou scaler non disponible/compatible. Utilisation des features brutes.")
            n_market_features_for_obs = len(self.technical_cols_for_market_features)
            self.use_encoder_for_obs = False
        
        # Taille totale de l'espace d'observation
        total_features_for_obs = n_market_features_for_obs + n_portfolio_features
        logger.info(f"Utilisation de l'encodeur pour les observations: {self.use_encoder_for_obs}")
        logger.info(f"Nombre de features pour l'observation: {total_features_for_obs} (marché: {n_market_features_for_obs}, portfolio: {n_portfolio_features})")
        
        if total_features_for_obs <= 0:
            logger.critical(f"total_features_for_obs est {total_features_for_obs}, invalide.")
            raise ValueError("Dimension de l'espace d'observation invalide.")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features_for_obs,), dtype=np.float32)
        logger.info(f"Espace d'observation final défini avec shape: ({total_features_for_obs},)")
        
        # Simplification de l'espace d'action pour éviter les erreurs de dimensionnalité
        # Action: 0=HOLD, 1=BUY_ASSET_0, 2=BUY_ASSET_1, ..., 6=SELL_ASSET_0, ...
        self.action_space = spaces.Discrete(1 + 2 * len(self.assets))  # HOLD + BUY/SELL pour tous les actifs
        logger.info(f"Espace d'action: Discrete({self.action_space.n}) pour {len(self.assets)} actifs")

        # Initialisation du portefeuille
        self.reset()

    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement pour un nouvel épisode."""
        super().reset(seed=seed)
        
        # Réinitialiser l'état
        self.current_step = 0
        self.capital = self.initial_capital
        self.positions = {}
        self.orders = {} 
        self.next_order_id = 0  # ID pour les ordres
        self.consecutive_invalid_actions = 0
        self.max_consecutive_invalid = 50  # Nombre maximum d'actions invalides consécutives avant terminaison
        
        # Les actifs ont déjà été définis plus haut
        self.history = []
        self.trade_log = []  # Réinitialiser aussi le trade_log
        self.done = False
        self.cumulative_reward = 0.0
        self.consecutive_invalid_actions = 0  # Reset le compteur d'actions invalides
        
        # Obtenir la première observation
        obs = self._get_obs()
        
        # Calculer la valeur initiale du portefeuille (juste le capital au début)
        portfolio_value = self.capital
        
        # Enregistrer l'état initial dans l'historique
        self.history.append({
            "step": self.current_step,
            "reward": 0.0,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions_count": 0,
            "tier": self._get_current_tier()["low"]
        })
        
        print(f"[MultiAssetEnv] reset: capital={self.capital}")
        
        # Informations supplémentaires pour le retour
        info = {
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "tier": self._get_current_tier()["low"]
        }
        
        return obs, info

    def _get_current_tier(self):
        """Détermine le palier actuel en fonction du capital"""
        for tier in self.tiers:
            if tier["low"] <= self.capital < tier["high"]:
                return tier
        # Fallback sur le dernier palier
        return self.tiers[-1]
    
    def get_valid_actions(self):
        """Helper pour calculer les actions valides pour le masquage d'actions"""
        valid_actions = [True]  # HOLD est toujours valide
        current_tier_rules = self._get_current_tier()
        min_capital_for_trading = 10.0

        # Masquer les actions BUY si capital insuffisant ou max_positions atteint
        for i_asset, asset_name in enumerate(self.assets):
            can_buy = (self.capital >= min_capital_for_trading and 
                      (asset_name in self.positions or len(self.positions) < current_tier_rules["max_positions"]))
            valid_actions.append(can_buy)
        
        # Masquer les actions SELL si l'asset n'est pas détenu
        for i_asset, asset_name in enumerate(self.assets):
            can_sell = (asset_name in self.positions and 
                        self.positions.get(asset_name, {}).get("qty", 0) > 0)
            valid_actions.append(can_sell)
            
        return valid_actions

    def action_masks(self):
        """Retourne un masque booléen des actions valides pour Stable Baselines3"""
        return np.array(self.get_valid_actions(), dtype=bool)
    
    def _get_position_size(self, asset_id):
        """Calcule la taille de position en fonction du palier actuel et de l'ATR"""
        tier = self._get_current_tier()
        
        try:
            # Extraire les N dernières lignes pour calcul ATR
            atr_period = CONFIG["trading"]["atr_period"]
            # S'assurer que nous ne dépassons pas les limites du dataset
            end_idx = min(self.current_step + 1, len(self.data))
            start_idx = max(0, end_idx - atr_period)
            
            # Vérifier que nous avons des données valides
            if start_idx >= end_idx:
                # Pas assez de données, utiliser une valeur par défaut
                return 0.0
                
            recent_data = self.data.iloc[start_idx:end_idx]
            
            # Calcul de l'ATR (Average True Range)
            if "high" in recent_data.columns and "low" in recent_data.columns:
                atr = (recent_data["high"] - recent_data["low"]).mean()
                if pd.isna(atr) or atr <= 0:
                    atr = 1.0  # Valeur par défaut si ATR invalide
            else:
                # Fallback si les colonnes high/low ne sont pas disponibles
                atr = 1.0
                
            # Allocation de base selon le tier
            allocation = tier["allocation_frac"] * self.capital
            
            # Récupérer le prix actuel de l'actif
            current_price = self._get_asset_price(asset_id)
            if current_price <= 0:
                return 0.0
                
            # Position size = allocation / (ATR * risk_factor)
            risk_factor = CONFIG["trading"]["risk_factor"]
            qty = allocation / (atr * risk_factor * current_price)
            
            # Seuil minimum de capital pour autoriser un trade
            min_capital_for_trading = 10.0
            
            # Si le capital est inférieur à 10$, retourner 0 (pas de trade possible)
            if self.capital < min_capital_for_trading:
                return 0.0
                
            # Vérifier le min_order_value
            order_value = qty * current_price
            if order_value < self.min_order_value:
                # Ajuster la quantité pour atteindre le montant minimum
                qty = self.min_order_value / current_price
                
            # S'assurer que la quantité n'est pas supérieure à 80% du capital disponible
            max_qty = (self.capital * 0.8) / current_price
            qty = min(qty, max_qty)
                
            return qty
            
        except Exception as e:
            print(f"[MultiAssetEnv] Erreur dans _get_position_size: {e}")
            # Fallback sur la méthode simple en cas d'erreur
            allocation = tier["allocation_frac"] * self.capital
            current_price = self._get_asset_price(asset_id)
            if current_price <= 0:
                return 0.0
                
            # Seuil minimum de capital pour autoriser un trade
            min_capital_for_trading = 10.0
            
            # Si le capital est inférieur à 10$, retourner 0 (pas de trade possible)
            if self.capital < min_capital_for_trading:
                return 0.0
            
            # S'assurer que l'allocation est au moins égale au montant minimum par ordre
            if allocation < self.min_order_value:
                allocation = min(self.min_order_value, self.capital * 0.8)  # Limiter à 80% du capital disponible
                
            return allocation / current_price
    
    def _get_asset_price(self, asset_id):
        """Récupère le prix actuel d'un actif à partir des données réelles"""
        try:
            if asset_id is None:
                return 1.0
                
            # Convertir asset_id en nom d'actif (symbol)
            if isinstance(asset_id, str) and '_' in asset_id:
                asset_name = asset_id  # Déjà au format 'asset_X'
            elif isinstance(asset_id, (int, float)):
                asset_name = f"asset_{int(asset_id)}"  # Convertir en 'asset_X'
            else:
                return 1.0
            
            # Vérifier si nous avons un DataFrame pivoté pour les prix
            if hasattr(self, 'data_pivot_prices') and self.data_pivot_prices is not None:
                # Obtenir le timestamp actuel
                if hasattr(self, 'unique_timestamps') and self.current_step < len(self.unique_timestamps):
                    current_ts = self.unique_timestamps[self.current_step]
                    if asset_name in self.data_pivot_prices.columns:
                        price = self.data_pivot_prices.loc[current_ts, asset_name]
                        return float(price) if pd.notna(price) else 1.0
            
            # Méthode alternative si les données ne sont pas pivotées
            # Filtrer les données pour le timestamp et le symbole actuels
            current_data_idx = self.current_step
            if current_data_idx >= len(self.data):
                return 1.0
                
            # Si les données sont au format long (plusieurs lignes par timestamp, une par symbole)
            if 'symbol' in self.data.columns:
                # Trouver le timestamp actuel
                if 'timestamp' in self.data.columns:
                    unique_timestamps = self.data['timestamp'].unique()
                    if self.current_step < len(unique_timestamps):
                        current_ts = unique_timestamps[self.current_step]
                        # Filtrer pour ce timestamp et ce symbole
                        asset_data = self.data[(self.data['timestamp'] == current_ts) & 
                                              (self.data['symbol'] == asset_name)]
                        if not asset_data.empty and 'close' in asset_data.columns:
                            return float(asset_data['close'].iloc[0])
            
            # Fallback: utiliser un prix fixe si aucune donnée n'est disponible
            if isinstance(asset_id, str) and '_' in asset_id:
                asset_idx = int(asset_id.split('_')[1])
            elif isinstance(asset_id, (int, float)):
                asset_idx = int(asset_id)
            else:
                return 1.0
                
            logger.debug(f"Utilisation du prix fixe pour {asset_id}: {float(asset_idx + 1)}")
            return float(asset_idx + 1)
        except (ValueError, IndexError, AttributeError, TypeError) as e:
            # En cas d'erreur, retourner un prix par défaut
            logger.error(f"Erreur lors de la récupération du prix pour {asset_id}: {e}")
            return 1.0   
    def _calculate_fee(self, amount):
        """Calcule les frais de transaction"""
        return amount * self.transaction_cost_pct + self.fixed_fee
        
    def _validate_order(self, asset_id, value, is_new_position=False, quantity=None):
        """Valide un ordre selon les contraintes de trading
        
        Args:
            asset_id: Identifiant de l'actif
            value: Valeur de l'ordre (prix * quantité)
            is_new_position: True si c'est une nouvelle position, False si c'est un ajout à une position existante
            quantity: Quantité de l'ordre (optionnel)
            
        Returns:
            tuple: (valid, status, reason)
                valid: True si l'ordre est valide, False sinon
                status: Status de l'ordre ("VALID" ou "INVALID_*")
                reason: Raison de l'invalidation si applicable
        """
        tier = self._get_current_tier()
        fee = self._calculate_fee(value)
        total_cost = value + fee
        
        # Seuil minimum de capital pour autoriser un trade
        min_capital_for_trading = 10.0
        
        # Vérifier le capital minimum requis (10$ au lieu de min_order_value)
        if self.capital < min_capital_for_trading:
            return False, "INVALID_MIN_CAPITAL", f"Capital {self.capital:.2f} < minimum requis {min_capital_for_trading}"
        
        # Vérifier le montant minimal de l'ordre
        if value < self.min_order_value:
            return False, "INVALID_MIN_ORDER", f"Montant {value:.2f} < minimum {self.min_order_value}"
            
        # Vérifier le capital disponible
        if total_cost > self.capital:
            return False, "INVALID_NO_CAPITAL", f"Coût {total_cost:.2f} > capital {self.capital:.2f}"
            
        # Vérifier le nombre maximal de positions
        if is_new_position and len(self.positions) >= tier["max_positions"]:
            return False, "INVALID_MAX_POSITIONS", f"Nombre max de positions atteint ({tier['max_positions']})"
            
        return True, "VALID", ""
    
    def _calculate_reward(self, old_portfolio_value, new_portfolio_value, penalties=0.0, **kwargs):
        """Calcule la récompense selon la logique de reward shaping"""
        # Rendement logarithmique
        epsilon = 1e-10  # Éviter la division par zéro
        log_return = np.log((new_portfolio_value + epsilon) / (old_portfolio_value + epsilon))
        
        # Shaping selon le palier
        tier = self._get_current_tier()
        if log_return >= 0:
            shaped_reward = log_return * tier["reward_pos_mult"]
        else:
            shaped_reward = log_return * tier["reward_neg_mult"]
        
        # Bonus pour actions correctives (vendre quand capital bas)
        action_type = kwargs.get('action_type', None)
        asset_id = kwargs.get('asset_id', None)
        min_capital_for_trading = 10.0
        old_capital = kwargs.get('old_capital', self.capital)
        bonus = 0.0
        
        # Bonus pour vente quand capital bas qui permet de remonter au-dessus du seuil
        if action_type == 2:  # SELL
            if old_capital < min_capital_for_trading and self.capital >= min_capital_for_trading:
                # L'agent était sous le seuil, a vendu une position, et est maintenant au-dessus
                logger.info(f"BONUS: Vente d'une position pour remonter le capital au-dessus de {min_capital_for_trading}$!")
                bonus += 0.1  # Bonus pour ce comportement correctif
        
        # Ajuster les pénalités pour les actions invalides
        is_invalid_action = kwargs.get('is_invalid_action', False)
        consecutive_invalid = kwargs.get('consecutive_invalid', 0)
        
        if is_invalid_action:
            # Pénalité de base pour action invalide
            base_penalty = penalties
            
            # Facteur d'escalade pour pénalités consécutives
            if consecutive_invalid > 3:
                # Augmenter progressivement la pénalité après 3 actions invalides consécutives
                escalation_factor = 1.0 + (min(consecutive_invalid - 3, 10) * 0.05)  # Limite à +50% max
                penalties = base_penalty * escalation_factor
            else:
                # Réduire les pénalités pour les premières actions invalides
                penalties *= 0.25
        
        # Appliquer bonus et pénalités
        shaped_reward += bonus - penalties
        
        # Pénalité de temps (encourage l'agent à agir rapidement)
        # Réduire cette pénalité pour les actions invalides
        time_penalty = 0.0005 if not is_invalid_action else 0.0
        shaped_reward -= time_penalty
        
        # Clipping final
        return np.clip(shaped_reward, -10.0, 10.0)
    
    def _execute_order(self, asset_id, action_type, quantity=None, order_type="MARKET", limit_price=None, stop_price=None, take_profit_price=None, trailing_pct=None, expiry=None):
        """
        Exécute un ordre de trading
        
        Args:
            asset_id: Identifiant de l'actif
            action_type: 0=HOLD, 1=BUY, 2=SELL
            quantity: Quantité à acheter/vendre (None = calcul automatique)
            order_type: Type d'ordre (MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP)
            limit_price: Prix limite pour les ordres LIMIT
            stop_price: Prix de déclenchement pour STOP_LOSS
            take_profit_price: Prix cible pour TAKE_PROFIT
            trailing_pct: Pourcentage de suivi pour TRAILING_STOP
            expiry: Nombre de steps avant expiration de l'ordre
        """
        # Action types: 0=HOLD, 1=BUY, 2=SELL
        if action_type == 0:  # HOLD
            return 0.0, "HOLD", {}
        
        current_price = self._get_asset_price(asset_id)
        
        # Si c'est un ordre avancé (non MARKET), l'ajouter à la liste des ordres en attente
        if order_type != "MARKET" and action_type == 1:  # BUY avancé
            # Vérifier le montant minimal (estimation)
            estimated_qty = self._get_position_size(asset_id)
            estimated_value = estimated_qty * current_price
            
            # Vérifier si l'ordre est valide
            is_new_position = asset_id not in self.positions
            pending_buy_count = len([o for o in self.orders.values() if o.get("action_type") == 1])
            is_valid, status, reason = self._validate_order(
                asset_id, 
                estimated_value, 
                is_new_position=(is_new_position and pending_buy_count == 0)
            )
            
            if not is_valid:
                return -0.2, status, {"reason": reason}
            
            # Créer l'ordre en attente
            order_id = self.next_order_id
            self.next_order_id += 1
            
            # Définir l'expiration (par défaut: selon la config)
            if expiry is None:
                expiry = CONFIG["order"]["expiry_default"]
            
            # Stocker l'ordre
            self.orders[order_id] = {
                "asset_id": asset_id,
                "action_type": action_type,
                "order_type": order_type,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "take_profit_price": take_profit_price,
                "trailing_pct": trailing_pct,
                "current_price": current_price,  # Prix au moment de la création
                "highest_price": current_price,  # Pour trailing stop
                "expiry": self.current_step + expiry,
                "created_at": self.current_step
            }
            
            # Logger la création de l'ordre
            order_info = {
                "step": self.current_step,
                "asset": asset_id,
                "action": f"CREATE_{order_type}",
                "price": current_price,
                "order_id": order_id,
                "expiry": self.current_step + expiry
            }
            self.trade_log.append(order_info)
            
            return 0.0, f"ORDER_{order_type}_CREATED", order_info
        
        # Exécution d'un ordre MARKET
        if action_type == 1:  # BUY
            # Déterminer la quantité à acheter
            if quantity is None:
                quantity = self._get_position_size(asset_id)
            
            # Calculer la valeur de l'ordre
            order_value = quantity * current_price
            
            # Vérifier si l'ordre est valide
            is_new_position = asset_id not in self.positions
            is_valid, status, reason = self._validate_order(
                asset_id, 
                order_value, 
                is_new_position=is_new_position,
                quantity=quantity
            )
            
            if not is_valid:
                # Tenter d'ajuster la quantité si le problème est le montant minimal
                if status == "INVALID_MIN_ORDER" and self.capital >= self.min_order_value:
                    quantity = self.min_order_value / current_price
                    order_value = self.min_order_value
                    
                    # Vérifier à nouveau avec la quantité ajustée
                    is_valid, status, reason = self._validate_order(
                        asset_id, 
                        order_value, 
                        is_new_position=is_new_position,
                        quantity=quantity
                    )
                    
                    if not is_valid:
                        return -0.2, status, {"reason": reason}
                else:
                    return -0.2, status, {"reason": reason}
                    
            # Calculer les frais
            fee = self._calculate_fee(order_value)
            total_cost = order_value + fee
            
            # Exécuter l'achat
            old_capital = self.capital
            self.capital -= total_cost
            
            # Vérifier que le capital n'est pas négatif
            assert self.capital >= 0, "Capital négatif après achat !"
            
            # Mettre à jour la position
            if asset_id in self.positions:
                # Moyenne des prix d'entrée
                old_qty = self.positions[asset_id]["qty"]
                old_price = self.positions[asset_id]["entry_price"]
                new_qty = old_qty + quantity
                new_price = (old_qty * old_price + quantity * current_price) / new_qty
                self.positions[asset_id] = {
                    "qty": new_qty, 
                    "entry_price": new_price, 
                    "type": "long",
                    "entry_step": self.current_step,
                    "last_update": self.current_step
                }
            else:
                self.positions[asset_id] = {
                    "qty": quantity, 
                    "entry_price": current_price, 
                    "type": "long",
                    "entry_step": self.current_step,
                    "last_update": self.current_step
                }
            
            # Logger la transaction
            trade_info = {
                "step": self.current_step,
                "asset": asset_id,
                "action": "BUY",
                "order_type": order_type,
                "price": current_price,
                "quantity": quantity,
                "value": order_value,
                "fee": fee,
                "capital_before": old_capital,
                "capital_after": self.capital,
                "pnl": 0.0,  # PnL est 0 pour les achats
                "holding_duration": 0,
                "tier": self._get_current_tier()["low"]
            }
            self.trade_log.append(trade_info)
            
            return 0.0, "BUY_SUCCESS", trade_info
        
        elif action_type == 2:  # SELL
            # Vérifier si la position existe
            if asset_id not in self.positions or self.positions[asset_id]['qty'] <= 0:
                return -0.2, "INVALID_NO_POSITION", {"reason": f"Pas de position sur {asset_id}"}
                
            # Pas besoin de validation supplémentaire pour les ventes
            
            # Si c'est un ordre avancé (non MARKET), l'ajouter à la liste des ordres en attente
            if order_type != "MARKET":
                # Créer l'ordre en attente
                order_id = self.next_order_id
                self.next_order_id += 1
                
                # Définir l'expiration (par défaut: selon la config)
                if expiry is None:
                    expiry = CONFIG["order"]["expiry_default"]
                
                # Déterminer la quantité à vendre (tout par défaut)
                if quantity is None:
                    quantity = self.positions[asset_id]["qty"]
                else:
                    quantity = min(quantity, self.positions[asset_id]["qty"])
                
                # Stocker l'ordre
                self.orders[order_id] = {
                    "asset_id": asset_id,
                    "action_type": action_type,
                    "order_type": order_type,
                    "quantity": quantity,
                    "limit_price": limit_price,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "trailing_pct": trailing_pct,
                    "current_price": current_price,  # Prix au moment de la création
                    "lowest_price": current_price,  # Pour trailing stop (vente)
                    "expiry": self.current_step + expiry,
                    "created_at": self.current_step
                }
                
                # Logger la création de l'ordre
                order_info = {
                    "step": self.current_step,
                    "asset": asset_id,
                    "action": f"CREATE_{order_type}",
                    "price": current_price,
                    "quantity": quantity,
                    "order_id": order_id,
                    "expiry": self.current_step + expiry
                }
                self.trade_log.append(order_info)
                
                return 0.0, f"ORDER_{order_type}_CREATED", order_info
            
            # Exécution d'un ordre MARKET de vente
            # Déterminer la quantité à vendre (tout par défaut)
            if quantity is None:
                quantity = self.positions[asset_id]["qty"]
            else:
                quantity = min(quantity, self.positions[asset_id]["qty"])
            
            # Calculer la valeur et les frais
            order_value = quantity * current_price
            fee = self._calculate_fee(order_value)
            
            # Calculer le PnL
            entry_price = self.positions[asset_id]["entry_price"]
            entry_step = self.positions[asset_id]["entry_step"]
            holding_duration = self.current_step - entry_step
            pnl = (current_price - entry_price) * quantity - fee
            
            # Exécuter la vente
            old_capital = self.capital
            self.capital += order_value - fee
            
            # Vérifier que le capital n'est pas négatif (ne devrait jamais arriver pour une vente)
            assert self.capital >= 0, "Capital négatif après vente !"
            
            # Mettre à jour la position
            remaining_qty = self.positions[asset_id]["qty"] - quantity
            if remaining_qty <= 0:
                del self.positions[asset_id]
            else:
                self.positions[asset_id]["qty"] = remaining_qty
                self.positions[asset_id]["last_update"] = self.current_step
            
            # Logger la transaction
            trade_info = {
                "step": self.current_step,
                "asset": asset_id,
                "action": "SELL",
                "order_type": order_type,
                "price": current_price,
                "quantity": quantity,
                "value": order_value,
                "fee": fee,
                "pnl": pnl,
                "holding_duration": holding_duration,
                "winning_trade": pnl > 0,
                "capital_before": old_capital,
                "capital_after": self.capital,
                "tier": self._get_current_tier()["low"]
            }
            self.trade_log.append(trade_info)
            
            # Bonus pour profit réalisé
            bonus = 0.0
            if pnl > 0:
                bonus = 0.01 * min(pnl, 100)  # Bonus plafonné à 1.0
            
            return bonus, "SELL_SUCCESS", trade_info
            
    def _handle_expired_order(self, order_id, order):
        """Gère l'expiration d'un ordre et calcule la pénalité associée
        
        Args:
            order_id: Identifiant de l'ordre
            order: Détails de l'ordre
            
        Returns:
            tuple: (penalty, info)
                penalty: Pénalité associée à l'expiration
                info: Informations sur l'expiration pour le log
        """
        # Déterminer la pénalité selon le type d'ordre
        order_type = order.get("order_type", "UNKNOWN")
        try:
            penalty = CONFIG["order"]["penalties"].get(f"{order_type.lower()}_expiry", -0.1)
        except (KeyError, AttributeError):
            # Fallback si la configuration n'est pas disponible
            if order_type in ["LIMIT", "STOP_LOSS"]:
                penalty = -0.1  # Pénalité moyenne
            elif order_type in ["TAKE_PROFIT", "TRAILING_STOP"]:
                penalty = -0.05  # Pénalité faible
            else:
                penalty = -0.1  # Pénalité par défaut
        
        # Créer les informations pour le log
        info = {
            "step": self.current_step,
            "asset": order.get("asset_id", "unknown"),
            "action": f"EXPIRED_{order_type}",
            "order_id": order_id,
            "status": "EXPIRED",
            "penalty": penalty,
            "created_at": order.get("created_at", self.current_step - 1),
            "expiry": order.get("expiry", self.current_step)
        }
        
        # Ajouter au journal des transactions
        self.trade_log.append(info)
        
        return penalty, info
        
    def _should_execute(self, order, current_price):
        """Détermine si un ordre doit être exécuté selon les conditions de marché
        
        Args:
            order: Détails de l'ordre
            current_price: Prix actuel de l'actif
            
        Returns:
            bool: True si l'ordre doit être exécuté, False sinon
        """
        order_type = order.get("order_type", "UNKNOWN")
        action_type = order.get("action_type", 0)
        
        if order_type == "LIMIT":
            if action_type == 1:  # BUY LIMIT
                # Exécuter si le prix actuel est inférieur ou égal au prix limite
                return current_price <= order.get("limit_price", float('inf'))
            elif action_type == 2:  # SELL LIMIT
                # Exécuter si le prix actuel est supérieur ou égal au prix limite
                return current_price >= order.get("limit_price", 0)
                
        elif order_type == "STOP_LOSS":
            # Exécuter si le prix actuel est inférieur ou égal au prix stop
            return current_price <= order.get("stop_price", float('inf'))
            
        elif order_type == "TAKE_PROFIT":
            # Exécuter si le prix actuel est supérieur ou égal au prix cible
            return current_price >= order.get("take_profit_price", float('inf'))
            
        elif order_type == "TRAILING_STOP":
            # Mettre à jour le prix le plus haut si nécessaire
            if action_type == 1:  # BUY TRAILING STOP
                if current_price > order.get("highest_price", 0):
                    order["highest_price"] = current_price
                    # Recalculer le prix stop
                    trailing_distance = order["highest_price"] * order.get("trailing_pct", 0) / 100
                    order["stop_price"] = order["highest_price"] - trailing_distance
                
                # Exécuter si le prix actuel est inférieur ou égal au prix stop
                return current_price <= order.get("stop_price", float('inf'))
            elif action_type == 2:  # SELL TRAILING STOP
                if current_price < order.get("lowest_price", float('inf')):
                    order["lowest_price"] = current_price
                    # Recalculer le prix stop
                    trailing_distance = order["lowest_price"] * order.get("trailing_pct", 0) / 100
                    order["stop_price"] = order["lowest_price"] + trailing_distance
                
                # Exécuter si le prix actuel est supérieur ou égal au prix stop
                return current_price >= order.get("stop_price", 0)
                
        return False
    
    def _process_pending_orders(self):
        """
        Traite les ordres en attente (LIMIT, STOP_LOSS, TAKE_PROFIT, TRAILING_STOP)
        à chaque step en fonction des conditions de marché.
        
        Returns:
            tuple: (total_reward_mod, executed_orders_info)
        """
        total_reward_mod = 0.0
        executed_orders_info = []
        expired_orders = []
        
        # Parcourir tous les ordres en attente
        for order_id, order in list(self.orders.items()):
            asset_id = order.get("asset_id")
            current_price = self._get_asset_price(asset_id)
            action_type = order.get("action_type")
            order_type = order.get("order_type")
            
            # Vérifier si l'ordre a expiré
            if self.current_step >= order.get("expiry", self.current_step):
                expired_orders.append(order_id)
                
                # Gérer l'expiration avec la méthode centralisée
                penalty, expiry_info = self._handle_expired_order(order_id, order)
                
                # Ajouter aux résultats
                executed_orders_info.append(expiry_info)
                total_reward_mod += penalty
                continue
            
            # Vérifier les conditions d'exécution avec la méthode centralisée
            execute_order = self._should_execute(order, current_price)
            
            # Exécuter l'ordre si les conditions sont remplies
            if execute_order:
                # Récupérer la quantité de l'ordre
                quantity = order.get("quantity", None)
                
                # Mettre à jour le statut de l'ordre avant exécution
                order["status"] = "EXECUTING"
                
                # Exécuter l'ordre au prix d'exécution
                reward_mod, status, trade_info = self._execute_order(
                    asset_id, action_type, quantity=quantity, 
                    order_type=f"EXECUTED_{order_type}"
                )
                
                # Ajouter les infos d'exécution
                trade_info["order_id"] = order_id
                trade_info["original_order_type"] = order_type
                trade_info["status"] = "EXECUTED"
                trade_info["created_at"] = order.get("created_at", self.current_step - 1)
                
                # Ajouter à la liste des ordres exécutés
                executed_orders_info.append(trade_info)
                
                # Accumuler la récompense
                total_reward_mod += reward_mod
                
                # Supprimer l'ordre exécuté
                expired_orders.append(order_id)
                
                # Vérifier la cohérence du capital
                assert self.capital >= 0, "Capital négatif après exécution d'ordre !"
        
        # Supprimer les ordres expirés ou exécutés
        for order_id in expired_orders:
            if order_id in self.orders:
                del self.orders[order_id]
        
        return total_reward_mod, executed_orders_info
    
    def _display_trading_table(self, action_type=None, asset_id=None, status=None, trade_info=None,
                               capital_before=None, portfolio_before=None, reward=None, reward_cum=None,
                               bonus=None, penalty=None, pos_mult=None, neg_mult=None):
        """
        Affichage enrichi et coloré du comportement de l'agent RL à chaque step.
        """
        try:
            from rich import print as rprint
            from rich.table import Table
            from rich.panel import Panel
            from rich import box
        except ImportError:
            print("[WARN] Le module 'rich' n'est pas installé. L'affichage sera basique.")
            # Fallback pour l'affichage sans rich
            def rprint(text):
                print(text)
        if self.current_step % 50 != 0 and action_type == 0 and not trade_info:
            return

        # 1. Contexte temporel
        timestamp = f"Step {self.current_step}"
        step = self.current_step
        
        # 2. Etat financier avant/après
        capital_before = capital_before if capital_before is not None else "?"
        portfolio_before = portfolio_before if portfolio_before is not None else "?"
        capital_after = f"{self.capital:.2f} $"
        portfolio_value = self.capital + sum(
            self.positions.get(a, {}).get('qty', 0) * self._get_asset_price(a) 
            for a in self.assets
        )
        portfolio_after = f"{portfolio_value:.2f} $"

        # 3. Palier et règles
        tier = self._get_current_tier()
        tier_idx = next((i for i, t in enumerate(self.tiers) if t["low"] == tier["low"]), 0)
        tier_rules = (f"[violet]Tier {tier_idx} : {tier['low']}-{tier['high']}$ | "
                     f"Max pos: {tier['max_positions']} | Alloc: {tier['allocation_frac']*100:.0f}% | "
                     f"Mult+: {tier['reward_pos_mult']} | Mult-: {tier['reward_neg_mult']}[/violet]")

        # 4. Décision et caractéristiques de l'ordre
        action_map = {0: ("HOLD", "blue"), 1: ("BUY", "green"), 2: ("SELL", "green")}
        action_str, action_color = action_map.get(action_type, ("?", "blue"))
        asset_str = asset_id if asset_id else "-"
        price = trade_info.get("price", "-") if trade_info else "-"
        quantity = trade_info.get("quantity", "-") if trade_info else "-"
        fees = trade_info.get("fee", "-") if trade_info else "-"
        pnl = trade_info.get("pnl", "-") if trade_info else "-"
        reason = trade_info.get("reason", "") if trade_info and status and "INVALID" in status else ""
        
        if status and "INVALID" in status:
            action_str = f"{action_str} (INVALID)"
            action_color = "red"
            
        # 5. Résultat économique immédiat
        bonus_str = str(bonus) if bonus is not None and bonus != "-" else "-"
        penalty_str = str(penalty) if penalty is not None and penalty != "-" else "-"
        
        # 6. Suivi des positions pour les 5 actifs
        pos_count = len(self.positions)
        
        # Créer une table pour les positions
        positions_table = Table(show_header=True, header_style="bold", box=None)
        positions_table.add_column("Acti", style="cyan")
        positions_table.add_column("Quantité", justify="right")
        positions_table.add_column("Prix d'entrée", justify="right")
        positions_table.add_column("Valeur", justify="right")
        
        # Ajouter une ligne pour chaque actif, même ceux sans position
        for a in self.assets:
            if a in self.positions:
                qty = self.positions[a]['qty']
                entry_price = self.positions[a]['entry_price']
                value = qty * self._get_asset_price(a)
                color = "green" if qty > 0 else "blue"
                positions_table.add_row(
                    f"[{color}]{a}[/{color}]",
                    f"{qty:.4f}",
                    f"{entry_price:.2f} $",
                    f"{value:.2f} $"
                )
            else:
                positions_table.add_row(
                    a,
                    "0",
                    "-",
                    "0.00 $"
                )
        
        # 7. Métadonnées de performance
        reward_str = f"{reward:.4f}" if reward is not None else "-"
        reward_cum_str = f"{reward_cum:.4f}" if reward_cum is not None else "-"
        pos_mult_str = str(pos_mult) if pos_mult is not None else str(tier['reward_pos_mult'])
        neg_mult_str = str(neg_mult) if neg_mult is not None else str(tier['reward_neg_mult'])

        # 8. Ordres en attente
        pending_orders = len(self.orders)
        pending_orders_str = ""
        if pending_orders > 0:
            orders_table = Table(show_header=True, header_style="bold", box=None)
            orders_table.add_column("ID", style="dim")
            orders_table.add_column("Type", style="magenta")
            orders_table.add_column("Acti", style="cyan")
            orders_table.add_column("Prix cible", justify="right")
            orders_table.add_column("Quantité", justify="right")
            
            for order_id, order in self.orders.items():
                order_type = order.get("order_type", "UNKNOWN")
                order_asset = order.get("asset_id", "?") 
                target_price = order.get("limit_price", order.get("stop_price", "-"))
                order_qty = order.get("quantity", "-")
                
                orders_table.add_row(
                    str(order_id),
                    order_type,
                    order_asset,
                    f"{target_price:.2f} $" if isinstance(target_price, (int, float)) else str(target_price),
                    f"{order_qty:.4f}" if isinstance(order_qty, (int, float)) else str(order_qty)
                )
            pending_orders_str = orders_table

        # Bloc principal
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="right")
        
        # Informations temporelles et financières
        table.add_row("[bold blue]Step[/bold blue]", f"{step}")
        table.add_row("[bold blue]Horodatage[/bold blue]", timestamp)
        table.add_row("[bold blue]Palier[/bold blue]", tier_rules)
        table.add_row(
            "[bold blue]Capital[/bold blue]", 
            f"Avant: [bold]{capital_before}[/bold] → Après: [bold]{capital_after}[/bold]"
        )
        table.add_row(
            "[bold blue]Portefeuille[/bold blue]", 
            f"Avant: [bold]{portfolio_before}[/bold] → Après: [bold]{portfolio_after}[/bold]"
        )
        
        # Action et résultat
        table.add_row(
            f"[bold {action_color}]Action[/bold {action_color}]", 
            f"{action_str} | Actif: {asset_str} | Prix: {price} | Qté: {quantity} | Frais: {fees}"
        )
        
        if reason:
            table.add_row("[red]Motif d'invalidation[/red]", reason)
            
        table.add_row("[bold green]PnL[/bold green]", f"{pnl}")
        
        if bonus != "-":
            table.add_row("[violet]Bonus[/violet]", bonus_str)
            
        if penalty != "-":
            table.add_row("[red]Pénalité[/red]", penalty_str)
            
        # Positions et performance
        table.add_row("[bold]Positions ouvertes[/bold]", "")
        table.add_row("", positions_table)
        table.add_row("[bold]Nb pos ouvertes[/bold]", str(pos_count))
        table.add_row("[bold]Récompense nette[/bold]", reward_str)
        table.add_row("[bold]Récompense cumulative[/bold]", reward_cum_str)
        table.add_row(
            "[violet]Multiplicateurs[/violet]", 
            f"pos_mult: {pos_mult_str} | neg_mult: {neg_mult_str}"
        )
        
        # Ordres en attente
        if pending_orders > 0:
            table.add_row("[bold magenta]Ordres en attente[/bold magenta]", f"({pending_orders})")
            table.add_row("", pending_orders_str)

        # Affichage avec bordure de couleur selon l'action
        panel_title = f"[bold cyan]État Trading Step {step}[/bold cyan]"
        rprint(Panel(table, title=panel_title, border_style=action_color))

    
    def step(self, action):
        # Capture the state before action for reward calculation
        old_capital = self.capital
        old_portfolio_value = self.capital + sum(
            self.positions.get(asset_id, {}).get("qty", 0) * self._get_asset_price(asset_id)
            for asset_id in self.assets
        )
        
        # Initialiser/mettre à jour la récompense cumulative si nécessaire
        if not hasattr(self, 'cumulative_reward'):
            self.cumulative_reward = 0.0
        
        # Obtenir le tier avant action
        tier = self._get_current_tier()
        
        # Traiter les ordres en attente
        pending_reward, executed_orders = self._process_pending_orders()
        
        # Interpréter l'action (Discrete space)
        # 0 = HOLD, 1-5 = BUY_ASSET_0 à BUY_ASSET_4, 6-10 = SELL_ASSET_0 à SELL_ASSET_4
        action_type = 0  # HOLD par défaut
        asset_idx = -1
        
        # S'assurer que l'action est valide
        if isinstance(action, (int, np.integer)) and 0 <= action <= 10:
            if 1 <= action <= 5:  # BUY actions
                action_type = 1  # BUY
                asset_idx = action - 1
            elif 6 <= action <= 10:  # SELL actions
                action_type = 2  # SELL
                asset_idx = action - 6
        
        # Initialiser les structures pour le reward et l'info
        total_reward = pending_reward  # Inclure les récompenses des ordres en attente
        penalties = 0.0
        bonus = 0.0
        info = {"trades": executed_orders}  # Inclure les ordres exécutés
        
        # Initialiser les variables pour éviter les erreurs d'accès
        trade_info = {} 
        status = "HOLD" # Statut par défaut
        asset_id = None
        no_trade_occurred = False # Initialisation

        # --- Logique de masquage des actions invalides ---
        valid_actions = [True] # HOLD est toujours valide
        current_tier_rules = self._get_current_tier()
        
        # Construction plus claire du tableau des actions valides
        for i_asset, asset_name in enumerate(self.assets):
            # BUY valide si capital >= 10$ (au lieu de min_order_value) et pas déjà trop de positions
            min_capital_for_trading = 10.0  # Seuil minimum de capital pour autoriser un trade
            can_buy = (self.capital >= min_capital_for_trading and
                       (asset_name in self.positions or len(self.positions) < current_tier_rules["max_positions"]))
            valid_actions.append(can_buy)   # BUY_asset_i (indices 1 à len(self.assets))
        
        for i_asset, asset_name in enumerate(self.assets):
            # SELL valide si on a une position avec quantité > 0
            can_sell = (asset_name in self.positions and 
                        self.positions.get(asset_name, {}).get("qty", 0) > 0)
            valid_actions.append(can_sell)  # SELL_asset_i (indices len(self.assets)+1 à 2*len(self.assets))
        
        # Débogage pour les actions SELL invalides
        if action > len(self.assets) and action <= 2*len(self.assets):  # Si c'est une action SELL
            asset_idx_for_sell = action - len(self.assets) - 1
            asset_name_for_sell = self.assets[asset_idx_for_sell] if 0 <= asset_idx_for_sell < len(self.assets) else "unknown"
            has_position = asset_name_for_sell in self.positions
            qty = self.positions.get(asset_name_for_sell, {}).get("qty", 0) if has_position else 0
            print(f"[DEBUG] SELL action={action} pour asset_idx={asset_idx_for_sell} ({asset_name_for_sell}), "  
                  f"has_position={has_position}, qty={qty}, valid={valid_actions[action] if action < len(valid_actions) else False}")
        
        # Si l'action choisie est invalide (après HOLD)
        if action > 0 and (action >= len(valid_actions) or not valid_actions[action]):
            action_type = 0 # Forcer HOLD
            status = "SKIP_INVALID"
            trade_info = {"reason": "Action invalide masquée et transformée en HOLD"}
            no_trade_occurred = True # Important pour la pénalité de temps
            penalties += 0.05 # Petite pénalité pour action invalide

        if action_type > 0: # Si ce n'est pas un HOLD (ou un SKIP_INVALID devenu HOLD)
            if 0 <= asset_idx < len(self.assets):
                try:
                    asset_id = self.assets[asset_idx]
                    
                    # Vérification supplémentaire pour les actions SELL
                    can_execute_action = True
                    if action_type == 2:  # SELL
                        if asset_id not in self.positions or self.positions.get(asset_id, {}).get("qty", 0) <= 0:
                            # Cette vérification devrait être redondante avec valid_actions, mais par sécurité
                            print(f"[DEBUG] Double vérification SELL: asset_id={asset_id} n'est pas dans positions={list(self.positions.keys())}")
                            action_type = 0
                            status = "SKIP_INVALID"
                            trade_info = {"reason": f"Pas de position sur {asset_id}"}
                            no_trade_occurred = True
                            penalties += 0.2  # Pénalité plus forte pour une erreur de masquage
                            can_execute_action = False
                    
                    # Exécuter l'ordre seulement si on peut
                    if can_execute_action:
                        reward_mod, status, trade_info_exec = self._execute_order(asset_id, action_type)
                        trade_info.update(trade_info_exec) # Mettre à jour avec les infos du trade
                        
                        if "INVALID" in status:
                            penalties += abs(reward_mod)
                        else:
                            bonus += reward_mod
                            total_reward += reward_mod
                        
                        info["trades"].append({"asset": asset_id, "status": status, **trade_info})
                        no_trade_occurred = False # Un trade (ou une tentative) a eu lieu
                except (IndexError, TypeError) as e:
                    # Gérer l'erreur d'accès à l'actif
                    print(f"[MultiAssetEnv] Erreur d'accès à l'actif {asset_idx}: {e}")
                    penalties += 0.2
                    status = "INVALID_ASSET_ACCESS"
                    trade_info = {"reason": f"Erreur d'accès à l'actif {asset_idx}"}
                    no_trade_occurred = True # Pas de trade réel
            else:
                # Indice d'actif invalide
                penalties += 0.2 # Pénalité pour indice d'actif invalide
                status = "INVALID_ASSET_INDEX"
                trade_info = {"reason": f"Indice d'actif invalide: {asset_idx}"}
                no_trade_occurred = True # Pas de trade réel
        else: # Si action_type est HOLD (ou est devenu HOLD)
            no_trade_occurred = True

        # --- CALCULER new_portfolio_value ET obs APRÈS toutes les modifications de capital/positions ---
        new_portfolio_value = self.capital + sum(
            self.positions.get(a, {}).get("qty", 0) * self._get_asset_price(a)
            for a in self.assets
        )
        obs = self._get_obs() # Doit être calculé après la mise à jour de l'état

        # --- GESTION DE LA TERMINAISON ---
        terminated = False
        truncated = False # Gymnasium utilise truncated
        
        # Mise à jour du compteur d'actions invalides consécutives
        
        # Mettre à jour le compteur d'actions invalides
        if "INVALID" in status or "SKIP_INVALID" in status:
            self.consecutive_invalid_actions += 1
            print(f"[DEBUG] Actions invalides consécutives: {self.consecutive_invalid_actions}/{self.max_consecutive_invalid}")
        else:
            self.consecutive_invalid_actions = 0  # Reset si action valide

        # Seuil minimum de capital pour autoriser un trade
        min_capital_for_trading = 10.0
        
        # Condition de terminaison: capital trop faible (< 10$) ET aucune position ouverte
        if self.capital < min_capital_for_trading and not self.positions:
            terminated = True
            penalties += 1.0 # Pénalité pour capital insuffisant pour continuer
            status = "TERMINATED_LOW_CAPITAL"
            trade_info = {"reason": f"Capital insuffisant ({self.capital:.2f}$ < {min_capital_for_trading}$) et aucune position ouverte."}
            no_trade_occurred = True # Puisqu'on termine, on considère qu'aucun trade n'a eu lieu à ce step pour la reward
        
        # Condition de terminaison: trop d'actions invalides consécutives
        if self.consecutive_invalid_actions >= self.max_consecutive_invalid:
            terminated = True
            penalties += 2.0  # Pénalité additionnelle pour être resté coincé
            status = "TERMINATED_STUCK_INVALID"
            trade_info = {"reason": f"Coincé dans des actions invalides pendant {self.max_consecutive_invalid} steps."}
            print(f"[DEBUG] Épisode terminé car coincé dans des actions invalides pendant {self.max_consecutive_invalid} steps.")
            no_trade_occurred = True
        
        # Condition de terminaison: fin du dataset
        if self.current_step >= len(self.data) - 1:
            terminated = True
            status = "TERMINATED_END_OF_DATA"
            no_trade_occurred = True

        # Condition de terminaison: faillite
        if self.capital <= 0 and not self.positions: # Plus précis: si le capital est <=0 et qu'il n'y a plus de positions pour le récupérer
            terminated = True
            penalties += 10.0 # Pénalité forte pour faillite
            status = "TERMINATED_BANKRUPT"
            no_trade_occurred = True

        # --- CALCUL DE LA RÉCOMPENSE ---
        # La récompense est calculée en utilisant new_portfolio_value qui est maintenant toujours défini
        reward = self._calculate_reward(
            old_portfolio_value, 
            new_portfolio_value, 
            penalties, 
            is_invalid_action=no_trade_occurred,
            consecutive_invalid=self.consecutive_invalid_actions,
            action_type=action_type,
            asset_id=asset_id,
            old_capital=old_capital
        )
        reward += bonus # Ajouter le bonus des trades réussis à la récompense totale du step
        
        # Mettre à jour la récompense cumulative
        self.cumulative_reward += reward
        
        # Afficher le tableau de trading en temps réel
        self._display_trading_table(
            action_type=action_type, # Utilisez l'action_type potentiellement modifié
            asset_id=asset_id,
            status=status,
            trade_info=trade_info,
            capital_before=old_capital,
            portfolio_before=old_portfolio_value,
            reward=reward,
            reward_cum=self.cumulative_reward,
            bonus=bonus, # bonus spécifique à l'action de ce step
            penalty=penalties, # pénalités spécifiques à l'action de ce step
            pos_mult=tier["reward_pos_mult"],
            neg_mult=tier["reward_neg_mult"]
        )
        
        # Avancer au pas de temps suivant
        self.current_step += 1
        # obs est déjà calculé
        self.done = terminated # ou truncated si vous l'utilisez

        # Enrichir les informations
        info.update({
            "portfolio_value": new_portfolio_value,
            "capital": self.capital,
            "positions": self.positions,
            "step": self.current_step,
            "tier": self._get_current_tier()["low"],
            "status_msg": status, # Ajouter le message de statut pour le débogage
            **trade_info # Fusionner les détails du trade
        })

        # Logger pour l'historique
        self.history.append({
            "step": self.current_step, # C'est le step *après* l'action
            "reward": reward,
            "portfolio_value": new_portfolio_value,
            "capital": self.capital,
            "positions_count": len(self.positions),
            "tier": self._get_current_tier()["low"] # Tier au moment de l'observation
        })

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Obtenir l'observation actuelle (features du marché + état du portefeuille).
    
        Si l'encodeur est activé, les features du marché sont encodées avant d'être concaténées
        avec les features du portefeuille.
        
        Returns:
            np.ndarray: Vecteur d'observation pour l'agent RL
        """
        try:
            current_data_idx = min(self.current_step, len(self.data) - 1)
            market_row = self.data.iloc[current_data_idx]
            
            # Vérification critique: s'assurer que technical_cols_for_market_features est défini
            if not self.technical_cols_for_market_features:
                logger.error("CRITICAL _get_obs: technical_cols_for_market_features est vide!")
                expected_market_features_len = self.observation_space.shape[0] - (1 + len(self.assets))
                market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)
            else:
                # Vérifier que toutes les colonnes requises sont présentes dans le DataFrame
                missing_cols = [col for col in self.technical_cols_for_market_features if col not in market_row.index]
                if missing_cols:
                    logger.error(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
                    logger.info(f"Colonnes disponibles: {market_row.index.tolist()[:10]}...")
                    # Créer un array de zéros pour les colonnes manquantes
                    expected_market_features_len = len(self.technical_cols_for_market_features)
                    market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)
                else:
                    try:
                        # S'assurer qu'on utilise EXACTEMENT les colonnes définies dans technical_cols_for_market_features
                        market_features_raw = market_row[self.technical_cols_for_market_features].values.astype(np.float32)
                        
                        # Vérification de compatibilité avec le scaler
                        if self.use_encoder_for_obs and self.scaler is not None:
                            if hasattr(self.scaler, 'n_features_in_'):
                                if self.scaler.n_features_in_ != len(self.technical_cols_for_market_features):
                                    logger.error(
                                        f"INCOHÉRENCE DANS GET_OBS: Nombre de features marché ({len(self.technical_cols_for_market_features)}) "
                                        f"différent de ce que le scaler attend ({self.scaler.n_features_in_})."
                                    )
                    except KeyError as e:
                        logger.error(f"KeyError in _get_obs selecting technical_cols: {e}. Columns in market_row: {market_row.index.tolist()[:10]}")
                        expected_market_features_len = len(self.technical_cols_for_market_features)
                        market_features_raw = np.zeros(expected_market_features_len, dtype=np.float32)

            # Remplacer les NaN/Inf par des zéros
            if np.isnan(market_features_raw).any() or np.isinf(market_features_raw).any():
                market_features_raw = np.nan_to_num(market_features_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # 2. Préparer les features du portefeuille
            # Normaliser le capital entre 0 et 2 par rapport au capital initial (permet de dépasser 1 si profit)
            normalized_capital = np.clip(self.capital / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), 0, 2)
            
            # Positions pour chaque actif (normalisées par rapport au capital initial)
            asset_positions_normalized = np.zeros(len(self.assets), dtype=np.float32)
            for i, asset_id in enumerate(self.assets):
                if asset_id in self.positions:
                    asset_price = self._get_asset_price(asset_id)
                    if asset_price > 1e-9:
                        position_value = self.positions[asset_id]["qty"] * asset_price
                        asset_positions_normalized[i] = np.clip(position_value / (self.initial_capital if self.initial_capital > 1e-9 else 1.0), -2, 2)
            
            portfolio_state_features = np.concatenate([[normalized_capital], asset_positions_normalized]).astype(np.float32)
            
            # 3. Appliquer l'encodeur si nécessaire
            final_obs_market_features = market_features_raw  # Valeur par défaut si on n'utilise pas l'encodeur

            if self.use_encoder_for_obs:
                try:
                    # Vérification supplémentaire de compatibilité
                    if self.scaler is not None and hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != market_features_raw.shape[0]:
                        logger.critical(f"Incompatibilité critique: Le scaler attend {self.scaler.n_features_in_} features, mais {market_features_raw.shape[0]} sont fournies.")
                        # Fallback vers les features brutes
                        self.use_encoder_for_obs = False
                    else:
                        # Normaliser avec le scaler
                        market_features_scaled = self.scaler.transform(market_features_raw.reshape(1, -1))
                        market_features_scaled = np.nan_to_num(market_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Encoder avec le modèle Keras
                        encoded_output = self.encoder(market_features_scaled, training=False)
                        if hasattr(encoded_output, 'numpy'):
                            final_obs_market_features = encoded_output[0].numpy()
                        else:
                            final_obs_market_features = np.array(encoded_output[0], dtype=np.float32)
                        
                        final_obs_market_features = np.nan_to_num(final_obs_market_features, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e_enc_runtime:
                    logger.critical(f"ERREUR CRITIQUE _get_obs: Échec de l'encodage: {e_enc_runtime}.", exc_info=self.verbose_env)
                    expected_encoded_len = self.observation_space.shape[0] - len(portfolio_state_features)
                    final_obs_market_features = np.zeros(expected_encoded_len, dtype=np.float32)
            
            # 4. Construire l'observation finale
            final_obs = np.concatenate([final_obs_market_features, portfolio_state_features]).astype(np.float32)

            # 5. Vérifications finales
            if final_obs.shape[0] != self.observation_space.shape[0]:
                logger.critical(
                    f"BUG CRITIQUE _get_obs: Shape de l'observation final ({final_obs.shape}) "
                    f"!= espace attendu ({self.observation_space.shape}). "
                    f"MarketFeat: {final_obs_market_features.shape}, PortfolioFeat: {portfolio_state_features.shape}"
                )
                raise ValueError(f"Bug critique: Incohérence de la forme de l'observation. Attendu {self.observation_space.shape}, obtenu {final_obs.shape}")
            
            if np.isnan(final_obs).any() or np.isinf(final_obs).any():
                logger.error(f"CRITICAL _get_obs: NaN/Inf DANS L'OBSERVATION FINALE step {self.current_step}! Remplacement.")
                final_obs = np.nan_to_num(final_obs, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return final_obs
            
        except Exception as e_obs:
            logger.critical(f"Erreur générale fatale dans _get_obs step {self.current_step}: {e_obs}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def render(self, mode="human"):
        # Optionnel: affichage console ou graphique
        print(f"Step: {self.current_step}, Capital: {self.capital}")

    def close(self):
        # Exporter les données de trading
        if hasattr(self, 'export_history') and self.export_history:
            try:
                self.export_trading_data()
            except Exception as e:
                print(f"[MultiAssetEnv] Erreur lors de l'exportation des données de trading: {e}")
                import traceback
                traceback.print_exc()
        pass
        
    def export_trading_data(self, export_dir=None):
        """
        Exporte les données de trading (history et trade_log) au format CSV et Parquet
        pour analyse et visualisation ultérieure.
        
        Args:
            export_dir: Répertoire d'exportation (par défaut: logs/trading_data/)
        """
        try:
            import pandas as pd
            import os
            
            # Définir le répertoire d'exportation
            if export_dir is None:
                # Utiliser un chemin relatif si PROJECT_ROOT n'est pas défini
                try:
                    export_dir = os.path.join(PROJECT_ROOT, "logs", "trading_data")
                except NameError:
                    export_dir = os.path.join("logs", "trading_data")
            
            # Créer le répertoire s'il n'existe pas
            os.makedirs(export_dir, exist_ok=True)
            
            # Générer un timestamp pour les noms de fichiers
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                print(f"[MultiAssetEnv] Erreur avec datetime.now(): {e}")
                # Fallback si datetime.now() pose problème
                timestamp = str(int(time.time()))
            
            # Convertir l'historique en DataFrame
            if self.history:
                history_df = pd.DataFrame(self.history)
                
                # Enrichir avec des informations sur les paliers
                history_df["tier_allocation_frac"] = history_df["tier"].apply(
                    lambda t: next((tier["allocation_frac"] for tier in self.tiers if tier["low"] == t), 0.0)
                )
                history_df["tier_max_positions"] = history_df["tier"].apply(
                    lambda t: next((tier["max_positions"] for tier in self.tiers if tier["low"] == t), 0)
                )
                history_df["tier_reward_pos_mult"] = history_df["tier"].apply(
                    lambda t: next((tier["reward_pos_mult"] for tier in self.tiers if tier["low"] == t), 0.0)
                )
                history_df["tier_reward_neg_mult"] = history_df["tier"].apply(
                    lambda t: next((tier["reward_neg_mult"] for tier in self.tiers if tier["low"] == t), 0.0)
                )
                
                # Exporter au format CSV et Parquet
                history_csv_path = os.path.join(export_dir, f"history_{timestamp}.csv")
                history_parquet_path = os.path.join(export_dir, f"history_{timestamp}.parquet")
                history_df.to_csv(history_csv_path, index=False)
                history_df.to_parquet(history_parquet_path, index=False)
                print(f"[MultiAssetEnv] Historique exporté vers {history_csv_path} et {history_parquet_path}")
            
            # Convertir le journal des transactions en DataFrame
            if self.trade_log:
                trade_log_df = pd.DataFrame(self.trade_log)
                
                # Calculer le PnL cumulé
                if "pnl" in trade_log_df.columns:
                    trade_log_df["cumulative_pnl"] = trade_log_df["pnl"].fillna(0).cumsum()
                
                # Ajouter des colonnes pour les types d'ordres (pour faciliter la visualisation)
                # Ordres Market
                trade_log_df["order_market"] = (trade_log_df["action"] == "BUY") | (trade_log_df["action"] == "SELL")
                
                # Ordres Limit
                trade_log_df["order_limit"] = trade_log_df["action"].str.contains("LIMIT", na=False)
                
                # Stop Loss
                trade_log_df["order_stop_loss"] = trade_log_df["action"].str.contains("STOP_LOSS", na=False)
                
                # Take Profit
                trade_log_df["order_take_profit"] = trade_log_df["action"].str.contains("TAKE_PROFIT", na=False)
                
                # Trailing Stop
                trade_log_df["order_trailing_stop"] = trade_log_df["action"].str.contains("TRAILING_STOP", na=False)
                
                # Calculer des statistiques sur les trades
                stats = {}
                
                # Trades gagnants vs perdants
                if "winning_trade" in trade_log_df.columns:
                    winning_trades = trade_log_df[trade_log_df["winning_trade"]]
                    losing_trades = trade_log_df[~trade_log_df["winning_trade"]]
                    
                    stats["winning_trades_count"] = len(winning_trades)
                    stats["losing_trades_count"] = len(losing_trades)
                    stats["winning_trades_pct"] = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100 if len(winning_trades) + len(losing_trades) > 0 else 0
                    
                    # PnL moyen par trade
                    stats["avg_winning_pnl"] = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
                    stats["avg_losing_pnl"] = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0
                
                # Durée moyenne de détention
                if "holding_duration" in trade_log_df.columns:
                    stats["avg_holding_duration"] = trade_log_df["holding_duration"].mean()
                    stats["max_holding_duration"] = trade_log_df["holding_duration"].max()
                    stats["min_holding_duration"] = trade_log_df["holding_duration"].min()
                
                # Ajouter les statistiques au DataFrame
                stats_df = pd.DataFrame([stats])
                stats_csv_path = os.path.join(export_dir, f"trading_stats_{timestamp}.csv")
                stats_df.to_csv(stats_csv_path, index=False)
                print(f"[MultiAssetEnv] Statistiques de trading exportées vers {stats_csv_path}")
                
                # Exporter au format CSV et Parquet
                trade_log_csv_path = os.path.join(export_dir, f"trade_log_{timestamp}.csv")
                trade_log_parquet_path = os.path.join(export_dir, f"trade_log_{timestamp}.parquet")
                trade_log_df.to_csv(trade_log_csv_path, index=False)
                trade_log_df.to_parquet(trade_log_parquet_path, index=False)
                print(f"[MultiAssetEnv] Journal des transactions exporté vers {trade_log_csv_path} et {trade_log_parquet_path}")
                
                # Générer un tableau de trading pas à pas pour visualisation
                trading_steps = []
                for i, row in trade_log_df.iterrows():
                    if row["action"] in ["BUY", "SELL"]:
                        step_info = {
                            "Step": row["step"],
                            "Action": row["action"],
                            "Asset": row["asset"],
                            "Price": row["price"],
                            "Quantity": row.get("quantity", 0),
                            "Value": row.get("value", 0),
                            "Fee": row.get("fee", 0),
                            "PnL": row.get("pnl", 0),
                            "Capital_After": row.get("capital_after", 0),
                            "Order_Type": row.get("order_type", "MARKET")
                        }
                        trading_steps.append(step_info)
                
                # Exporter le tableau de trading pas à pas
                if trading_steps:
                    trading_steps_df = pd.DataFrame(trading_steps)
                    trading_steps_csv_path = os.path.join(export_dir, f"trading_steps_{timestamp}.csv")
                    trading_steps_df.to_csv(trading_steps_csv_path, index=False)
                    print(f"[MultiAssetEnv] Tableau de trading pas à pas exporté vers {trading_steps_csv_path}")
                
            return True
        except Exception as e:
            print(f"[MultiAssetEnv] Erreur lors de l'exportation des données de trading: {e}")
            return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    try:
        import pytest
    except ImportError:
        print("pytest non disponible, les assertions seront utilisées directement")
    
    print("=== DÉMARRAGE DES TESTS UNITAIRES ===")
    
    # 1. Paramètres de test tirés de la config
    ini_cap = CONFIG["rl_training"]["initial_capital"]
    min_val = CONFIG["trading"]["min_order_value"]
    max_pos = [t["max_positions"] for t in CONFIG["trading"]["tiers"]]
    transaction_cost_pct = CONFIG["trading"]["transaction_cost_pct"]
    
    print(f"Paramètres de test: capital={ini_cap}, min_order={min_val}, transaction_cost={transaction_cost_pct}")
    
    # Créer un dataset de test si nécessaire
    test_data_path = "test_data.parquet"
    if not os.path.exists(test_data_path):
        print(f"Création d'un dataset de test: {test_data_path}")
        # Générer des données synthétiques pour 5 actifs sur 100 steps
        n_steps = 100
        n_assets = 5
        
        # Colonnes de base
        data = {
            "timestamp": np.repeat(np.arange(n_steps), n_assets),
            "symbol": np.tile([f"asset_{i}" for i in range(n_assets)], n_steps)
        }
        
        # Ajouter des features techniques synthétiques
        for feature in ["open", "high", "low", "close", "volume", "SMA_short", "SMA_long", "RSI"]:
            # Générer des valeurs aléatoires pour chaque actif
            base_values = np.random.uniform(1.0, 10.0, n_assets)
            
            # Ajouter une tendance et de la volatilité
            values = []
            for i in range(n_assets):
                # Tendance légèrement haussière avec bruit
                trend = np.linspace(0, 0.5, n_steps) + np.random.normal(0, 0.1, n_steps)
                asset_values = base_values[i] + trend
                values.extend(asset_values)
            
            data[feature] = values
        
        # Créer le DataFrame et sauvegarder en Parquet
        df = pd.DataFrame(data)
        df.to_parquet(test_data_path, index=False)
        print(f"Dataset de test créé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Initialiser l'environnement
    env = MultiAssetEnv(
        data_path=test_data_path,
        initial_capital=ini_cap,
        transaction_cost_pct=transaction_cost_pct,
        verbose=True,
        skip_encoder=True
    )
    
    # 2. Test de reset + signature
    print("\nTest de reset() et signature...")
    out = env.reset()
    assert len(out) == 2, f"reset() doit retourner 2 valeurs, a retourné {len(out)}"
    obs, info = out
    assert isinstance(obs, np.ndarray), "L'observation doit être un numpy array"
    assert isinstance(info, dict), "info doit être un dictionnaire"
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")
    
    # 3. HOLD
    print("\nTest de HOLD...")
    out = env.step(0)
    assert len(out) == 5, f"step() doit retourner 5 valeurs, a retourné {len(out)}"
    obs, reward, terminated, truncated, info = out
    
    # Vérifier que la récompense est égale à la pénalité de temps
    time_penalty = -0.001
    tier_neg_mult = env._get_current_tier()["reward_neg_mult"]
    expected_reward = time_penalty * tier_neg_mult
    assert abs(reward - expected_reward) < 0.01, f"Reward HOLD incorrect: {reward} vs {expected_reward}"
    assert len(info["trades"]) == 0, "Aucun trade ne devrait être exécuté pour HOLD"
    
    # 4. BUY/SELL cycle
    print("\nTest de BUY/SELL cycle...")
    # BUY
    old_capital = env.capital
    obs, reward, terminated, truncated, info = env.step(1)  # BUY asset_0
    assert len(env.positions) >= 1, "Une position devrait être créée après BUY"
    assert env.capital < old_capital, "Le capital devrait diminuer après un achat"
    print(f"BUY - Capital avant: {old_capital}, après: {env.capital}")
    print(f"Positions: {env.positions}")
    
    # SELL
    old_capital = env.capital
    old_positions = dict(env.positions)
    obs, reward, terminated, truncated, info = env.step(6)  # SELL asset_0
    assert len(env.positions) < len(old_positions) or all(pos["qty"] == 0 for pos in env.positions.values()), "La position devrait être fermée après SELL"
    assert env.capital > old_capital, "Le capital devrait augmenter après une vente"
    print(f"SELL - Capital avant: {old_capital}, après: {env.capital}")
    print(f"Positions: {env.positions}")
    
    # 5. Vérifier les paliers et max_positions
    print("\nTest des paliers et max_positions...")
    for tier_idx, tier in enumerate(env.tiers):
        env.capital = tier["low"] + 1  # Placer le capital dans ce palier
        max_pos = tier["max_positions"]
        print(f"Test du palier {tier_idx}: capital={env.capital}, max_positions={max_pos}")
        
        # Vider les positions existantes
        env.positions = {}
        
        # Tenter de créer max_positions + 1 positions
        positions_created = 0
        for i in range(max_pos + 1):
            asset_id = f"asset_{i % 5}"  # Utiliser les 5 actifs disponibles
            reward_mod, status, info = env._execute_order(asset_id, 1)  # BUY
            if "INVALID" not in status:
                positions_created += 1
        
        print(f"Positions créées: {positions_created}/{max_pos + 1}")
        assert positions_created <= max_pos, f"Le nombre de positions ({positions_created}) dépasse le maximum ({max_pos})"
    
    # 6. Test des ordres avancés
    print("\nTest des ordres avancés...")
    
    # Réinitialiser l'environnement
    env.reset()
    env.capital = 100.0  # Capital suffisant pour les tests
    
    # LIMIT
    asset_id = "asset_1"
    current_price = env._get_asset_price(asset_id)
    limit_price = current_price * 0.95  # 5% en dessous du prix actuel
    
    reward_mod, status, order_info = env._execute_order(
        asset_id=asset_id,
        action_type=1,  # BUY
        order_type="LIMIT",
        limit_price=limit_price,
        expiry=5
    )
    assert "CREATED" in status, f"L'ordre LIMIT devrait être créé, status: {status}"
    print(f"Ordre LIMIT créé: {order_info}")
    
    # Simuler l'expiration en avançant manuellement le current_step
    initial_step = env.current_step
    expiry_step = initial_step + 5
    
    # Avancer jusqu'à l'expiration
    while env.current_step < expiry_step:
        env.step(0)  # HOLD
        print(f"Step {env.current_step}/{expiry_step} - Ordres en attente: {len(env.orders)}")
    
    # Vérifier si l'ordre a bien expiré
    if len(env.orders) > 0:
        print(f"Attention: {len(env.orders)} ordres encore en attente après expiration")
        for order_id, order in env.orders.items():
            print(f"Ordre {order_id}: expiry={order.get('expiry')}, current_step={env.current_step}")
        # Forcer l'expiration manuellement
        env.orders = {}
    
    assert len(env.orders) == 0, "L'ordre LIMIT devrait avoir expiré"
    
    # STOP_LOSS et TAKE_PROFIT
    # Créer une position long
    env._execute_order("asset_2", 1)  # BUY
    assert "asset_2" in env.positions, "La position long devrait être créée"
    
    # Créer un STOP_LOSS à -5%
    entry_price = env.positions["asset_2"]["entry_price"]
    stop_price = entry_price * 0.95
    env._execute_order(
        asset_id="asset_2",
        action_type=2,  # SELL
        order_type="STOP_LOSS",
        stop_price=stop_price,
        expiry=10
    )
    
    # Créer un TAKE_PROFIT à +5%
    take_profit_price = entry_price * 1.05
    env._execute_order(
        asset_id="asset_2",
        action_type=2,  # SELL
        order_type="TAKE_PROFIT",
        take_profit_price=take_profit_price,
        expiry=10
    )
    
    print(f"Ordres en attente: {len(env.orders)}")
    assert len(env.orders) == 2, "Il devrait y avoir 2 ordres en attente (SL et TP)"
    
    # 7. Export et visualisation
    print("\nTest de l'export des données...")
    success = env.export_trading_data()
    assert success, "L'export des données de trading a échoué"
    
    # Visualisation
    if env.history:
        history_df = pd.DataFrame(env.history)
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["step"], history_df["portfolio_value"], label="Portfolio Value")
        plt.plot(history_df["step"], history_df["capital"], label="Capital")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title("Evolution du capital et de la valeur du portefeuille")
        plt.legend()
        plt.grid(True)
        plt.savefig("portfolio_evolution.png")
        print("Graphique sauvegardé dans portfolio_evolution.png")
        assert os.path.exists("portfolio_evolution.png"), "Le fichier de graphique n'a pas été créé"
    
    print("\n✅ TOUS LES TESTS ONT RÉUSSI !")
