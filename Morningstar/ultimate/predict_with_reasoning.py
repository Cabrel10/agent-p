#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour faire des pru00e9dictions avec le modu00e8le de raisonnement.
Ce script utilise le modu00e8le entrau00eenu00e9 pour faire des pru00e9dictions sur de nouvelles donnu00e9es.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, StandardScaler
import json # Ajout pour le log des commandes

from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
# Importer ReasoningModule ET ExplanationDecoder
from model.reasoning.reasoning_module import ReasoningModule, ExplanationDecoder
from config.config import Config  # Importer la classe Config
from telegram_bot import notify_trade_sync  # Importer la fonction de notification Telegram

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Arguments parsu00e9s
    """
    parser = argparse.ArgumentParser(description="Pru00e9dictions avec le modu00e8le de raisonnement")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/real/final_dataset.parquet",
        help="Chemin vers le dataset de donnu00e9es ru00e9elles",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/predictions",
        help="Ru00e9pertoire de sortie pour les pru00e9dictions",
    )

    return parser.parse_args()


def preprocess_data(data_path):
    """
    Pru00e9traite les donnu00e9es pour les pru00e9dictions.

    Args:
        data_path: Chemin vers le dataset

    Returns:
        Donnu00e9es pru00e9traitu00e9es et noms des features
    """
    # Charger les donnu00e9es
    logger.info(f"Chargement des donnu00e9es depuis {data_path}")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")

    # Identifier les diffu00e9rents types de colonnes
    price_cols = ["open", "high", "low", "close", "volume"]
    technical_cols = [
        col
        for col in df.columns
        if col
        not in [
            "timestamp",
            "symbol",
            "market_regime",
            "level_sl",
            "level_tp",
            "hmm_regime",
            "hmm_prob_0",
            "hmm_prob_1",
            "hmm_prob_2",
            "split",
        ]
        and not col.startswith("llm_")
        and not col.startswith("mcp_")
        and not col.startswith("sentiment_")
        and not col.startswith("cryptobert_")
        and not col.startswith("market_info_")
    ]

    # Exclure les colonnes non numu00e9riques
    exclude_cols = ["timestamp", "symbol", "split"]
    technical_cols = [
        col
        for col in technical_cols
        if col not in exclude_cols and df[col].dtype != "object" and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    # Normaliser les donnu00e9es manuellement
    logger.info("Normalisation des donnu00e9es")
    df_norm = df.copy()

    # Normalisation simple des donnu00e9es numu00e9riques
    for col in price_cols + technical_cols:
        if col in df.columns and df[col].dtype != "object" and not pd.api.types.is_datetime64_any_dtype(df[col]):
            # Normalisation robuste (similaire u00e0 RobustScaler)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df_norm[col] = (df[col] - q1) / iqr
            else:
                # Fallback u00e0 la normalisation standard
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                else:
                    df_norm[col] = 0

    # Colonnes LLM (si pru00e9sentes)
    llm_cols = [col for col in df.columns if col.startswith("llm_")]
    if not llm_cols:  # Si pas de colonnes LLM, cru00e9er un vecteur vide
        df_norm["llm_dummy"] = 0.0
        llm_cols = ["llm_dummy"]

    # Colonnes MCP (si pru00e9sentes)
    mcp_cols = [col for col in df.columns if col.startswith("mcp_")]
    if not mcp_cols:  # Si pas de colonnes MCP, cru00e9er un vecteur vide
        df_norm["mcp_dummy"] = 0.0
        mcp_cols = ["mcp_dummy"]

    # Colonnes HMM
    # hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'] # Ancienne logique statique

    # Nouvelle logique dynamique basée sur la config (patch)
    cfg = Config()  # Charger la config pour lire num_hmm
    num_hmm = cfg.get_config("model.num_hmm", 0)  # Lire depuis la section model

    if num_hmm == 1:
        # Single regime label
        hmm_cols = ["hmm_regime"]
        logger.info("Utilisation de 'hmm_regime' comme feature HMM (num_hmm=1)")
    elif num_hmm > 1:
        # Probability columns
        hmm_cols = [f"hmm_prob_{i}" for i in range(num_hmm)]
        logger.info(f"Utilisation de {hmm_cols} comme features HMM (num_hmm={num_hmm})")
    else:
        # Pas de features HMM
        hmm_cols = []
        logger.info("Aucune feature HMM utilisée (num_hmm=0)")
        # Créer une colonne dummy si aucune feature HMM n'est utilisée mais que le modèle l'attend
        if "hmm_input" in build_reasoning_model.__code__.co_varnames:  # Vérifier si le modèle attend hmm_input
            if not hmm_cols:
                df_norm["hmm_dummy"] = 0.0
                hmm_cols = ["hmm_dummy"]
                logger.info("Création de hmm_dummy car num_hmm=0 mais le modèle attend hmm_input.")

    # Vérifier que les colonnes HMM sélectionnées existent
    missing_hmm_cols = [col for col in hmm_cols if col not in df_norm.columns and col != "hmm_dummy"]
    if missing_hmm_cols:
        logger.warning(
            f"Colonnes HMM manquantes dans les données: {missing_hmm_cols}. L'input HMM sera peut-être incorrect."
        )
        # Optionnel: Remplacer les colonnes manquantes par des dummies ou lever une erreur
        # Pour l'instant, on continue, mais cela pourrait causer une erreur plus tard
        # Si on veut ajouter des dummies pour les colonnes manquantes:
        for col in missing_hmm_cols:
            if col not in df_norm.columns:
                df_norm[col] = 0.0
        # S'assurer que hmm_cols ne contient que des colonnes existantes ou dummy
        hmm_cols = [col for col in hmm_cols if col in df_norm.columns]

    # Nouvelles colonnes de sentiment (si pru00e9sentes)
    sentiment_cols = [col for col in df.columns if col.startswith("sentiment_")]
    if not sentiment_cols:  # Si pas de colonnes de sentiment, cru00e9er un vecteur vide
        df_norm["sentiment_dummy"] = 0.0
        sentiment_cols = ["sentiment_dummy"]

    # Nouvelles colonnes CryptoBERT (si pru00e9sentes)
    cryptobert_cols = [col for col in df.columns if col.startswith("cryptobert_")]
    if not cryptobert_cols:  # Si pas de colonnes CryptoBERT, cru00e9er un vecteur vide
        df_norm["cryptobert_dummy"] = 0.0
        cryptobert_cols = ["cryptobert_dummy"]

    # Nouvelles colonnes d'informations de marchu00e9 (si pru00e9sentes)
    market_info_cols = [col for col in df.columns if col.startswith("market_info_")]
    if not market_info_cols:  # Si pas de colonnes d'informations de marchu00e9, cru00e9er un vecteur vide
        df_norm["market_info_dummy"] = 0.0
        market_info_cols = ["market_info_dummy"]

    # Convertir les symboles en entiers pour l'embedding
    symbol_mapping = {symbol: i for i, symbol in enumerate(df_norm["symbol"].unique())}
    df_norm["symbol_id"] = df_norm["symbol"].map(symbol_mapping)

    # Cru00e9er un dictionnaire de noms de features
    # S'assurer que les colonnes dummy ne sont pas incluses si elles n'étaient pas dans le df original
    original_df_cols = df.columns 
    feature_names = [
        col for col in (
            technical_cols + llm_cols + mcp_cols + hmm_cols + 
            sentiment_cols + cryptobert_cols + market_info_cols
        ) if col in original_df_cols and not col.endswith("_dummy")
    ]


    # Pru00e9parer les donnu00e9es d'entru00e9e
    X = {
        "technical_input": df_norm[technical_cols].values.astype(np.float32),
        "llm_input": df_norm[llm_cols].values.astype(np.float32),
        "mcp_input": df_norm[mcp_cols].values.astype(np.float32),
        "hmm_input": df_norm[hmm_cols].values.astype(np.float32),
        "instrument_input": df_norm[["symbol_id"]].values.astype(np.int64),
        "sentiment_input": df_norm[sentiment_cols].values.astype(np.float32),
        "cryptobert_input": df_norm[cryptobert_cols].values.astype(np.float32),
        "market_input": df_norm[market_info_cols].values.astype(np.float32),
    }

    return X, df_norm, feature_names


def _prepare_data_from_df(df: pd.DataFrame, cfg: Config):
    """
    Prépare les données X et df_norm à partir d'un DataFrame déjà chargé.
    Cette fonction est une adaptation de la logique de preprocess_data.
    Retourne X, df_norm, et la liste des noms de features techniques.
    """
    logger.info(f"Préparation des données depuis DataFrame: {len(df)} lignes.")
    # La logique ici devrait refléter celle de preprocess_data concernant
    # l'identification des colonnes, la normalisation, et la création de X et feature_names.
    # Pour la normalisation, idéalement, on utiliserait un scaler pré-entraîné.
    # Pour cet exemple, on réplique la normalisation simple de preprocess_data.

    price_cols = ["open", "high", "low", "close", "volume"]
    technical_cols = [
        col for col in df.columns if col not in [
            "timestamp", "symbol", "market_regime", "level_sl", "level_tp",
            "hmm_regime", "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "split",
        ] and not any(col.startswith(p) for p in ["llm_", "mcp_", "sentiment_", "cryptobert_", "market_info_"])
    ]
    exclude_cols = ["timestamp", "symbol", "split"]
    technical_cols = [
        col for col in technical_cols if col not in exclude_cols and \
        df[col].dtype != "object" and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    df_norm = df.copy()
    for col in price_cols + technical_cols:
        if col in df_norm.columns and df_norm[col].dtype != "object" and not pd.api.types.is_datetime64_any_dtype(df_norm[col]):
            q1 = df_norm[col].quantile(0.25)
            q3 = df_norm[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df_norm[col] = (df_norm[col] - q1) / iqr
            else:
                mean_val = df_norm[col].mean() # Renommé pour éviter conflit avec .mean()
                std_val = df_norm[col].std()   # Renommé pour éviter conflit avec .std()
                if std_val > 0:
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
                else:
                    df_norm[col] = 0
    
    llm_cols = [col for col in df.columns if col.startswith("llm_")] or ["llm_dummy"]
    if "llm_dummy" in llm_cols and "llm_dummy" not in df_norm.columns: df_norm["llm_dummy"] = 0.0
    
    mcp_cols = [col for col in df.columns if col.startswith("mcp_")] or ["mcp_dummy"]
    if "mcp_dummy" in mcp_cols and "mcp_dummy" not in df_norm.columns: df_norm["mcp_dummy"] = 0.0

    num_hmm = cfg.get_config("model.num_hmm", 0)
    if num_hmm == 1: hmm_cols = ["hmm_regime"]
    elif num_hmm > 1: hmm_cols = [f"hmm_prob_{i}" for i in range(num_hmm)]
    else: hmm_cols = []
    if not hmm_cols and "hmm_input" in build_reasoning_model.__code__.co_varnames:
        hmm_cols = ["hmm_dummy"]
    if "hmm_dummy" in hmm_cols and "hmm_dummy" not in df_norm.columns: df_norm["hmm_dummy"] = 0.0
    # S'assurer que toutes les hmm_cols existent dans df_norm, sinon les ajouter avec 0
    for col in hmm_cols:
        if col not in df_norm.columns: df_norm[col] = 0.0


    sentiment_cols = [col for col in df.columns if col.startswith("sentiment_")] or ["sentiment_dummy"]
    if "sentiment_dummy" in sentiment_cols and "sentiment_dummy" not in df_norm.columns: df_norm["sentiment_dummy"] = 0.0
    
    cryptobert_cols = [col for col in df.columns if col.startswith("cryptobert_")] or ["cryptobert_dummy"]
    if "cryptobert_dummy" in cryptobert_cols and "cryptobert_dummy" not in df_norm.columns: df_norm["cryptobert_dummy"] = 0.0

    market_info_cols = [col for col in df.columns if col.startswith("market_info_")] or ["market_info_dummy"]
    if "market_info_dummy" in market_info_cols and "market_info_dummy" not in df_norm.columns: df_norm["market_info_dummy"] = 0.0

    if "symbol" in df_norm.columns:
        unique_symbols = df_norm["symbol"].unique()
        symbol_mapping = {symbol: i for i, symbol in enumerate(unique_symbols)}
        df_norm["symbol_id"] = df_norm["symbol"].map(symbol_mapping)
    else: # Fallback si 'symbol' n'est pas là (ne devrait pas arriver avec des données valides)
        df_norm["symbol_id"] = 0


    actual_feature_names = [
        col for col in (
            technical_cols + llm_cols + mcp_cols + hmm_cols + 
            sentiment_cols + cryptobert_cols + market_info_cols
        ) if col in df.columns and not col.endswith("_dummy") # Utiliser df.columns ici
    ]
    
    X_output = {
        "technical_input": df_norm[technical_cols].values.astype(np.float32),
        "llm_input": df_norm[llm_cols].values.astype(np.float32),
        "mcp_input": df_norm[mcp_cols].values.astype(np.float32),
        "hmm_input": df_norm[hmm_cols].values.astype(np.float32),
        "instrument_input": df_norm[["symbol_id"]].values.astype(np.int64),
        "sentiment_input": df_norm[sentiment_cols].values.astype(np.float32),
        "cryptobert_input": df_norm[cryptobert_cols].values.astype(np.float32),
        "market_input": df_norm[market_info_cols].values.astype(np.float32),
    }
    return X_output, df_norm, actual_feature_names


def create_model(X, feature_names, cfg: Config): # Ajout de cfg
    """
    Cru00e9e ou charge un modu00e8le de raisonnement pour les pru00e9dictions.

    Args:
        X: Donnu00e9es d'entru00e9e
        feature_names: Noms des features

    Returns:
        Modu00e8le de raisonnement
    """
    # Du00e9terminer les dimensions d'entru00e9e
    tech_input_shape = X["technical_input"].shape[1:]
    llm_input_shape = X["llm_input"].shape[1:]
    mcp_input_shape = X["mcp_input"].shape[1:]
    hmm_input_shape = X["hmm_input"].shape[1:]
    sentiment_input_shape = X["sentiment_input"].shape[1:]
    cryptobert_input_shape = X["cryptobert_input"].shape[1:]
    market_input_shape = X["market_input"].shape[1:]

    # Charger le modu00e8le entrau00eenu00e9 ou le reconstruire
    # Utiliser le chemin du modèle depuis la configuration si disponible
    model_base_path = cfg.get_config("paths.output_dir", "outputs") # outputs/enhanced_reasoning par exemple
    model_name = cfg.get_config("model.name", "enhanced_reasoning_model") # Nom du modèle
    
    # Construire le chemin complet vers le modèle .keras et les poids .h5
    # Exemple: outputs/enhanced_reasoning/enhanced_reasoning_model.keras
    #          outputs/enhanced_reasoning/enhanced_reasoning_model.weights.h5
    # Le script d'entraînement enhanced_reasoning_training.py sauvegarde sous "best_model.h5" ou "final_model.h5"
    # dans un sous-répertoire. Adapter ici pour correspondre.
    # Pour l'instant, on garde la logique existante mais on note que le chemin pourrait être plus dynamique.
    
    # Tentative de chemin basé sur la structure de enhanced_reasoning_training.py
    # Supposons que le modèle est dans "outputs/enhanced/best_model.h5" ou "final_model.h5"
    # ou un chemin plus générique comme "models/current_production_model.keras"
    
    # Pour cet exemple, nous allons utiliser un chemin fixe comme avant, mais il devrait être configurable.
    # Le script d'entraînement sauvegarde dans "outputs/enhanced/best_model.h5"
    # Le nom du répertoire "enhanced_reasoning_model" dans le code original de create_model
    # semble être une convention locale, pas celle du script d'entraînement.
    
    # Utilisons le chemin par défaut du script d'entraînement `enhanced_reasoning_training.py`
    # qui est `outputs/enhanced/best_model.h5` (ou `final_model.h5`)
    # Le nom du modèle est "best_model" ou "final_model", pas "model".
    # Le répertoire est "outputs/enhanced".
    
    # Chemin du modèle sauvegardé par enhanced_reasoning_training.py
    # TODO: Rendre ce chemin configurable via config.yaml ou argument CLI
    saved_model_dir = cfg.get_config("paths.trained_model_dir", "outputs/enhanced") # Exemple
    model_filename = cfg.get_config("model.filename_keras", "best_model.keras") # ou .h5
    
    model_path_keras = os.path.join(saved_model_dir, model_filename) # ex: outputs/enhanced/best_model.keras
    # Les poids sont souvent sauvegardés avec le modèle .keras ou séparément.
    # Si ModelCheckpoint(save_weights_only=False) est utilisé, le .keras contient tout.

    if os.path.exists(model_path_keras):
        logger.info(f"Chargement du modu00e8le depuis {model_path_keras}")
        try:
            custom_objects = {"ReasoningModule": ReasoningModule} # Nécessaire si ReasoningModule est une couche custom
            model = load_model(model_path_keras, custom_objects=custom_objects, compile=False) # compile=False est souvent plus sûr
            
            # Recompiler le modèle avec les paramètres attendus par ce script si besoin
            # compile_reasoning_model(model, learning_rate=cfg.get_config("training.learning_rate", 0.001))
            # Pour l'inférence pure, la compilation n'est pas toujours nécessaire si on n'entraîne plus.
            # Mais si on veut utiliser model.evaluate ou des métriques compilées, il faut compiler.
            # Pour l'instant, on suppose que le modèle chargé est prêt pour predict().
            logger.info("Modu00e8le chargu00e9 avec succu00e8s.")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modu00e8le depuis {model_path_keras}: {e}", exc_info=True)
            logger.info("Tentative de reconstruction du modu00e8le (sans poids pré-entraînés).")
    else:
        logger.warning(f"Fichier modu00e8le non trouvu00e9 u00e0 {model_path_keras}. Reconstruction du modu00e8le sans poids pré-entraînés.")

    # Si le chargement échoue ou si le fichier n'existe pas, reconstruire (architecture seulement)
    logger.info("Construction d'une nouvelle instance du modu00e8le de raisonnement (non entraîné).")
    
    model_params_from_config = cfg.get_config("model.reasoning_architecture", {})
    # S'assurer que les dimensions d'input correspondent à X
    model_params_from_config["tech_input_shape"] = tech_input_shape
    model_params_from_config["mcp_input_dim"] = mcp_input_shape[0] if mcp_input_shape else 0
    model_params_from_config["hmm_input_dim"] = hmm_input_shape[0] if hmm_input_shape else 0
    model_params_from_config["sentiment_input_dim"] = sentiment_input_shape[0] if sentiment_input_shape else 0
    model_params_from_config["cryptobert_input_dim"] = cryptobert_input_shape[0] if cryptobert_input_shape else 0
    model_params_from_config["market_input_dim"] = market_input_shape[0] if market_input_shape else 0
    # instrument_vocab_size et instrument_embedding_dim devraient aussi venir de la config ou être déduits
    model_params_from_config["instrument_vocab_size"] = cfg.get_config("model.instrument_vocab_size", 10)
    model_params_from_config["instrument_embedding_dim"] = cfg.get_config("model.instrument_embedding_dim", 8)
    
    # num_market_regime_classes et num_sl_tp_outputs
    market_regime_mapping = cfg.get_config("data.label_mappings.market_regime", {})
    num_market_regime_classes = max(market_regime_mapping.values()) + 1 if market_regime_mapping else 2
    model_params_from_config["num_market_regime_classes"] = num_market_regime_classes
    model_params_from_config["num_sl_tp_outputs"] = cfg.get_config("model.num_sl_tp_outputs", 2)
    
    model_params_from_config["feature_names"] = feature_names # Passer les noms de features

    model = build_reasoning_model(**model_params_from_config)
    
    # Compiler le modèle reconstruit (nécessaire avant predict)
    active_outputs_cfg = cfg.get_config("model.active_outputs", ["market_regime", "sl_tp", "reasoning"])
    
    lr_from_config = cfg.get_config("training.learning_rate", 0.001)
    try:
        learning_rate_float = float(lr_from_config)
    except (ValueError, TypeError):
        logger.warning(f"Impossible de convertir learning_rate '{lr_from_config}' en float. Utilisation de 0.001 par défaut.")
        learning_rate_float = 0.001
        
    compile_reasoning_model(model, 
                            learning_rate=learning_rate_float,
                            active_outputs=active_outputs_cfg)
    logger.info("Modu00e8le reconstruit et compilu00e9 (sans poids entrau00eenu00e9s).")
    return model

def generate_predictions(model, X, df_original_norm, cfg: Config): # Ajout de cfg
    """
    Gu00e9nu00e8re des pru00e9dictions avec le modu00e8le et stocke toutes les sorties.
    """
    logger.info("Gu00e9nu00e9ration des pru00e9dictions complu00e8tes du modu00e8le...")
    full_predictions_dict = model.predict(X)

    # Nombre d'u00e9chantillons
    num_samples = df_original_norm.shape[0]
    
    results_list = []
    
    # Obtenir le nombre d'u00e9tapes de raisonnement depuis la config pour itérer
    reasoning_arch_cfg = cfg.get_config("model.reasoning_architecture", {})
    num_reasoning_steps = reasoning_arch_cfg.get("num_reasoning_steps", 0) # Du00e9faut u00e0 0 si non du00e9fini
    use_cot_flag = reasoning_arch_cfg.get("use_chain_of_thought", False)


    for i in range(num_samples):
        record = {
            "timestamp": df_original_norm["timestamp"].iloc[i],
            "symbol": df_original_norm["symbol"].iloc[i],
            "close": df_original_norm["close"].iloc[i], # Utiliser le close original (non normalisé) si disponible, ou normalisé
                                                     # df_original_norm est normalisé, il faudrait le df original pour le close non-norm.
                                                     # Pour l'instant, on utilise le close de df_original_norm.
        }
        
        # Ajouter toutes les sorties du dictionnaire de prédictions
        for key, value_array in full_predictions_dict.items():
            record[key] = value_array[i].tolist() # Convertir le vecteur numpy en liste pour le DataFrame
            
        results_list.append(record)
        
    predictions_df = pd.DataFrame(results_list)
    
    # Convertir les probas de ru00e9gime en classe et confiance
    if "market_regime" in predictions_df.columns:
        market_regime_probs_stacked = np.stack(predictions_df["market_regime"].values)
        predictions_df["market_regime_pred"] = np.argmax(market_regime_probs_stacked, axis=1)
        predictions_df["market_regime_confidence"] = np.max(market_regime_probs_stacked, axis=1)

    # Extraire SL et TP
    if "sl_tp" in predictions_df.columns:
        sl_tp_stacked = np.stack(predictions_df["sl_tp"].values)
        predictions_df["sl_pred"] = sl_tp_stacked[:, 0]
        predictions_df["tp_pred"] = sl_tp_stacked[:, 1]
        predictions_df["risk_reward_ratio"] = np.abs(predictions_df["tp_pred"] / predictions_df["sl_pred"].replace(0, np.nan))
        predictions_df["risk_reward_ratio"].fillna(0, inplace=True)


    return predictions_df, full_predictions_dict # Retourner aussi le dict brut pour ExplanationDecoder


def generate_trading_insights(
    main_predictions_df: pd.DataFrame, # Renommé depuis predictions_df_enriched
    all_model_outputs: dict, # Renommé depuis full_predictions_dict
    df_original_norm: pd.DataFrame, 
    feature_names: list,
    cfg: Config
) -> pd.DataFrame:
    """
    Gu00e9nu00e8re des insights de trading et des explications CoT.
    """
    logger.info("Gu00e9nu00e9ration des insights de trading et explications CoT...")

    decoder = ExplanationDecoder(feature_names=feature_names)
    
    reasoning_arch_cfg = cfg.get_config("model.reasoning_architecture", {})
    num_reasoning_steps = reasoning_arch_cfg.get("num_reasoning_steps", 0)
    use_cot_flag = reasoning_arch_cfg.get("use_chain_of_thought", False)

    cot_explanations = []

    for i in range(len(main_predictions_df)):
        market_data_i = df_original_norm.iloc[i][['open', 'high', 'low', 'close', 'volume']].to_dict()

        single_instance_main_preds = {}
        if "market_regime" in all_model_outputs: # Utiliser all_model_outputs
            single_instance_main_preds["market_regime"] = all_model_outputs["market_regime"][i:i+1]
        if "sl_tp" in all_model_outputs: # Utiliser all_model_outputs
            single_instance_main_preds["sl_tp"] = all_model_outputs["sl_tp"][i:i+1]
        # S'assurer d'inclure 'signal' et 'volatility_quantiles' si ExplanationDecoder les attend
        if "signal" in all_model_outputs: # Nom de sortie du modèle pour le signal de trading
             single_instance_main_preds["signal"] = all_model_outputs["signal"][i:i+1]
        if "volatility_quantiles" in all_model_outputs: # Nom de sortie pour la volatilité
             single_instance_main_preds["volatility_quantiles"] = all_model_outputs["volatility_quantiles"][i:i+1]


        # Extraire tous les vecteurs de raisonnement pour l'instance i
        final_reasoning_vec_i = all_model_outputs.get("final_reasoning", np.array([]))[i:i+1]
        if final_reasoning_vec_i.size == 0: final_reasoning_vec_i = None

        market_regime_expl_vec_all_classes_i = all_model_outputs.get("market_regime_explanation", np.array([]))[i:i+1]
        # Sélectionner le vecteur d'explication pour le régime prédit
        market_regime_expl_vec_i = None
        if market_regime_expl_vec_all_classes_i.size > 0 and "market_regime_pred" in main_predictions_df.columns:
            pred_idx = main_predictions_df["market_regime_pred"].iloc[i]
            # market_regime_explanation a la forme (batch, num_regimes, reasoning_units)
            # donc pour l'instance i et le régime pred_idx:
            if market_regime_expl_vec_all_classes_i.ndim == 3: # (1, num_regimes, units)
                 market_regime_expl_vec_i = market_regime_expl_vec_all_classes_i[0, pred_idx, :]
            elif market_regime_expl_vec_all_classes_i.ndim == 2 and market_regime_expl_vec_all_classes_i.shape[0] == cfg.get_config("model.reasoning_architecture.num_market_regime_classes",2): # (num_regimes, units) pour une instance
                 market_regime_expl_vec_i = market_regime_expl_vec_all_classes_i[pred_idx, :]


        sl_expl_vec_i = all_model_outputs.get("sl_explanation", np.array([]))[i:i+1]
        if sl_expl_vec_i.size == 0: sl_expl_vec_i = None
        
        tp_expl_vec_i = all_model_outputs.get("tp_explanation", np.array([]))[i:i+1]
        if tp_expl_vec_i.size == 0: tp_expl_vec_i = None

        reasoning_steps_vecs_i = []
        if use_cot_flag:
            for j in range(num_reasoning_steps):
                step_key = f"reasoning_step_{j}"
                if step_key in all_model_outputs:
                    step_output = all_model_outputs[step_key][i:i+1]
                    if step_output.size > 0:
                        reasoning_steps_vecs_i.append(step_output)
        if not reasoning_steps_vecs_i: reasoning_steps_vecs_i = None
        
        attention_scores_vec_i = all_model_outputs.get("attention_scores", np.array([]))[i:i+1]
        if attention_scores_vec_i.size == 0: attention_scores_vec_i = None
        
        try:
            cot_text = decoder.generate_chain_of_thought_explanation(
                market_data=market_data_i,
                predictions=single_instance_main_preds,
                final_reasoning_vec=final_reasoning_vec_i,
                market_regime_expl_vec=market_regime_expl_vec_i, # Vecteur pour le régime prédit
                sl_expl_vec=sl_expl_vec_i,
                tp_expl_vec=tp_expl_vec_i,
                reasoning_steps_vecs=reasoning_steps_vecs_i,
                attention_scores_vec=attention_scores_vec_i
            )
        except Exception as e:
            logger.error(f"Erreur lors de la gu00e9nu00e9ration de l'explication CoT pour l'index {i}: {e}", exc_info=True)
            row = main_predictions_df.iloc[i]
            cot_text = (f"Marchu00e9 {'haussier' if row['market_regime_pred'] == 1 else 'stable/baissier'} "
                        f"(confiance: {row['market_regime_confidence']:.2f}). "
                        f"Ratio risque/ru00e9compense: {row['risk_reward_ratio']:.2f}. "
                        f"Stop loss u00e0 {row['sl_pred']:.4f}, Take profit u00e0 {row['tp_pred']:.4f}.")
        
        cot_explanations.append(cot_text)

    main_predictions_df["reasoning"] = cot_explanations # Assigner au DataFrame d'entrée/sortie
    
    main_predictions_df["trading_decision"] = main_predictions_df.apply(
        lambda row: (
            "BUY"
            if row["market_regime_pred"] == 1 and row["risk_reward_ratio"] >= 2.0
            else "SELL" if row["market_regime_pred"] == 0 and row["risk_reward_ratio"] >= 2.0 else "HOLD"
        ),
        axis=1,
    )

    main_predictions_df["trading_confidence"] = main_predictions_df.apply(
        lambda row: row["market_regime_confidence"] * min(row["risk_reward_ratio"] / 3.0, 1.0), axis=1
    )

    if not main_predictions_df.empty:
        last_prediction = main_predictions_df.iloc[-1]
        if last_prediction["trading_decision"] != "HOLD":
            logger.info(
                f"Envoi de la notification Telegram pour le dernier signal: {last_prediction['trading_decision']}"
            )
            notify_trade_sync(
                signal=f"{last_prediction['symbol']} - {last_prediction['trading_decision']}",
                price=last_prediction["close"],
                reasoning=last_prediction["reasoning"],
            )

    return main_predictions_df


def explain_signal(signal_data_df: pd.DataFrame, cfg: Config, feature_names_list: list) -> str: # Renommé feature_names
    """
    Génère une explication CoT pour une seule instance de données (ou un petit batch).
    """
    logger.info(f"Début de l'explication pour {len(signal_data_df)} instance(s).")
    
    # Utiliser la nouvelle fonction _prepare_data_from_df pour traiter le DataFrame d'entrée
    X_signal, df_norm_signal, actual_feature_names = _prepare_data_from_df(signal_data_df.copy(), cfg)

    # Charger/Créer le modèle
    model = create_model(X_signal, actual_feature_names, cfg) # actual_feature_names est la liste des noms

    logger.info("Obtention des prédictions complètes pour l'explication...")
    all_model_outputs_signal = model.predict(X_signal)
    
    # Initialiser ExplanationDecoder avec les noms de features dérivés de signal_data_df
    decoder = ExplanationDecoder(feature_names=actual_feature_names)

    # Expliquer la première instance (i=0) car explain_signal est pour un signal à la fois
    i = 0 
    market_data_i = df_norm_signal.iloc[i][['open', 'high', 'low', 'close', 'volume']].to_dict()
    
    single_instance_main_preds_i = {}
    # Extraire les prédictions principales pour l'instance i
    if "market_regime" in all_model_outputs_signal:
        single_instance_main_preds_i["market_regime"] = all_model_outputs_signal["market_regime"][i:i+1]
    if "sl_tp" in all_model_outputs_signal:
        single_instance_main_preds_i["sl_tp"] = all_model_outputs_signal["sl_tp"][i:i+1]
    if "signal" in all_model_outputs_signal:
         single_instance_main_preds_i["signal"] = all_model_outputs_signal["signal"][i:i+1]
    if "volatility_quantiles" in all_model_outputs_signal:
         single_instance_main_preds_i["volatility_quantiles"] = all_model_outputs_signal["volatility_quantiles"][i:i+1]

    # Extraire les vecteurs de raisonnement pour l'instance i
    final_reasoning_vec_i = all_model_outputs_signal.get("final_reasoning", np.array([]))[i:i+1]
    if final_reasoning_vec_i.size == 0: final_reasoning_vec_i = None

    market_regime_expl_vec_all_i = all_model_outputs_signal.get("market_regime_explanation", np.array([]))[i:i+1]
    market_regime_expl_vec_i = None
    if market_regime_expl_vec_all_i.size > 0 and "market_regime" in single_instance_main_preds_i:
        pred_idx_expl = np.argmax(single_instance_main_preds_i["market_regime"].flatten())
        if market_regime_expl_vec_all_i.ndim == 3: # (1, num_regimes, units)
            market_regime_expl_vec_i = market_regime_expl_vec_all_i[0, pred_idx_expl, :]
        elif market_regime_expl_vec_all_i.ndim == 2 and market_regime_expl_vec_all_i.shape[0] == cfg.get_config("model.reasoning_architecture.num_market_regime_classes",2):
            market_regime_expl_vec_i = market_regime_expl_vec_all_i[pred_idx_expl, :]


    sl_expl_vec_i = all_model_outputs_signal.get("sl_explanation", np.array([]))[i:i+1]
    if sl_expl_vec_i.size == 0: sl_expl_vec_i = None
    
    tp_expl_vec_i = all_model_outputs_signal.get("tp_explanation", np.array([]))[i:i+1]
    if tp_expl_vec_i.size == 0: tp_expl_vec_i = None

    reasoning_arch_cfg = cfg.get_config("model.reasoning_architecture", {})
    num_reasoning_steps = reasoning_arch_cfg.get("num_reasoning_steps", 0)
    use_cot_flag = reasoning_arch_cfg.get("use_chain_of_thought", False)
    
    reasoning_steps_vecs_i = []
    if use_cot_flag:
        for j in range(num_reasoning_steps):
            step_key = f"reasoning_step_{j}"
            if step_key in all_model_outputs_signal:
                step_output = all_model_outputs_signal[step_key][i:i+1]
                if step_output.size > 0:
                    reasoning_steps_vecs_i.append(step_output)
    if not reasoning_steps_vecs_i: reasoning_steps_vecs_i = None
            
    attention_scores_vec_i = all_model_outputs_signal.get("attention_scores", np.array([]))[i:i+1]
    if attention_scores_vec_i.size == 0: attention_scores_vec_i = None
    
    logger.info("Génération du texte d'explication CoT pour explain_signal...")
    try:
        cot_text = decoder.generate_chain_of_thought_explanation(
            market_data=market_data_i,
            predictions=single_instance_main_preds_i,
            final_reasoning_vec=final_reasoning_vec_i,
            market_regime_expl_vec=market_regime_expl_vec_i,
            sl_expl_vec=sl_expl_vec_i,
            tp_expl_vec=tp_expl_vec_i,
            reasoning_steps_vecs=reasoning_steps_vecs_i,
            attention_scores_vec=attention_scores_vec_i
        )
        logger.info("Texte d'explication CoT généré pour explain_signal.")
        return cot_text
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'explication CoT dans explain_signal: {e}", exc_info=True)
        return f"Erreur lors de la génération de l'explication : {e}"


def main():
    """
    Fonction principale.
    """
    cfg = Config() # Charger la configuration
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X, df_norm, feature_names = preprocess_data(args.data_path)

    # Sauvegarder feature_names pour le test du bot Telegram
    # Ceci est temporaire pour obtenir le fichier. Normalement, il est créé par le script d'entraînement.
    temp_feature_names_path = "outputs/enhanced/feature_names.json"
    os.makedirs(os.path.dirname(temp_feature_names_path), exist_ok=True)
    try:
        with open(temp_feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Liste de features temporaire sauvegardée dans {temp_feature_names_path} pour le test du bot.")
    except Exception as e:
        logger.error(f"Erreur sauvegarde feature_names temporaire: {e}")

    model = create_model(X, feature_names, cfg) # Passe cfg

    # generate_predictions a été modifié pour retourner aussi full_predictions_dict
    predictions_df, full_predictions_dict = generate_predictions(model, X, df_norm, cfg) # Passe cfg

    # generate_trading_insights a été modifié pour prendre plus d'arguments
    insights_df = generate_trading_insights(predictions_df, full_predictions_dict, df_norm, feature_names, cfg) # Passe cfg

    predictions_path = os.path.join(args.output_dir, "trading_predictions_with_reasoning.csv") # Nom de fichier mis à jour
    insights_df.to_csv(predictions_path, index=False)
    logger.info(f"Pru00e9dictions avec raisonnement CoT sauvegardu00e9es dans {predictions_path}")

    logger.info("\nExemples de pru00e9dictions avec raisonnement CoT:")
    for i, (_, row) in enumerate(insights_df.head(3).iterrows()): # Afficher 3 exemples
        logger.info(f"\n--- Exemple {i+1} ---")
        logger.info(f"Timestamp: {row['timestamp']}, Symbol: {row['symbol']}, Close: {row['close']}")
        logger.info(f"Du00e9cision: {row['trading_decision']} (Confiance: {row.get('trading_confidence', 0.0):.2f})")
        logger.info(f"Raisonnement CoT: {row['reasoning']}")
        logger.info(f"SL: {row['sl_pred']:.4f}, TP: {row['tp_pred']:.4f}, R/R: {row['risk_reward_ratio']:.2f}")

    # Exemple d'utilisation de explain_signal (pourrait être dans un test ou une autre fonction)
    if not df_norm.empty:
        logger.info("\n--- Test de la fonction explain_signal pour la première instance ---")
        try:
            # Prendre la première ligne de df_norm comme exemple de signal_data
            # Important: df_norm est normalisé. Pour une explication fidèle, il faudrait
            # idéalement les données brutes avant normalisation pour market_data dans ExplanationDecoder.
            # Et le scaler utilisé pendant l'entraînement pour normaliser les inputs du modèle.
            # Pour l'instant, explain_signal réutilise preprocess_data qui re-normalise.
            single_instance_df = pd.read_parquet(args.data_path).iloc[0:1] # Charger la donnée brute originale
            
            explanation_text = explain_signal(single_instance_df, cfg, feature_names)
            logger.info(f"Explication pour la première instance:\n{explanation_text}")
        except Exception as e:
            logger.error(f"Erreur lors du test de explain_signal: {e}", exc_info=True)


if __name__ == "__main__":
    main()
