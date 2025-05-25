#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour le modèle monolithique avec dataset minutely
----------------------------------------------------------------

Ce script charge le dataset minutely, prépare les données pour l'entraînement
et entraîne un modèle monolithique sur une petite partie des données.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_monolith_minutely")

# Répertoire du projet
PROJECT_ROOT = Path("/home/morningstar/Desktop/crypto_robot/Morningstar")
sys.path.append(str(PROJECT_ROOT))

# Chemin pour le modèle monolithique
sys.path.append("/tmp/morningstar_monolith")

# Importer le modèle monolithique
from monolith_model import MonolithModel

def load_dataset(file_path):
    """Charge le dataset depuis un fichier parquet."""
    logger.info(f"Chargement du dataset depuis {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset chargé avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
        # Afficher quelques colonnes pour débogage
        logger.info(f"Aperçu des colonnes: {list(df.columns)[:10]}...")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        raise

def prepare_data(df, test_size=0.2, validation_size=0.1):
    """Prépare les données pour l'entraînement du modèle monolithique."""
    logger.info("Préparation des données pour l'entraînement")
    
    # Extraire les différentes catégories de features
    tech_cols = [col for col in df.columns if not col.startswith(("llm_", "mcp_", "hmm_")) and col != "symbol"]
    hmm_cols = [col for col in df.columns if col.startswith("hmm_")]
    llm_cols = [col for col in df.columns if col.startswith("llm_")]
    mcp_cols = [col for col in df.columns if col.startswith("mcp_")]
    
    logger.info(f"Features techniques: {len(tech_cols)}")
    logger.info(f"Features HMM: {len(hmm_cols)}")
    logger.info(f"Features LLM: {len(llm_cols)}")
    logger.info(f"Features MCP: {len(mcp_cols)}")
    
    # Convertir les symboles en nombres entiers (au lieu de one-hot encoding)
    # Créer un dictionnaire de mapping symbol -> id
    unique_symbols = df["symbol"].unique()
    symbol_to_id = {symbol: i for i, symbol in enumerate(unique_symbols)}
    num_symbols = len(unique_symbols)
    logger.info(f"Nombre de symboles uniques: {num_symbols} - {unique_symbols}")
    
    # Convertir les symboles en IDs
    instrument_ids = np.array([symbol_to_id[symbol] for symbol in df["symbol"]]).reshape(-1, 1)
    
    # Préparer les features techniques
    tech_features = df[tech_cols].values
    tech_scaler = StandardScaler()
    tech_features_scaled = tech_scaler.fit_transform(tech_features)
    
    # Préparer les features MCP
    mcp_features = df[mcp_cols].values
    mcp_scaler = StandardScaler()
    mcp_features_scaled = mcp_scaler.fit_transform(mcp_features)
    
    # Extraire les embeddings LLM
    llm_embeddings = np.stack(df["llm_embedding"].values)
    
    # Créer des labels synthétiques pour le test
    # 1. Signal: 0 (Vendre), 1 (Neutre), 2 (Acheter)
    # Utiliser les mouvements de prix pour créer des labels de signal
    df["next_return"] = df["close"].pct_change(periods=10).shift(-10)
    signal_thresholds = df["next_return"].quantile([0.33, 0.66])
    
    signals = np.zeros(len(df))
    signals[(df["next_return"] > signal_thresholds.iloc[0]) & (df["next_return"] <= signal_thresholds.iloc[1])] = 1
    signals[df["next_return"] > signal_thresholds.iloc[1]] = 2
    
    # 2. Niveaux SL: utilisez ATR comme proxy
    sl_levels = df["ATR"] * 2  # 2 * ATR en dessous du prix
    
    # 3. Niveaux TP: utilisez ATR comme proxy
    tp_levels = df["ATR"] * 3  # 3 * ATR au-dessus du prix
    
    # Supprimer les lignes avec des valeurs NaN
    valid_idx = ~(df["next_return"].isna() | np.isnan(sl_levels) | np.isnan(tp_levels))
    
    tech_features_scaled = tech_features_scaled[valid_idx]
    mcp_features_scaled = mcp_features_scaled[valid_idx]
    llm_embeddings = llm_embeddings[valid_idx]
    instrument_ids = instrument_ids[valid_idx]
    signals = signals[valid_idx]
    sl_levels = sl_levels[valid_idx]
    tp_levels = tp_levels[valid_idx]
    
    # Encoder les signaux en one-hot
    signal_one_hot = np.zeros((len(signals), 3))
    for i, s in enumerate(signals):
        signal_one_hot[i, int(s)] = 1
    
    # Division en ensembles d'entraînement, de validation et de test
    # Première division: train+val vs test
    X_train_val_tech, X_test_tech, X_train_val_mcp, X_test_mcp, X_train_val_llm, X_test_llm, \
    X_train_val_symbol, X_test_symbol, y_train_val_signal, y_test_signal, \
    y_train_val_sl, y_test_sl, y_train_val_tp, y_test_tp = train_test_split(
        tech_features_scaled, mcp_features_scaled, llm_embeddings, instrument_ids,
        signal_one_hot, sl_levels, tp_levels, test_size=test_size, random_state=42
    )
    
    # Deuxième division: train vs val
    val_ratio = validation_size / (1 - test_size)
    X_train_tech, X_val_tech, X_train_mcp, X_val_mcp, X_train_llm, X_val_llm, \
    X_train_symbol, X_val_symbol, y_train_signal, y_val_signal, \
    y_train_sl, y_val_sl, y_train_tp, y_val_tp = train_test_split(
        X_train_val_tech, X_train_val_mcp, X_train_val_llm, X_train_val_symbol,
        y_train_val_signal, y_train_val_sl, y_train_val_tp, test_size=val_ratio, random_state=42
    )
    
    # Vérifier le type de données pour instrument_input
    logger.info(f"Type de X_train_symbol: {X_train_symbol.dtype}, shape: {X_train_symbol.shape}")
    
    # S'assurer que instrument_input est bien du type entier
    X_train_symbol = X_train_symbol.astype(np.int32)
    X_val_symbol = X_val_symbol.astype(np.int32)
    X_test_symbol = X_test_symbol.astype(np.int32)
    
    # Créer des entrées de CoT factices (chaînes vides) pour la compatibilité
    X_train_cot = np.zeros((X_train_tech.shape[0], 1))
    X_val_cot = np.zeros((X_val_tech.shape[0], 1))
    X_test_cot = np.zeros((X_test_tech.shape[0], 1))
    
    # Préparer les données d'entraînement
    train_data = {
        "technical_input": X_train_tech,
        "embeddings_input": X_train_llm,
        "instrument_input": X_train_symbol,
        "mcp_input": X_train_mcp,
        "cot_input": X_train_cot,
    }
    
    # Préparer les données de validation
    val_data = {
        "technical_input": X_val_tech,
        "embeddings_input": X_val_llm,
        "instrument_input": X_val_symbol,
        "mcp_input": X_val_mcp,
        "cot_input": X_val_cot,
    }
    
    # Préparer les données de test
    test_data = {
        "technical_input": X_test_tech,
        "embeddings_input": X_test_llm,
        "instrument_input": X_test_symbol,
        "mcp_input": X_test_mcp,
        "cot_input": X_test_cot,
    }
    
    # Préparer les cibles d'entraînement
    train_targets = {
        "signal_output": y_train_signal,
        "sl_tp_output": np.column_stack((y_train_sl, y_train_tp)),
    }
    
    # Préparer les cibles de validation
    val_targets = {
        "signal_output": y_val_signal,
        "sl_tp_output": np.column_stack((y_val_sl, y_val_tp)),
    }
    
    # Préparer les cibles de test
    test_targets = {
        "signal_output": y_test_signal,
        "sl_tp_output": np.column_stack((y_test_sl, y_test_tp)),
    }
    
    logger.info(f"Données d'entraînement préparées: {X_train_tech.shape[0]} échantillons")
    logger.info(f"Données de validation préparées: {X_val_tech.shape[0]} échantillons")
    logger.info(f"Données de test préparées: {X_test_tech.shape[0]} échantillons")
    
    return train_data, train_targets, val_data, val_targets, test_data, test_targets, tech_cols, tech_scaler, num_symbols

def train_model(train_data, train_targets, val_data, val_targets, num_symbols, epochs=10, batch_size=64):
    """Entraîne le modèle monolithique."""
    logger.info("Initialisation du modèle monolithique")
    
    # Déterminer les dimensions de chaque type d'entrée
    tech_dim = train_data["technical_input"].shape[1]
    mcp_dim = train_data["mcp_input"].shape[1]
    instr_dim = train_data["instrument_input"].shape[1]
    emb_dim = train_data["embeddings_input"].shape[1]
    
    # Configurer le modèle monolithique
    model_config = {
        "tech_input_shape": (tech_dim,),
        "embeddings_input_shape": emb_dim,
        "mcp_input_shape": mcp_dim,
        "instrument_vocab_size": num_symbols,
        "instrument_embedding_dim": 8,
        "cot_input_shape": 1,  # Dimension fictive pour Chain-of-Thought
        "backbone_config": {
            "dense_units": 128,
            "lstm_units": 64,
            "transformer_blocks": 2,
            "transformer_heads": 4,
            "transformer_dim": 64,
            "ff_dim": 128
        },
        "head_config": {
            "signal": {"units": [32], "classes": 3},
            "sl_tp": {"units": [32], "outputs": 2}
        },
        "l2_reg": 0.001,
        "dropout_rate": 0.3,
        "use_batch_norm": True
    }
    
    logger.info(f"Configuration du modèle: instrument_vocab_size={num_symbols}")
    
    # Initialiser le modèle monolithique
    model = MonolithModel(config=model_config)
    
    # Compiler le modèle
    model.model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "signal_output": keras.losses.CategoricalCrossentropy(),
            "sl_tp_output": keras.losses.MeanSquaredError(),
        },
        metrics={
            "signal_output": ["accuracy"],
            "sl_tp_output": [keras.metrics.MeanAbsoluteError()],
        },
        loss_weights={
            "signal_output": 1.0,
            "sl_tp_output": 0.5,
        }
    )
    
    # Résumé du modèle
    model.summary()
    
    # Créer un répertoire pour les logs et les checkpoints
    output_dir = os.path.join(PROJECT_ROOT, "ultimate", "model", "output", "minutely_test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks pour l'entraînement
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "minutely_model_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "tensorboard"),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ),
    ]
    
    # Entraîner le modèle
    logger.info(f"Début de l'entraînement pour {epochs} époques avec batch_size={batch_size}")
    history = model.model.fit(
        train_data,
        train_targets,
        validation_data=(val_data, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Sauvegarder le modèle
    model_path = os.path.join(output_dir, "minutely_model_final.keras")
    model.model.save(model_path)
    logger.info(f"Modèle sauvegardé à {model_path}")
    
    return model, history, model_path

def test_model(model, test_data, test_targets):
    """Évalue le modèle sur l'ensemble de test."""
    logger.info("Évaluation du modèle sur l'ensemble de test")
    
    # Évaluer le modèle
    results = model.model.evaluate(test_data, test_targets, verbose=1)
    
    # Imprimer les résultats
    for i, metric_name in enumerate(model.model.metrics_names):
        logger.info(f"{metric_name}: {results[i]}")
    
    # Prédire sur l'ensemble de test
    predictions = model.predict(test_data)
    
    # Extraire les signaux prédits
    predicted_signals = np.argmax(predictions["signal_output"], axis=1)
    true_signals = np.argmax(test_targets["signal_output"], axis=1)
    
    # Calculer la matrice de confusion
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(true_signals, predicted_signals)
    logger.info(f"Matrice de confusion:\n{cm}")
    
    # Rapport de classification
    report = classification_report(true_signals, predicted_signals)
    logger.info(f"Rapport de classification:\n{report}")
    
    # Sauvegarder les résultats
    output_dir = os.path.join(PROJECT_ROOT, "ultimate", "model", "output", "minutely_test")
    results_path = os.path.join(output_dir, "test_results.txt")
    
    with open(results_path, "w") as f:
        f.write(f"Matrice de confusion:\n{cm}\n\n")
        f.write(f"Rapport de classification:\n{report}\n\n")
        
        for i, metric_name in enumerate(model.model.metrics_names):
            f.write(f"{metric_name}: {results[i]}\n")
    
    logger.info(f"Résultats de test sauvegardés à {results_path}")
    
    return results, predictions

def main():
    """Fonction principale."""
    logger.info("Début du test du modèle monolithique avec dataset minutely")
    
    # Chemin vers le dataset minutely
    dataset_path = os.path.join(PROJECT_ROOT, "ultimate", "data", "processed", "minutely_crypto_dataset_fixed.parquet")
    
    # Charger le dataset
    df = load_dataset(dataset_path)
    
    # Préparer les données
    train_data, train_targets, val_data, val_targets, test_data, test_targets, tech_cols, tech_scaler, num_symbols = prepare_data(df)
    
    # Entraîner le modèle (avec moins d'époques pour un test rapide)
    model, history, model_path = train_model(train_data, train_targets, val_data, val_targets, num_symbols, epochs=5, batch_size=32)
    
    # Tester le modèle
    results, predictions = test_model(model, test_data, test_targets)
    
    logger.info("Test du modèle monolithique terminé avec succès")
    
    return model, history, results

if __name__ == "__main__":
    main() 