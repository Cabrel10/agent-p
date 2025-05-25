#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de démonstration du modèle monolithique Morningstar.

Ce script crée des données synthétiques et montre comment:
1. Instancier et configurer le modèle monolithique
2. Entraîner le modèle sur des données
3. Évaluer les performances
4. Faire des prédictions
5. Sauvegarder et charger le modèle
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from pathlib import Path

# --- Fonctions d'augmentation et utilitaires contrastifs ---
from .contrastive_utils import (
    jitter, scaling, time_masking,
    generate_contrastive_pairs_batch, info_nce_loss, tf_info_nce_loss
)
# Importer les composants monolithiques
from .monolith_model import MonolithModel
from .inference_monolith import prepare_inference_data, interpret_predictions, visualize_predictions

# Configuration du logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("demo_monolith")


def create_synthetic_data(n_samples=1000, n_technical=38, n_mcp=128, seq_length=None):
    """
    Crée des données synthétiques pour la démonstration.
    
    Args:
        n_samples: Nombre d'échantillons
        n_technical: Nombre de features techniques
        n_mcp: Nombre de features MCP
        seq_length: Longueur de séquence (None pour non-séquentiel)
    
    Returns:
        Tuple (inputs, outputs, df)
    """
    logger.info(f"Création de {n_samples} échantillons de données synthétiques")
    
    # Générer des dates
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Créer DataFrame
    df = pd.DataFrame(index=dates)
    
    # Générer prix (pour visualisation)
    price = 100 + np.cumsum(np.random.normal(0, 1, n_samples) * 0.5)
    df['open'] = price - np.random.uniform(0, 1, n_samples)
    df['high'] = price + np.random.uniform(0, 2, n_samples)
    df['low'] = price - np.random.uniform(0, 2, n_samples)
    df['close'] = price + np.random.uniform(-1, 1, n_samples)
    df['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Générer des indicateurs techniques (aléatoires pour la démo)
    for i in range(n_technical - 5):  # -5 car on a déjà OHLCV
        df[f'tech_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Générer un embedding synthétique
    embeddings = np.random.normal(0, 1, (n_samples, 768))
    df['news_embedding'] = [emb.tolist() for emb in embeddings]
    
    # Générer des features MCP
    for i in range(n_mcp):
        df[f'mcp_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Générer des instruments (3 différents)
    instruments = np.random.choice(['BTC', 'ETH', 'SOL'], n_samples)
    df['symbol'] = instruments
    
    # Générer des cibles pour l'entraînement
    
    # Signal: 0=Sell, 1=Neutral, 2=Buy (généré aléatoirement mais avec biais)
    trend = np.cumsum(np.random.normal(0, 1, n_samples) * 0.1)
    trend_norm = (trend - np.min(trend)) / (np.max(trend) - np.min(trend))
    
    # Convertir en probabilités de classe
    probs = np.zeros((n_samples, 3))
    for i in range(n_samples):
        t = trend_norm[i]
        if t < 0.3:
            # Tendance baissière -> plus de sell
            probs[i] = [0.6, 0.3, 0.1]
        elif t > 0.7:
            # Tendance haussière -> plus de buy
            probs[i] = [0.1, 0.3, 0.6]
        else:
            # Neutre
            probs[i] = [0.2, 0.6, 0.2]
        
        # Ajouter du bruit
        probs[i] += np.random.normal(0, 0.1, 3)
        probs[i] = np.clip(probs[i], 0.01, 0.99)
        probs[i] /= np.sum(probs[i])  # Normaliser
    
    df['signal_sell'] = probs[:, 0]
    df['signal_neutral'] = probs[:, 1]
    df['signal_buy'] = probs[:, 2]
    
    # Générer SL/TP basés sur la tendance
    df['stop_loss'] = df['close'] * (1 - np.random.uniform(0.01, 0.05, n_samples))
    df['take_profit'] = df['close'] * (1 + np.random.uniform(0.02, 0.10, n_samples))
    
    # Préparer les entrées pour le modèle
    tech_cols = ['open', 'high', 'low', 'close', 'volume'] + [f'tech_{i}' for i in range(n_technical - 5)]
    mcp_cols = [f'mcp_{i}' for i in range(n_mcp)]
    
    # Préparer les dictionnaires d'entrée/sortie
    inputs = {
        "technical_input": df[tech_cols].values,
        "embeddings_input": embeddings,
        "mcp_input": df[mcp_cols].values,
        "instrument_input": np.array([{'BTC': 0, 'ETH': 1, 'SOL': 2}[s] for s in df['symbol']]).reshape(-1, 1)
    }
    
    outputs = {
        "signal_output": probs,
        "sl_tp_output": np.column_stack([df['stop_loss'].values, df['take_profit'].values])
    }
    
    # Ajouter séquence si nécessaire
    if seq_length is not None:
        logger.info(f"Conversion des données en séquences de longueur {seq_length}")
        
        # Transformer les données techniques en séquences
        X_tech_seq = []
        for i in range(seq_length, n_samples):
            X_tech_seq.append(inputs["technical_input"][i-seq_length:i])
        X_tech_seq = np.array(X_tech_seq)
        
        # Adapter les autres entrées à la nouvelle taille
        seq_n = len(X_tech_seq)
        inputs = {
            "technical_input": X_tech_seq,
            "embeddings_input": inputs["embeddings_input"][-seq_n:],
            "mcp_input": inputs["mcp_input"][-seq_n:],
            "instrument_input": inputs["instrument_input"][-seq_n:]
        }
        
        # Adapter les sorties
        outputs = {
            "signal_output": outputs["signal_output"][-seq_n:],
            "sl_tp_output": outputs["sl_tp_output"][-seq_n:]
        }
        
        # Adapter le DataFrame
        df = df.iloc[-seq_n:]
    
    logger.info(f"Données synthétiques créées: {len(df)} échantillons")
    return inputs, outputs, df


def demo_training(model, inputs, outputs):
    """Démo d'entraînement du modèle."""
    logger.info("Démonstration de l'entraînement")
    
    # Diviser les données en train/validation
    n_samples = len(next(iter(inputs.values())))
    n_train = int(n_samples * 0.8)
    
    train_inputs = {k: v[:n_train] for k, v in inputs.items()}
    train_outputs = {k: v[:n_train] for k, v in outputs.items()}
    
    val_inputs = {k: v[n_train:] for k, v in inputs.items()}
    val_outputs = {k: v[n_train:] for k, v in outputs.items()}
    
    # Configurer les callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Entraîner le modèle
    history = model.train(
        train_inputs, 
        train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=20,  # Réduit pour la démo
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluer le modèle
    results = model.evaluate(val_inputs, val_outputs, verbose=1)
    logger.info(f"Résultats d'évaluation: {results}")
    
    # Visualiser l'historique d'entraînement
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Perte d\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'signal_output_accuracy' in history.history:
        plt.plot(history.history['signal_output_accuracy'], label='Signal Accuracy (Train)')
        plt.plot(history.history['val_signal_output_accuracy'], label='Signal Accuracy (Val)')
        plt.title('Précision du signal')
        plt.xlabel('Époque')
        plt.ylabel('Précision')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Historique d'entraînement sauvegardé dans training_history.png")
    
    return history, results


def demo_inference(model, inputs, df):
    """Démo d'inférence avec le modèle."""
    logger.info("Démonstration de l'inférence")
    
    # Faire des prédictions
    predictions = model.predict(inputs)
    
    # Interpréter les prédictions
    interpreted = interpret_predictions(predictions)
    
    # Créer un DataFrame avec les prédictions
    result_df = df.copy()
    
    # Ajouter les prédictions au DataFrame
    for key, values in interpreted.items():
        result_df[f'pred_{key}'] = values
    
    # Pour les probabilités de signal, ajouter chaque classe
    if "signal_output" in predictions:
        signal_probs = predictions["signal_output"]
        result_df['pred_sell'] = signal_probs[:, 0]
        result_df['pred_neutral'] = signal_probs[:, 1]
        result_df['pred_buy'] = signal_probs[:, 2]
    
    # Visualiser les prédictions
    plt.figure(figsize=(12, 10))
    visualize_predictions(result_df, interpreted, 'predictions.png')
    logger.info("Visualisation des prédictions sauvegardée dans predictions.png")
    
    return result_df


def main():
    """Fonction principale de démonstration."""
    logger.info("Démarrage de la démonstration du modèle monolithique")
    
    # 1. Créer des données synthétiques
    inputs, outputs, df = create_synthetic_data(n_samples=1000)
    
    # 2. Configurer le modèle monolithique
    config = {
        "tech_input_shape": (inputs["technical_input"].shape[1],),
        "embeddings_input_shape": inputs["embeddings_input"].shape[1],
        "mcp_input_shape": inputs["mcp_input"].shape[1],
        "instrument_vocab_size": 3,  # BTC, ETH, SOL
        "backbone_config": {
            "dense_units": 64,
            "lstm_units": 32,
            "transformer_blocks": 1  # Réduit pour la démo
        },
        "dropout_rate": 0.2,
        "learning_rate": 5e-4
    }
    
    # 3. Instancier le modèle
    model = MonolithModel(config=config)
    model.summary()

    # --- Démo contrastive : création de paires augmentées et passage par la tête de projection ---
    logger.info("Démo contrastive : génération de paires positives/négatives et passage par la tête de projection")

    # Sélectionner un échantillon (fenêtre) de technical_input
    idx = 100
    window = inputs["technical_input"][idx].copy()  # shape: (n_features,)
    # Générer une version positive (augmentation)
    window_pos = jitter(window, sigma=0.05)
    # Générer une version négative (autre fenêtre)
    idx_neg = 200
    window_neg = inputs["technical_input"][idx_neg].copy()

    # Préparer les batchs pour le modèle (reshape et compléter les autres entrées)
    def make_batch(window_arr):
        # window_arr: (n_features,)
        batch = {
            "technical_input": np.expand_dims(window_arr, axis=0),
            "embeddings_input": np.expand_dims(inputs["embeddings_input"][idx], axis=0),
            "mcp_input": np.expand_dims(inputs["mcp_input"][idx], axis=0),
            "instrument_input": np.expand_dims(inputs["instrument_input"][idx], axis=0)
        }
        return batch

    batch_anchor = make_batch(window)
    batch_pos = make_batch(window_pos)
    batch_neg = make_batch(window_neg)

    # Passer par le modèle (projection head)
    proj_anchor = model.model.predict(batch_anchor)["projection"]
    proj_pos = model.model.predict(batch_pos)["projection"]
    proj_neg = model.model.predict(batch_neg)["projection"]

    logger.info(f"Shape projection anchor: {proj_anchor.shape}, pos: {proj_pos.shape}, neg: {proj_neg.shape}")
    logger.info(f"Cosine(anchor, pos): {np.dot(proj_anchor, proj_pos.T).item():.4f} | Cosine(anchor, neg): {np.dot(proj_anchor, proj_neg.T).item():.4f}")

    # --- Pipeline contrastif : génération dynamique d'un batch et calcul de la perte InfoNCE ---
    logger.info("Pipeline contrastif : génération dynamique d'un batch et calcul de la perte InfoNCE")
    batch_anchor, batch_pos, batch_neg = generate_contrastive_pairs_batch(inputs, batch_size=32, augmentation_fn=jitter)

    # Passage par la tête de projection pour chaque vue
    proj_anchor = model.model.predict(batch_anchor)["projection"]
    proj_pos = model.model.predict(batch_pos)["projection"]
    proj_neg = model.model.predict(batch_neg)["projection"]

    # Calcul de la perte InfoNCE sur le batch
    loss_contrastive = info_nce_loss(proj_anchor, proj_pos, proj_neg, temperature=0.1)
    logger.info(f"Perte InfoNCE sur le batch contrastif : {loss_contrastive:.4f}")

    # --- Démonstration de l'entraînement contrastif ---
    logger.info("Démonstration de l'entraînement contrastif du MonolithModel")

    # Utiliser les 'inputs' synthétiques. 'outputs' n'est pas nécessaire pour l'entraînement contrastif.
    # Choisir une fonction d'augmentation parmi celles définies (jitter, scaling, time_masking)
    augmentation_pour_contrastif = jitter # ou scaling, ou time_masking

    # Entraîner en mode contrastif pour quelques époques
    contrastive_history = model.train(
        inputs=inputs, 
        outputs=None, # Non utilisé par la boucle d'entraînement contrastif
        epochs=5,     # Petit nombre d'époques pour la démonstration
        batch_size=16, # Peut être différent du batch_size de l'auto-encodeur
        contrastive_training=True,
        contrastive_augmentation_fn=augmentation_pour_contrastif,
        contrastive_temperature=0.07 # Exemple de température
    )
    logger.info(f"Historique de l'entraînement contrastif : {contrastive_history.history}")
    logger.info("L'entraînement contrastif a utilisé la boucle personnalisée dans MonolithModel.")

    # 4. Entraîner le modèle
    # history, results = demo_training(model, inputs, outputs)
    
    # 5. Faire des prédictions
    result_df = demo_inference(model, inputs, df)
    
    # 6. Sauvegarder le modèle
    output_dir = "output_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "monolith_model.keras")
    model.save(model_path)
    logger.info(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarder les métadonnées
    metadata = {
        "tech_cols": ['open', 'high', 'low', 'close', 'volume'] + 
                    [f'tech_{i}' for i in range(config["tech_input_shape"][0] - 5)],
        "llm_cols": ["news_embedding"],
        "mcp_cols": [f'mcp_{i}' for i in range(config["mcp_input_shape"])],
        "instrument_map": {"BTC": 0, "ETH": 1, "SOL": 2},
        "tech_input_shape": config["tech_input_shape"][0],
        "embeddings_input_shape": config["embeddings_input_shape"],
        "mcp_input_shape": config["mcp_input_shape"]
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 7. Charger le modèle et vérifier qu'il fonctionne
    loaded_model = MonolithModel.load(model_path)
    
    # Faire des prédictions avec le modèle chargé
    loaded_predictions = loaded_model.predict(inputs)
    logger.info(f"Vérification des prédictions du modèle chargé: {list(loaded_predictions.keys())}")
    
    # Sauvegarder les résultats de prédiction
    result_df.to_csv(os.path.join(output_dir, "predictions.csv"))
    
    logger.info("Démonstration terminée avec succès!")


if __name__ == "__main__":
    main()
