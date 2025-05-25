#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour collecter des données enrichies avec des informations de marché externes.
Ce script combine les données de marché de base avec des informations supplémentaires
de CoinMarketCap et d'autres sources gratuites.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests  # Ajout
import time  # Ajout
import warnings  # Ajout
from tqdm import tqdm  # Ajout
from transformers import AutoTokenizer, AutoModel  # Ajout
import torch  # Ajout
from hmmlearn import hmm  # Ajout (déjà présent via HMMRegimeDetector mais bon)
from sklearn.preprocessing import MinMaxScaler  # Ajout
from sklearn.decomposition import PCA  # Ajout
from sklearn.exceptions import ConvergenceWarning  # Ajout
import sys  # Ajout

# Importer nos modules personnalisés
from data_collectors.market_data_collector import MarketDataCollector
from data_collectors.news_collector import CryptoNewsCollector
from data_collectors.sentiment_analyzer import GeminiSentimentAnalyzer
from data_collectors.market_info_collector import initialize_market_info_collector
from data_processors.hmm_regime_detector import HMMRegimeDetector
from data_processors.cryptobert_processor import CryptoBERTProcessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# === CONFIGURATION SPÉCIFIQUE À L'ENRICHISSEMENT (inspiré de gpt01.py) ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL = "ElKulako/cryptobert"

# Credentials Google Search (à gérer via .env ou config sécurisée)
API_CREDENTIALS = [
    {"key": os.getenv("SEARCH_API_KEY1"), "cx": os.getenv("SEARCH_ENGINE_ID1")},
    # Ajouter d'autres clés si nécessaire
]
# Filtrer les credentials non valides
API_CREDENTIALS = [cred for cred in API_CREDENTIALS if cred["key"] and cred["cx"]]
enable_search = bool(API_CREDENTIALS)  # Activer la recherche seulement si des credentials sont valides

# === FONCTIONS UTILITAIRES (inspiré de gpt01.py) ===


def get_news_snippets(query: str, cache: dict) -> str:
    """
    Récupère les snippets quotidiens en utilisant la rotation de credentials
    et backoff exponentiel en cas de code 429.
    """
    if not enable_search:
        logger.warning("Google Search désactivé (pas de credentials valides trouvés dans .env)")
        return ""
    if query in cache:
        return cache[query]

    for cred in API_CREDENTIALS:
        backoff = 1
        for attempt in range(3):  # Limiter les tentatives par clé
            try:
                resp = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"key": cred["key"], "cx": cred["cx"], "q": query},
                    timeout=10,
                )
                resp.raise_for_status()  # Lève une exception pour les codes 4xx/5xx

                items = resp.json().get("items", [])
                text = " ".join(item.get("snippet", "") for item in items[:5])  # Limiter à 5 snippets
                cache[query] = text
                return text

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"Clé {cred['key'][:5]}... limitée, backoff {backoff}s")
                    time.sleep(backoff)
                    backoff *= 2
                    continue  # Essayer à nouveau avec la même clé après backoff
                else:
                    logger.warning(f"Clé {cred['key'][:5]}... erreur HTTP {e.response.status_code}: {e}")
                    break  # Essayer la clé suivante
            except requests.exceptions.RequestException as e:
                logger.warning(f"Erreur réseau Google Search: {e}")
                time.sleep(backoff)  # Attendre avant de réessayer (peut-être avec la même clé)
                backoff *= 2
                continue
            except Exception as e:
                logger.error(f"Erreur inattendue Google Search: {e}")
                break  # Essayer la clé suivante
        # Si toutes les tentatives échouent pour cette clé, passer à la suivante

    logger.warning(f"Impossible de récupérer les news pour '{query}' après toutes les tentatives.")
    cache[query] = ""  # Mettre en cache l'échec pour éviter de réessayer
    return ""


def get_bert_embeddings(texts: list, tokenizer, model, device, batch_size: int = 32) -> np.ndarray:
    """
    Génère les embeddings CLS pour une liste de textes via HuggingFace.
    Affiche une barre de progression.
    """
    if not texts:
        return np.array([]).reshape(0, model.config.hidden_size)  # Retourner un array vide avec la bonne dimension

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings", leave=False):
            batch = texts[i : i + batch_size]
            # Remplacer les None ou NaN par des chaînes vides
            batch = [str(text) if pd.notna(text) else "" for text in batch]

            try:
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=128,  # Augmenter un peu la longueur max ?
                )
                ids = encoded["input_ids"].to(device)
                mask = encoded["attention_mask"].to(device)
                outputs = model(input_ids=ids, attention_mask=mask)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
            except Exception as e:
                logger.error(f"Erreur lors de l'encodage BERT du batch {i}: {e}")
                # Ajouter des embeddings zéros pour ce batch pour maintenir la taille
                num_failed = len(batch)
                zero_emb = np.zeros((num_failed, model.config.hidden_size))
                embeddings.append(zero_emb)

    if not embeddings:
        return np.array([]).reshape(0, model.config.hidden_size)

    return np.vstack(embeddings)


# === FIN FONCTIONS UTILITAIRES ===


def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Collecte de données enrichies")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT",
        help="Liste des symboles de crypto-monnaies séparés par des virgules",
    )
    parser.add_argument(
        "--timeframe", type=str, default="1d", help="Timeframe pour les données OHLCV (1m, 5m, 15m, 1h, 4h, 1d)"
    )
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Date de début au format YYYY-MM-DD")
    parser.add_argument(
        "--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="Date de fin au format YYYY-MM-DD"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/enriched", help="Répertoire de sortie pour les données enrichies"
    )
    parser.add_argument("--use-sentiment", action="store_true", help="Inclure l'analyse de sentiment")
    parser.add_argument("--use-news", action="store_true", help="Inclure les actualités crypto")
    parser.add_argument(
        "--use-market-info", action="store_true", help="Inclure les informations de marché de CoinMarketCap"
    )
    parser.add_argument("--use-hmm", action="store_true", help="Inclure la détection de régime HMM")
    parser.add_argument("--use-cryptobert", action="store_true", help="Inclure les embeddings CryptoBERT")

    return parser.parse_args()


def collect_market_data(symbols, timeframe, start_date, end_date):
    """
    Collecte les données de marché de base.

    Args:
        symbols: Liste des symboles
        timeframe: Timeframe pour les données OHLCV
        start_date: Date de début
        end_date: Date de fin

    Returns:
        DataFrame avec les données de marché
    """
    # Convertir la liste de symboles en liste si elle est fournie comme une chaîne
    if isinstance(symbols, str):
        symbols_list = symbols.split(",")
    else:
        symbols_list = symbols

    logger.info(f"Collecte des données de marché pour {len(symbols_list)} symboles")

    # Initialiser le collecteur de données de marché
    market_collector = MarketDataCollector()

    # Collecter les données OHLCV pour chaque symbole et les concaténer
    all_data = []
    for symbol in symbols_list:
        logger.info(f"Collecte des données pour {symbol}")
        try:
            # Utiliser la méthode fetch_ohlcv disponible
            symbol_data = market_collector.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=start_date, until=end_date
            )

            if symbol_data is not None and not symbol_data.empty:
                # Ajouter la colonne symbole
                symbol_data["symbol"] = symbol
                all_data.append(symbol_data)
            else:
                logger.warning(f"Aucune donnée collectée pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données pour {symbol}: {e}")

    # Concaténer tous les DataFrames
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Données collectées: {len(df)} lignes pour {len(all_data)} symboles")

        # Ajouter les indicateurs techniques
        df = market_collector.add_technical_indicators(df)

        logger.info(f"Données de marché collectées avec succès: {len(df)} lignes")
        return df
    else:
        logger.error("Aucune donnée collectée")
        return pd.DataFrame()


def enrich_with_market_info(df, symbols):
    """
    Enrichit les données avec des informations de marché supplémentaires.

    Args:
        df: DataFrame à enrichir
        symbols: Liste des symboles

    Returns:
        DataFrame enrichi
    """
    logger.info("Enrichissement avec des informations de marché supplémentaires")

    # Initialiser le collecteur d'informations de marché
    market_info_collector = initialize_market_info_collector()

    # Enrichir les données
    enriched_df = market_info_collector.enrich_market_data(df, symbols)

    logger.info(f"Données enrichies avec des informations de marché: {len(enriched_df.columns)} colonnes")
    return enriched_df


def enrich_with_sentiment(df, symbols):
    """
    Enrichit les données avec l'analyse de sentiment.

    Args:
        df: DataFrame à enrichir
        symbols: Liste des symboles

    Returns:
        DataFrame enrichi
    """
    logger.info("Enrichissement des données avec l'analyse de sentiment")

    # Initialiser l'analyseur de sentiment
    sentiment_analyzer = GeminiSentimentAnalyzer()

    # Collecter les actualités pour chaque symbole
    news_collector = CryptoNewsCollector()

    # Initialiser le collecteur d'informations de marché pour obtenir des actualités supplémentaires
    market_info_collector = initialize_market_info_collector()

    # Créer une copie du DataFrame
    df_enriched = df.copy()

    # Ajouter des colonnes pour le sentiment
    df_enriched["sentiment_score"] = 0.0
    df_enriched["sentiment_magnitude"] = 0.0
    df_enriched["sentiment_positive"] = 0.0
    df_enriched["sentiment_negative"] = 0.0
    df_enriched["sentiment_neutral"] = 0.0

    # Traiter chaque date unique dans le DataFrame
    unique_dates = df_enriched["timestamp"].dt.date.unique()

    for date in unique_dates:
        logger.info(f"Traitement du sentiment pour la date {date}")

        # Convertir la date en string pour fetch_crypto_news
        date_str = date.strftime("%Y-%m-%d")

        # Pour chaque symbole
        for symbol in symbols:
            # Extraire le symbole de base (sans /USDT)
            base_symbol = symbol.split("/")[0]

            try:
                # Obtenir les actualités de notre collecteur
                crypto_news = news_collector.fetch_crypto_news(date_str, base_symbol)

                # Obtenir les actualités de CoinMarketCap
                cmc_news = market_info_collector.get_crypto_news(base_symbol, limit=5)

                # Fusionner les actualités
                symbol_news = crypto_news + cmc_news

                if symbol_news:
                    # Analyser le sentiment des actualités
                    sentiment_result = sentiment_analyzer.analyze_sentiment(symbol_news, base_symbol)

                    # Filtrer les lignes pour ce symbole et cette date
                    mask = (df_enriched["symbol"] == symbol) & (df_enriched["timestamp"].dt.date == date)

                    # Mettre à jour les valeurs de sentiment
                    df_enriched.loc[mask, "sentiment_score"] = sentiment_result.get("sentiment_score", 0)
                    df_enriched.loc[mask, "sentiment_magnitude"] = sentiment_result.get("sentiment_magnitude", 0)
                    df_enriched.loc[mask, "sentiment_positive"] = sentiment_result.get("bullish_probability", 0.5)
                    df_enriched.loc[mask, "sentiment_negative"] = sentiment_result.get("bearish_probability", 0.5)
                    df_enriched.loc[mask, "sentiment_neutral"] = (
                        1.0
                        - (
                            sentiment_result.get("bullish_probability", 0.5)
                            + sentiment_result.get("bearish_probability", 0.5)
                        )
                        / 2
                    )
            except Exception as e:
                logger.warning(f"Erreur lors de l'analyse du sentiment pour {symbol} à la date {date}: {e}")

    return df_enriched


def enrich_with_hmm(df):
    """
    Enrichit les données avec la détection de régime HMM.

    Args:
        df: DataFrame à enrichir

    Returns:
        DataFrame enrichi
    """
    logger.info("Enrichissement avec la détection de régime HMM")

    # Initialiser le détecteur de régime HMM
    hmm_detector = HMMRegimeDetector()

    # Créer une copie du DataFrame pour éviter de modifier l'original
    enriched_df = df.copy()

    # Ajouter des colonnes pour les résultats HMM
    if "hmm_regime" not in enriched_df.columns:
        enriched_df["hmm_regime"] = 0
    if "hmm_prob_0" not in enriched_df.columns:
        enriched_df["hmm_prob_0"] = 0.0
    if "hmm_prob_1" not in enriched_df.columns:
        enriched_df["hmm_prob_1"] = 0.0
    if "hmm_prob_2" not in enriched_df.columns:
        enriched_df["hmm_prob_2"] = 0.0

    # Pour chaque symbole
    for symbol in enriched_df["symbol"].unique():
        logger.info(f"Détection de régime HMM pour {symbol}")

        try:
            # Filtrer les données pour ce symbole
            symbol_data = enriched_df[enriched_df["symbol"] == symbol].copy()

            # Trier par timestamp
            symbol_data = symbol_data.sort_values("timestamp")

            # Calculer les rendements journaliers
            symbol_data["returns"] = symbol_data["close"].pct_change()

            # Supprimer les lignes avec des rendements NaN
            symbol_data = symbol_data.dropna(subset=["returns"])

            if len(symbol_data) > 10:  # Vérifier qu'il y a assez de données
                # Extraire les rendements comme un array numpy
                returns = symbol_data["returns"].values

                # Entraîner le modèle HMM et détecter les régimes
                regimes, proba = hmm_detector.detect_regimes(returns)

                # Ajouter les résultats au DataFrame
                symbol_data["hmm_regime"] = regimes

                # Ajouter les probabilités pour chaque régime
                for i in range(min(3, proba.shape[1])):
                    symbol_data[f"hmm_prob_{i}"] = proba[:, i]

                # Mettre à jour le DataFrame original
                for col in ["hmm_regime", "hmm_prob_0", "hmm_prob_1", "hmm_prob_2"]:
                    if col in symbol_data.columns:
                        # Utiliser les index pour la mise à jour
                        enriched_df.loc[symbol_data.index, col] = symbol_data[col].values
            else:
                logger.warning(f"Pas assez de données pour {symbol} pour la détection de régime HMM")
        except Exception as e:
            logger.error(f"Erreur lors de la détection de régime HMM pour {symbol}: {e}")

    logger.info(f"Données enrichies avec la détection de régime HMM")
    return enriched_df


# Modifier enrich_with_cryptobert pour utiliser les nouvelles fonctions
def enrich_with_cryptobert(df, symbols):
    """
    Enrichit les données avec les embeddings CryptoBERT.

    Args:
        df: DataFrame à enrichir
        symbols: Liste des symboles

    Returns:
        DataFrame enrichi
    """
    """
    Enrichit les données avec les embeddings CryptoBERT (version gpt01.py).
    """
    logger.info("Enrichissement des données avec les embeddings CryptoBERT (style gpt01)")

    # Créer une copie du DataFrame
    df_enriched = df.copy()

    # Assurer l'existence de la colonne 'date'
    if "date" not in df_enriched.columns:
        df_enriched["date"] = df_enriched["timestamp"].dt.date

    # Récupération des snippets d'actualité (quotidien)
    logger.info("Récupération des snippets d'actualités...")
    cache = {}
    unique_dates = df_enriched["date"].unique()
    snippets_map = {}
    # Utiliser tqdm pour la barre de progression
    for date in tqdm(unique_dates, desc="Fetching news snippets"):
        for symbol in symbols:
            base_symbol = symbol.split("/")[0]
            query = f"{base_symbol} crypto market news {date}"
            # Utiliser un tuple (date, symbol) comme clé pour le mapping
            snippets_map[(date, symbol)] = get_news_snippets(query, cache)

    # Mapper les snippets au DataFrame principal
    # Créer une clé multi-index temporaire pour le mapping
    df_enriched["date_symbol_key"] = list(zip(df_enriched["date"], df_enriched["symbol"]))
    df_enriched["news_snippets"] = df_enriched["date_symbol_key"].map(snippets_map)
    df_enriched.drop(columns=["date_symbol_key"], inplace=True)  # Supprimer la clé temporaire

    # Génération des embeddings BERT
    logger.info("Génération des embeddings BERT...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        bert_model = AutoModel.from_pretrained(BERT_MODEL).to(DEVICE)

        # Remplacer None/NaN par "" avant de générer les embeddings
        news_list = df_enriched["news_snippets"].fillna("").tolist()

        if not news_list:
            logger.warning("Aucun snippet d'actualité à traiter pour BERT.")
            # Ajouter des colonnes bert_* avec des zéros si elles n'existent pas
            bert_cols = [f"bert_{i}" for i in range(768)]  # Supposons 768 dimensions
            for col in bert_cols:
                if col not in df_enriched.columns:
                    df_enriched[col] = 0.0
            return df_enriched

        emb = get_bert_embeddings(news_list, tokenizer, bert_model, DEVICE)

        if emb.shape[0] == len(df_enriched):
            bert_cols = [f"bert_{i}" for i in range(emb.shape[1])]
            df_enriched[bert_cols] = emb
            logger.info(f"Embeddings BERT ajoutés ({emb.shape[1]} dimensions)")
        else:
            logger.error(
                f"Incohérence de taille entre les embeddings ({emb.shape[0]}) et le DataFrame ({len(df_enriched)}). Skipping BERT."
            )
            # Ajouter des colonnes bert_* avec des zéros
            bert_cols = [f"bert_{i}" for i in range(768)]  # Supposons 768 dimensions
            for col in bert_cols:
                if col not in df_enriched.columns:
                    df_enriched[col] = 0.0

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation ou de la génération des embeddings BERT: {e}")
        # Ajouter des colonnes bert_* avec des zéros en cas d'erreur
        bert_cols = [f"bert_{i}" for i in range(768)]  # Supposons 768 dimensions
        for col in bert_cols:
            if col not in df_enriched.columns:
                df_enriched[col] = 0.0

    return df_enriched


def add_mcp_features(df):
    """Ajoute les Market Condition Proxies (MCP) par PCA."""
    logger.info("Ajout des features MCP par PCA")
    enriched_df = df.copy()

    # Colonnes à utiliser pour PCA (doivent être numériques et présentes)
    pca_cols = ["close", "volume", "hmm_regime"]

    # Vérifier si les colonnes existent et sont numériques
    valid_pca_cols = [
        col for col in pca_cols if col in enriched_df.columns and pd.api.types.is_numeric_dtype(enriched_df[col])
    ]

    if len(valid_pca_cols) < 2:  # Besoin d'au moins 2 features pour PCA
        logger.warning(f"Pas assez de colonnes valides ({valid_pca_cols}) pour calculer les MCP via PCA. Skipping.")
        # Ajouter des colonnes MCP avec des zéros si elles n'existent pas
        for i in range(1, 4):
            if f"mcp_{i}" not in enriched_df.columns:
                enriched_df[f"mcp_{i}"] = 0.0
        return enriched_df

    logger.info(f"Colonnes utilisées pour PCA MCP: {valid_pca_cols}")

    # Gérer les NaN avant PCA
    pca_data = enriched_df[valid_pca_cols].fillna(0)

    try:
        pca = PCA(n_components=3, random_state=42)
        mcp_features = pca.fit_transform(pca_data)

        # Ajouter les features MCP au DataFrame
        for i in range(mcp_features.shape[1]):
            enriched_df[f"mcp_{i+1}"] = mcp_features[:, i]
        logger.info("Features MCP ajoutées.")

    except Exception as e:
        logger.error(f"Erreur lors du calcul PCA pour MCP: {e}")
        # Ajouter des colonnes MCP avec des zéros en cas d'erreur
        for i in range(1, 4):
            if f"mcp_{i}" not in enriched_df.columns:
                enriched_df[f"mcp_{i}"] = 0.0

    return enriched_df


def generate_labels(df):
    """
    Génère les labels pour l'entraînement supervisé.

    Args:
        df: DataFrame à enrichir

    Returns:
        DataFrame avec les labels
    """
    logger.info("Génération des labels")

    # Créer une copie du DataFrame pour éviter de modifier l'original
    labeled_df = df.copy()

    # Pour chaque symbole
    for symbol in labeled_df["symbol"].unique():
        # Filtrer les données pour ce symbole
        symbol_data = labeled_df[labeled_df["symbol"] == symbol].copy()

        # Trier par timestamp
        symbol_data = symbol_data.sort_values("timestamp")

        # Calculer les rendements futurs
        symbol_data["future_return_1d"] = symbol_data["close"].pct_change(1).shift(-1)
        symbol_data["future_return_3d"] = symbol_data["close"].pct_change(3).shift(-3)
        symbol_data["future_return_7d"] = symbol_data["close"].pct_change(7).shift(-7)

        # Définir le régime de marché (0: stable/baissier, 1: haussier)
        symbol_data["market_regime"] = (symbol_data["future_return_1d"] > 0.01).astype(int)

        # Calculer les niveaux de stop loss et take profit optimaux
        # SL: -2% du prix de clôture si haussier, -1% si baissier
        # TP: +3% du prix de clôture si haussier, +1.5% si baissier
        symbol_data["level_sl"] = np.where(
            symbol_data["market_regime"] == 1, -0.02, -0.01  # SL pour régime haussier  # SL pour régime stable/baissier
        )

        symbol_data["level_tp"] = np.where(
            symbol_data["market_regime"] == 1, 0.03, 0.015  # TP pour régime haussier  # TP pour régime stable/baissier
        )

        # Mettre à jour le DataFrame original
        labeled_df.loc[labeled_df["symbol"] == symbol, "future_return_1d"] = symbol_data["future_return_1d"]
        labeled_df.loc[labeled_df["symbol"] == symbol, "future_return_3d"] = symbol_data["future_return_3d"]
        labeled_df.loc[labeled_df["symbol"] == symbol, "future_return_7d"] = symbol_data["future_return_7d"]
        labeled_df.loc[labeled_df["symbol"] == symbol, "market_regime"] = symbol_data["market_regime"]
        labeled_df.loc[labeled_df["symbol"] == symbol, "level_sl"] = symbol_data["level_sl"]
        labeled_df.loc[labeled_df["symbol"] == symbol, "level_tp"] = symbol_data["level_tp"]

    # Supprimer les lignes avec des valeurs NaN dans les labels
    labeled_df = labeled_df.dropna(subset=["future_return_1d", "market_regime", "level_sl", "level_tp"])

    logger.info(f"Labels générés avec succès: {len(labeled_df)} lignes")
    return labeled_df


def split_dataset(df):
    """
    Divise le dataset en ensembles d'entraînement, de validation et de test.

    Args:
        df: DataFrame à diviser

    Returns:
        DataFrame avec une colonne 'split' indiquant l'ensemble
    """
    logger.info("Division du dataset")

    # Créer une copie du DataFrame pour éviter de modifier l'original
    split_df = df.copy()

    # Trier par timestamp
    split_df = split_df.sort_values("timestamp")

    # Définir les proportions
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Ajouter une colonne 'split'
    split_df["split"] = "train"

    # Pour chaque symbole
    for symbol in split_df["symbol"].unique():
        # Filtrer les données pour ce symbole
        symbol_data = split_df[split_df["symbol"] == symbol]

        # Calculer les indices de division
        n = len(symbol_data)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)

        # Mettre à jour les splits
        split_df.loc[symbol_data.index[train_idx:val_idx], "split"] = "val"
        split_df.loc[symbol_data.index[val_idx:], "split"] = "test"

    # Compter le nombre d'exemples dans chaque ensemble
    train_count = (split_df["split"] == "train").sum()
    val_count = (split_df["split"] == "val").sum()
    test_count = (split_df["split"] == "test").sum()

    logger.info(f"Dataset divisé: {train_count} train, {val_count} val, {test_count} test")
    return split_df


def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()

    # Convertir la liste de symboles en liste
    symbols = args.symbols.split(",")

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)

    # Collecter les données de marché de base
    df = collect_market_data(symbols, args.timeframe, args.start_date, args.end_date)

    # Enrichir avec des informations de marché supplémentaires si demandé
    if args.use_market_info:
        df = enrich_with_market_info(df, symbols)

    # Enrichir avec l'analyse de sentiment si demandé
    if args.use_sentiment:
        df = enrich_with_sentiment(df, symbols)

    # Enrichir avec la détection de régime HMM si demandé (nécessaire pour MCP)
    if args.use_hmm:
        df = enrich_with_hmm(df)
    else:
        # S'assurer que la colonne hmm_regime existe même si HMM n'est pas calculé,
        # pour que add_mcp_features ne plante pas (elle sera remplie de 0)
        if "hmm_regime" not in df.columns:
            df["hmm_regime"] = 0
            logger.info("Colonne 'hmm_regime' ajoutée avec des zéros car --use-hmm n'est pas activé.")

    # Ajouter les features MCP (calculées à partir de close, volume, hmm_regime)
    # On les ajoute systématiquement si les colonnes sources sont disponibles
    df = add_mcp_features(df)

    # Enrichir avec les embeddings CryptoBERT si demandé
    if args.use_cryptobert:
        # La fonction enrich_with_cryptobert gère maintenant la récupération des news
        df = enrich_with_cryptobert(df, symbols)
    else:
        # S'assurer que les colonnes bert_* existent même si non calculées
        # (remplies de 0) pour la cohérence du schéma
        bert_cols = [f"bert_{i}" for i in range(768)]  # Supposons 768 dimensions
        for col in bert_cols:
            if col not in df.columns:
                df[col] = 0.0
        logger.info("Embeddings CryptoBERT non générés (--use-cryptobert non activé). Colonnes ajoutées avec zéros.")

    # Générer les labels
    df = generate_labels(df)

    # Diviser le dataset
    df = split_dataset(df)

    # Sauvegarder le dataset final
    output_path = os.path.join(args.output_dir, "enriched_dataset.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset enrichi sauvegardé dans {output_path}")

    # Afficher des statistiques sur le dataset final
    logger.info(f"Dataset final: {len(df)} lignes, {len(df.columns)} colonnes")
    logger.info(f"Colonnes: {', '.join(df.columns.tolist()[:10])}...")


if __name__ == "__main__":
    main()
