#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour exécuter des backtests du modèle Morningstar en ligne de commande.

Usage:
    python run_backtest.py --pair ADA/USDT \\
        --data-path ultimate/data/processed/enriched_dataset.parquet \\
        --model ultimate/outputs/enhanced/best_model.keras \\
        --results-dir ultimate/results/backtest_cli_output \\
        [--plot] [--loglevel DEBUG]
    
    python run_backtest.py --pair ADA/USDT \\
        --data-dir ultimate/data/ \\
        --model ultimate/outputs/enhanced/best_model.keras \\
        [--initial-capital 1000] [--commission 0.001]
"""
import argparse
import logging
import logging.handlers # For RotatingFileHandler
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import backtrader as bt
import matplotlib.pyplot as plt # For saving plots

# Logger placeholder, will be configured in main()
logger = logging.getLogger(__name__)

# ----------------------------------------
# Constantes par défaut
# ----------------------------------------
DEFAULT_DATA_DIR_NAME    = "data" # Default directory name if only --data-dir is used relative to project
DEFAULT_RESULTS_DIR_NAME = "results/backtest" # Default directory name
DEFAULT_MODEL_FILENAME   = "model/saved_model/morningstar_final.h5"
DEFAULT_INITIAL_CAPITAL  = 100.0
DEFAULT_COMMISSION_FEE   = 0.002 # Renamed for clarity (was DEFAULT_TRANSACTION_FEE)
DEFAULT_SLIPPAGE_PERC    = 0.0005 # Explicitly percentage (0.05%)
DEFAULT_SIGNAL_THRESHOLD = 0.6

# ----------------------------------------
# Strategy et DataFeed pour Backtrader
# ----------------------------------------
class MorningstarStrategy(bt.Strategy):
    params = (
        ("default_sl_pct", 0.02),    # Default Stop Loss percentage
        ("default_tp_pct", 0.04),    # Default Take Profit percentage
        ("use_sl_tp", True),         # WhADAer to use SL/TP orders
        ("risk_per_trade", 0.02),    # Percentage of portfolio to risk per trade
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open # For logging or other logic if needed

        # Custom signal lines from SignalData
        self.signal   = self.datas[0].signal
        self.sl_level = self.datas[0].sl_level # SL level predicted by model (if any)
        self.tp_level = self.datas[0].tp_level # TP level predicted by model (if any)
        
        self.order      = None # To keep track of pending orders
        self.entry_price = None # Price of the last entry
        self.sl_order   = None # To keep track of SL order
        self.tp_order   = None # To keep track of TP order
        
        # Example indicators (optional, can be removed if not used by strategy logic)
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=50)
        logger.debug("MorningstarStrategy initialized.")

    def log(self, txt, dt=None, level=logging.INFO):
        """ Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.log(level, f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            price_str = f"{order.price:.5f}" if order.price is not None else "N/A"
            self.log(f"Order {order.getordername()} Submitted/Accepted: Ref: {order.ref}, Size: {order.size}, Price: {price_str}")
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Ref: {order.ref}, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.5f}, Size: {order.executed.size}")
                self.entry_price = order.executed.price
                if self.params.use_sl_tp and not self.sl_order and not self.tp_order: # Place SL/TP only if not already set
                    self.place_sl_tp()
            elif order.issell(): # Sell
                self.log(f"SELL EXECUTED, Ref: {order.ref}, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.5f}, Size: {order.executed.size}")
                # If a sell order is completed, it might be a TP or SL or a manual sell signal
                # Cancel any pending SL/TP orders related to the closed position
                if self.sl_order and self.sl_order.ref == order.ref: # SL hit
                    self.log(f"STOP LOSS HIT by order Ref: {order.ref}")
                    self.sl_order = None
                    if self.tp_order: self.cancel(self.tp_order); self.tp_order = None
                elif self.tp_order and self.tp_order.ref == order.ref: # TP hit
                    self.log(f"TAKE PROFIT HIT by order Ref: {order.ref}")
                    self.tp_order = None
                    if self.sl_order: self.cancel(self.sl_order); self.sl_order = None
                else: # Manual sell signal that closed the position
                    if self.sl_order: self.cancel(self.sl_order); self.sl_order = None
                    if self.tp_order: self.cancel(self.tp_order); self.tp_order = None
            
            self.bar_executed = len(self) # Bar when order was executed

        elif order.status == order.Partial:
            self.log(f"Order PARTIALLY FILLED: Ref: {order.ref}, Size: {order.executed.size}, Price: {order.executed.price:.5f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f"Order Canceled/Margin/Rejected/Expired: Ref: {order.ref}, Status: {order.getstatusname()}")
            # If an SL or TP order is cancelled by us or rejected, reset its tracker
            if self.sl_order and self.sl_order.ref == order.ref: self.sl_order = None
            if self.tp_order and self.tp_order.ref == order.ref: self.tp_order = None
        
        self.order = None # Reset pending order tracker

    def place_sl_tp(self):
        if not self.position: # No position, no SL/TP
            return

        current_price = self.datas[0].close[0] # Fallback if entry_price is None
        entry = self.entry_price if self.entry_price is not None else current_price

        if self.position.size > 0: # Long position
            sl_price = self.sl_level[0] if self.sl_level[0] > 0 else entry * (1 - self.params.default_sl_pct)
            tp_price = self.tp_level[0] if self.tp_level[0] > 0 else entry * (1 + self.params.default_tp_pct)
            
            # Ensure SL is below entry and TP is above entry for long
            sl_price = min(sl_price, entry * 0.999) # At least a tiny bit below
            tp_price = max(tp_price, entry * 1.001) # At least a tiny bit above

            self.sl_order = self.sell(exectype=bt.Order.Stop, price=sl_price, size=self.position.size)
            self.tp_order = self.sell(exectype=bt.Order.Limit, price=tp_price, size=self.position.size)
            self.log(f"Placed SL sell at {sl_price:.5f} and TP sell at {tp_price:.5f} for long position (Entry: {entry:.5f}). Refs: SL={self.sl_order.ref}, TP={self.tp_order.ref}")
        
        # To implement short selling, add elif self.position.size < 0:
        # elif self.position.size < 0: # Short position
        #     sl_price = self.sl_level[0] if self.sl_level[0] > 0 else entry * (1 + self.params.default_sl_pct)
        #     tp_price = self.tp_level[0] if self.tp_level[0] > 0 else entry * (1 - self.params.default_tp_pct)
        #     sl_price = max(sl_price, entry * 1.001)
        #     tp_price = min(tp_price, entry * 0.999)
        #     self.sl_order = self.buy(exectype=bt.Order.Stop, price=sl_price, size=abs(self.position.size))
        #     self.tp_order = self.buy(exectype=bt.Order.Limit, price=tp_price, size=abs(self.position.size))
        #     self.log(f"Placed SL buy (cover) at {sl_price:.5f} and TP buy (cover) at {tp_price:.5f} for short position. Refs: SL={self.sl_order.ref}, TP={self.tp_order.ref}")


    def next(self):
        # self.log(f"Next bar: Open={self.dataopen[0]:.2f}, High={self.datas[0].high[0]:.2f}, Low={self.datas[0].low[0]:.2f}, Close={self.dataclose[0]:.2f}, Volume={self.datas[0].volume[0]:.2f}, Signal={self.signal[0]}")
        if self.order: # Check if an order is pending... if yes, can't send another
            return

        current_price = self.dataclose[0]
        current_signal = self.signal[0]

        if not self.position:  # Not in the market
            if current_signal > 0:  # BUY signal
                cash = self.broker.getcash()
                risk_amount = cash * self.params.risk_per_trade
                size_to_buy = risk_amount / current_price
                
                self.log(f"BUY CREATE: Signal={current_signal:.2f}, Price={current_price:.5f}, Size={size_to_buy:.8f}")
                self.order = self.buy(size=size_to_buy)
            # To implement short selling:
            # elif current_signal < 0: # SELL (short) signal
            #     cash = self.broker.getcash()
            #     risk_amount = cash * self.params.risk_per_trade
            #     size_to_sell = risk_amount / current_price
            #     self.log(f"SHORT SELL CREATE: Signal={current_signal:.2f}, Price={current_price:.5f}, Size={size_to_sell:.8f}")
            #     self.order = self.sell(size=size_to_sell) # This initiates a short position
        
        else:  # Already in the market
            if self.position.size > 0: # Currently long
                if current_signal < 0:  # SELL signal (to close long)
                    self.log(f"CLOSE LONG CREATE: Signal={current_signal:.2f}, Price={current_price:.5f}, Size={self.position.size}")
                    self.order = self.sell(size=self.position.size) # Close position
            # To implement short selling closing:
            # elif self.position.size < 0: # Currently short
            #     if current_signal > 0: # BUY signal (to cover short)
            #         self.log(f"COVER SHORT CREATE: Signal={current_signal:.2f}, Price={current_price:.5f}, Size={abs(self.position.size)}")
            #         self.order = self.buy(size=abs(self.position.size)) # Close position (buy to cover)


class SignalData(bt.feeds.PandasData):
    lines = ("signal", "sl_level", "tp_level") # Add any other custom lines from your df
    params = (
        ("datetime", None), # Expects datetime in index, otherwise specify column index (e.g., 0)
        ("open", -1),       # Auto-detect column by name if -1, or provide column index
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("openinterest", None), # No open interest in crypto spot
        ("signal", -1),     # Custom signal column
        ("sl_level", -1),   # Custom SL level column
        ("tp_level", -1),   # Custom TP level column
    )

# ----------------------------------------
# Helpers: load_data, prepare_features, generate_signals
# ----------------------------------------
def load_data(pair: str, data_dir: Path = None, data_path: Path = None) -> pd.DataFrame | None:
    """
    Loads OHLCV data for a given pair.
    Prioritizes data_path if provided, otherwise searches in data_dir.
    Expects data to have a DatetimeIndex or a 'timestamp' column convertible to DatetimeIndex.
    """
    if data_path:
        p = Path(data_path)
        if p.exists():
            logger.info(f"Loading data directly from specified path: {p}")
            df = pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
        else:
            logger.error(f"Specified data file not found: {data_path}")
            return None # Or raise FileNotFoundError(f"Specified data file not found: {data_path}")
    elif data_dir:
        data_dir_path = Path(data_dir)
        pfn = pair.replace("/", "").lower()
        # Try common naming patterns
        potential_files = [
            data_dir_path / f"{pfn}_data.parquet", data_dir_path / f"{pfn}_data.csv",
            data_dir_path / f"{pfn}.parquet", data_dir_path / f"{pfn}.csv"
        ]
        found_path = None
        for p_try in potential_files:
            if p_try.exists():
                found_path = p_try
                break
        if found_path:
            logger.info(f"Loading data from discovered file: {found_path}")
            df = pd.read_parquet(found_path) if found_path.suffix==".parquet" else pd.read_csv(found_path)
        else:
            logger.error(f"No data file found for {pair} in directory {data_dir_path} with common naming patterns.")
            return None # Or raise FileNotFoundError(f"No data file for {pair} in {data_dir_path}")
    else: # Should be caught by CLI validation, but good to have
        logger.error("Neither data_dir nor data_path was provided to load_data.")
        return None

    # Ensure DatetimeIndex
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        except Exception as e:
            logger.error(f"Error converting 'timestamp' column to DatetimeIndex: {e}", exc_info=True)
            return None
    elif not isinstance(df.index, pd.DatetimeIndex):
        try: # If index is not datetime but might be convertible (e.g. string dates)
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to parse DataFrame index as datetime: {e}", exc_info=True)
            return None
    
    if df.index.tz is not None: # Ensure timezone naive for Backtrader compatibility
        logger.debug(f"Data loaded with timezone {df.index.tz}. Converting to naive UTC for consistency.")
        df.index = df.index.tz_localize(None)

    logger.info(f"Loaded and processed data for {pair} from {data_path or data_dir} ({len(df)} rows)")
    return df


def prepare_features(df: pd.DataFrame, pair: str) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Prepares the feature dictionary required by the model.
    
    Args:
        df: DataFrame with OHLCV data and a DatetimeIndex.
        pair: Trading pair string (e.g., "ADA/USDT").

    Returns:
        A tuple containing:
        - features_dict: Dictionary of features for the model.
        - df_with_features: DataFrame including all original and new features, with DatetimeIndex.
        Returns (None, None) on error.
    """
    try:
        from utils.feature_engineering import apply_feature_pipeline 
        
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        # apply_feature_pipeline expects 'timestamp' as a column
        if isinstance(df.index, pd.DatetimeIndex):
            df_input_for_pipeline = df[essential_cols].copy()
            df_input_for_pipeline.reset_index(inplace=True) # Moves DatetimeIndex to 'timestamp' (or original name)
            # Ensure the timestamp column is named 'timestamp'
            if df_input_for_pipeline.columns[0] != 'timestamp' and 'timestamp' not in df_input_for_pipeline.columns:
                 current_idx_name = df_input_for_pipeline.columns[0]
                 logger.debug(f"Renaming index column from '{current_idx_name}' to 'timestamp' for apply_feature_pipeline.")
                 df_input_for_pipeline.rename(columns={current_idx_name: 'timestamp'}, inplace=True)
        else:
            logger.error("Input DataFrame to prepare_features must have a DatetimeIndex.")
            return None, None

        logger.info("Préparation des features à partir des colonnes OHLCV de base uniquement.")
        logger.debug(f"Shape du DataFrame passé à apply_feature_pipeline: {df_input_for_pipeline.shape}")
        logger.debug(f"Colonnes du DataFrame passé à apply_feature_pipeline: {df_input_for_pipeline.columns.tolist()}")
        
        df_feat = apply_feature_pipeline(df_input_for_pipeline) 
        
        if df_feat is None or df_feat.empty:
            logger.error("apply_feature_pipeline returned None or empty DataFrame.")
            return None, None
        
        logger.debug(f"DataFrame shape après apply_feature_pipeline: {df_feat.shape}")
        logger.debug(f"DataFrame columns après apply_feature_pipeline: {df_feat.columns.tolist()}")

        # Ensure 'timestamp' column from df_feat is set as DatetimeIndex
        if 'timestamp' in df_feat.columns:
            try:
                df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
                df_feat = df_feat.set_index('timestamp')
                logger.debug(f"Index restauré en timestamp à partir de la colonne. Type d'index: {type(df_feat.index)}")
            except Exception as e_idx:
                logger.error(f"Erreur lors de la conversion de la colonne 'timestamp' de df_feat en DatetimeIndex: {e_idx}", exc_info=True)
                return None, None
        elif not isinstance(df_feat.index, pd.DatetimeIndex): # Should not happen if apply_feature_pipeline preserves index or returns timestamp col
            logger.error("df_feat n'a pas d'index Datetime ou de colonne 'timestamp' après apply_feature_pipeline.")
            return None, None

        # Select technical feature columns
        tech_cols = [c for c in df_feat.columns if c.startswith("feature_")]
        if not tech_cols: 
            logger.warning("No columns starting with 'feature_' found after apply_feature_pipeline. Using all numeric columns excluding OHLCV and known non-feature columns.")
            # Define columns that are definitely NOT features
            non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                                'signal', 'sl_level', 'tp_level', # These are added later
                                'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'market_regime', # HMM might be features or meta
                                'pair', 'timeframe', 'timestamp'] # Meta
            tech_cols = [c for c in df_feat.select_dtypes(include=np.number).columns if c not in non_feature_cols]

        if not tech_cols:
            logger.error("No technical feature columns identified after fallback for model input.")
            return None, None

        logger.info(f"Identified {len(tech_cols)} technical feature columns for model input: {tech_cols[:5]}...")
        tech_input_data = df_feat[tech_cols].values.astype(np.float32)
        
        if tech_input_data.ndim != 2:
            logger.error(f"Technical features input data has unexpected ndim: {tech_input_data.ndim}. Expected 2.")
            return None, None
        if tech_input_data.shape[0] != len(df_feat):
             logger.error(f"Length mismatch: technical_input ({tech_input_data.shape[0]}) vs df_feat ({len(df_feat)})")
             return None
        logger.info(f"Shape of technical features input (X_techniques): {tech_input_data.shape}")
        
        num_mcp_features = 3 # Default
        try:
            from config.config import Config as ConfigClassForPrepareFeatures 
            local_cfg_instance = ConfigClassForPrepareFeatures()
            if local_cfg_instance and hasattr(local_cfg_instance, 'get_config') and local_cfg_instance.yaml_config:
                 num_mcp_features = local_cfg_instance.get_config("model.num_mcp_features", 3) 
                 logger.info(f"Nombre de features MCP récupéré depuis la config dans prepare_features: {num_mcp_features}")
            # else: # Warnings already handled by Config class or here
            #      logger.warning(f"Impossible de charger Config ou d'utiliser get_config dans prepare_features. Utilisation de la valeur par défaut pour num_mcp_features: {num_mcp_features}")
        except ImportError:
            logger.warning("Impossible d'importer Config dans prepare_features. Utilisation de la valeur par défaut pour num_mcp_features.")
        except Exception as e_cfg: # Catch any other exception from Config loading
            logger.warning(f"Erreur lors de la récupération de num_mcp_features depuis Config dans prepare_features: {e_cfg}. Utilisation de la valeur par défaut: {num_mcp_features}")

        features_dict = {
            "technical_input": tech_input_data,
            "cryptobert_input": np.zeros((len(df_feat), 768), dtype=np.float32), # Placeholder
            "instrument_input": np.zeros((len(df_feat), 1),   dtype=np.int32),   # Placeholder
            "mcp_input":        np.zeros((len(df_feat), num_mcp_features), dtype=np.float32) # Placeholder
        }
        logger.debug(f"Clés du dictionnaire features passé au modèle: {list(features_dict.keys())}")
        
        # Validation: len(features["technical_input"]) == len(df_feat)
        if len(features_dict["technical_input"]) != len(df_feat):
            logger.error(f"CRITICAL: Length mismatch between technical_input ({len(features_dict['technical_input'])}) and df_feat ({len(df_feat)}) before returning from prepare_features.")
            return None, None
            
        return features_dict, df_feat 
    except Exception as e:
        logger.error(f"prepare_features error: {e}", exc_info=True)
        return None, None

def generate_signals(df_with_features: pd.DataFrame, model, features_dict: dict, threshold: float = DEFAULT_SIGNAL_THRESHOLD) -> pd.DataFrame | None:
    """
    Generates trading signals from model predictions.
    df_with_features must have a DatetimeIndex.
    """
    try:
        preds = model.predict(features_dict)
        logger.debug(f"Raw model prediction output (preds): {preds}")
        logger.debug(f"Type of preds: {type(preds)}")

        sig_pred_array = None
        sltp_array = None

        if isinstance(preds, dict):
            logger.info(f"Model output is a dict. Keys: {list(preds.keys())}")
            if 'market_regime' in preds:
                sig_pred_array = preds['market_regime']
                logger.info(f"Extracted signal predictions from dict key 'market_regime'. Shape: {sig_pred_array.shape if hasattr(sig_pred_array, 'shape') else 'N/A'}")
                sltp_key = next((key for key in preds.keys() if 'sltp' in key.lower() and key != 'market_regime'), None)
                if sltp_key:
                    sltp_array = preds[sltp_key]
                    logger.info(f"Extracted SL/TP from dict key '{sltp_key}'. Shape: {sltp_array.shape if hasattr(sltp_array, 'shape') else 'N/A'}")
            elif len(preds.keys()) == 1: 
                single_key = list(preds.keys())[0]
                logger.info(f"Model output dict has a single key: '{single_key}'. Assuming it contains signal predictions.")
                sig_pred_array = preds[single_key]
            else: 
                logger.warning("'market_regime' key not found and multiple keys exist. Attempting to find signal output using common patterns.")
                signal_output_key = next((key for key in preds.keys() if 'signal' in key.lower() or key.startswith('output_0') or key.startswith('output_1')), None)
                if signal_output_key:
                    sig_pred_array = preds[signal_output_key]
                # SL/TP extraction would also need a fallback here if signal_output_key is found
        elif isinstance(preds, list): 
            logger.info(f"Model output is a list of {len(preds)} elements.")
            if len(preds) > 0: sig_pred_array = preds[0]
            if len(preds) > 1: sltp_array = preds[1]
        elif hasattr(preds, 'shape'): 
            sig_pred_array = preds
        else:
            logger.error(f"Model output format is unrecognized: {type(preds)}")
            return None

        if sig_pred_array is None or not hasattr(sig_pred_array, 'shape'):
            logger.error("Signal predictions (sig_pred_array) could not be properly extracted or are not a valid array.")
            return None
        if sig_pred_array.shape[0] != len(df_with_features):
             logger.error(f"Mismatch in length: predictions ({sig_pred_array.shape[0]}) vs input data ({len(df_with_features)}).")
             return None

        if sig_pred_array.ndim == 1: 
            cls = sig_pred_array.astype(int)
            prob = np.ones_like(cls, dtype=float)
        elif sig_pred_array.ndim == 2 and sig_pred_array.shape[1] >= 2: 
            cls = np.argmax(sig_pred_array, axis=1)
            prob = np.max(sig_pred_array, axis=1)
        else:
            logger.error(f"Unexpected shape for signal predictions: {sig_pred_array.shape}")
            return None
            
        mapper = {0: -1, 1: 0, 2: 1}  # SELL:0, NEUTRAL:1, BUY:2
        sigs = np.array([mapper.get(c, 0) for c in cls]) 
        sigs = np.where(prob >= threshold, sigs, 0) 

        df_out = df_with_features.copy() # df_with_features should have DatetimeIndex
        df_out["signal"] = sigs

        # SL/TP processing
        if sltp_array is None and sig_pred_array.ndim == 2 and sig_pred_array.shape[1] > 3: # Example: 3 classes for signal + 2 for SL/TP
            num_signal_classes = 3 # Define based on your model's signal output structure
            if sig_pred_array.shape[1] >= num_signal_classes + 2:
                logger.info(f"Attempting to extract SL/TP from additional columns of the primary signal output array.")
                sltp_array = sig_pred_array[:, num_signal_classes:num_signal_classes+2]

        if sltp_array is not None and hasattr(sltp_array, 'ndim') and sltp_array.ndim == 2 and sltp_array.shape[0] == len(df_out):
            if sltp_array.shape[1] >= 2:
                df_out["sl_level"] = sltp_array[:, 0]
                df_out["tp_level"] = sltp_array[:, 1]
            elif sltp_array.shape[1] == 1: 
                df_out["sl_level"] = sltp_array[:, 0]; df_out["tp_level"] = 0.0
        else: 
            df_out["sl_level"] = 0.0; df_out["tp_level"] = 0.0
        
        logger.info(f"Signal distribution: {df_out['signal'].value_counts(normalize=True).mul(100).round(2).to_dict()}")
        return df_out
        
    except Exception as e:
        logger.error(f"generate_signals error: {e}", exc_info=True)
        return None

# ----------------------------------------
# Backtest runner & save
# ----------------------------------------
def run_backtest(df_signals: pd.DataFrame, pair: str,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 commission_fee: float = DEFAULT_COMMISSION_FEE, # Renamed for consistency
                 slippage_perc: float = DEFAULT_SLIPPAGE_PERC): 
    try:
        df_bt = df_signals.copy() 

        if not isinstance(df_bt.index, pd.DatetimeIndex):
            # This case should ideally be handled before, df_signals should have DatetimeIndex
            logger.error("DataFrame for backtesting (df_signals) must have a DatetimeIndex.")
            return None, None, None, None 
        
        if df_bt.index.tz is not None: # Ensure timezone naive
            logger.debug(f"DataFrame index has timezone {df_bt.index.tz}. Converting to naive.")
            df_bt.index = df_bt.index.tz_localize(None)

        logger.debug(f"DataFrame for Backtrader (index type: {type(df_bt.index)}, sample: \n{df_bt.index[:3]})")
        logger.debug(f"Columns in df_bt for SignalData: {df_bt.columns.tolist()}")

        cerebro = bt.Cerebro()
        data_feed = SignalData(dataname=df_bt) 
        cerebro.adddata(data_feed)
        cerebro.addstrategy(MorningstarStrategy)
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=commission_fee) # Use consistent name
        
        if slippage_perc > 0:
            cerebro.broker.set_slippage_perc(perc=slippage_perc)
            logger.info(f"Applied slippage of {slippage_perc*100:.3f}%")

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days) # Removed annualization argument
        cerebro.addanalyzer(bt.analyzers.DrawDown,    _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer,_name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns,      _name="returns")
        cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn") 
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio') # For equity curve and more

        logger.info(f"Running backtest for {pair}...")
        results_run = cerebro.run()
        strat = results_run[0] # Get the strategy instance
        
        # --- Extract Metrics ---
        final_val = cerebro.broker.getvalue()
        pnl  = final_val - initial_capital
        roi  = (pnl / initial_capital) * 100 if initial_capital else 0
        
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0) if sharpe_analysis else 0.0
        
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown_analysis.get("max",{}).get("drawdown",0.0) if drawdown_analysis else 0.0
        
        trades_analysis  = strat.analyzers.trades.get_analysis()
        total_trades = trades_analysis.get("total",{}).get("total",0)
        won_trades   = trades_analysis.get("won",{}).get("total",0)
        lost_trades  = trades_analysis.get("lost",{}).get("total",0)
        win_rate    = (won_trades / total_trades) * 100 if total_trades > 0 else 0.0

        sqn_analysis = strat.analyzers.sqn.get_analysis()
        sqn_value = sqn_analysis.get("sqn", 0.0) if sqn_analysis else 0.0

        metrics = {
            "pair": pair, "initial_capital": initial_capital, "final_value": float(final_val),
            "pnl": float(pnl), "roi_pct": float(roi), 
            "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else 0.0,
            "sqn": float(sqn_value) if sqn_value is not None else 0.0,
            "max_drawdown_pct": float(max_drawdown) if max_drawdown is not None else 0.0,
            "total_trades": int(total_trades), "won_trades": int(won_trades),
            "lost_trades": int(lost_trades), "win_rate_pct": float(win_rate),
            "timestamp": datetime.now().isoformat(),
        }

        # --- Extract Equity Curve & Trades List ---
        pyfolio_analyzer = strat.analyzers.getbyname('pyfolio')
        returns, _, transactions, _ = pyfolio_analyzer.get_pf_items() # positions, gross_lev not used for now
        
        df_equity_curve = pd.DataFrame()
        if returns is not None and not returns.empty:
            equity_curve_values = (1 + returns).cumprod() * initial_capital
            df_equity_curve = equity_curve_values.to_frame(name='equity')
        else:
            logger.warning("PyFolio analyzer did not produce returns data for equity curve.")

        df_trades = pd.DataFrame()
        if transactions is not None and not transactions.empty:
            # Transactions from PyFolio are detailed but need transformation to a simple trade list
            # For a simpler trade list, TradeAnalyzer is often better if its output is parsed.
            # The current loop for TradeAnalyzer is more direct for a simple list.
            trades_list_from_analyzer = []
            for i in range(len(trades_analysis.get('len', []))): 
                trade_info = {
                    'ref': trades_analysis['ref'][i] if 'ref' in trades_analysis and i < len(trades_analysis['ref']) else None,
                    'status': trades_analysis['status'][i] if 'status' in trades_analysis and i < len(trades_analysis['status']) else None, 
                    'type': 'Long' if trades_analysis['buysell'][i] == 1 else 'Short', 
                    'opendt': str(bt.num2date(trades_analysis['dtopen'][i])) if 'dtopen' in trades_analysis and i < len(trades_analysis['dtopen']) else None,
                    'openprice': trades_analysis['price'][i] if 'price' in trades_analysis and i < len(trades_analysis['price']) else None,
                    'closedt': str(bt.num2date(trades_analysis['dtclose'][i])) if 'dtclose' in trades_analysis and i < len(trades_analysis['dtclose']) else None,
                    'closeprice': (trades_analysis['pnl'][i]/trades_analysis['size'][i] + trades_analysis['price'][i]) if trades_analysis.get('pnl') is not None and trades_analysis.get('size') is not None and i < len(trades_analysis['pnl']) and trades_analysis['size'][i] !=0 else None, 
                    'size': trades_analysis['size'][i] if 'size' in trades_analysis and i < len(trades_analysis['size']) else None,
                    'pnl': trades_analysis['pnl'][i] if 'pnl' in trades_analysis and i < len(trades_analysis['pnl']) else None,
                    'pnlnet': trades_analysis['pnlnet'][i] if 'pnlnet' in trades_analysis and i < len(trades_analysis['pnlnet']) else None,
                    'commission': trades_analysis['commission'][i] if 'commission' in trades_analysis and i < len(trades_analysis['commission']) else None,
                }
                trades_list_from_analyzer.append(trade_info)
            df_trades = pd.DataFrame(trades_list_from_analyzer)

        return metrics, cerebro, df_equity_curve, df_trades 
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return None, None, None, None

def save_results(results_metrics: dict, cerebro_obj, 
                 df_equity: pd.DataFrame, df_trades: pd.DataFrame, 
                 pair: str, results_dir: Path, plot_equity: bool = False):
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = pair.replace("/","").lower()
        
        fn_metrics = results_dir / f"{name_prefix}_backtest_metrics_{ts}.json"
        with open(fn_metrics,"w") as f:
            json.dump(results_metrics, f, indent=2)
        logger.info(f"Metrics saved to {fn_metrics}")

        if df_equity is not None and not df_equity.empty:
            fn_equity_csv = results_dir / f"{name_prefix}_equity_curve_{ts}.csv"
            df_equity.to_csv(fn_equity_csv)
            logger.info(f"Equity curve saved to {fn_equity_csv}")

        if df_trades is not None and not df_trades.empty:
            fn_trades_csv = results_dir / f"{name_prefix}_trades_list_{ts}.csv"
            df_trades.to_csv(fn_trades_csv, index=False)
            logger.info(f"Trades list saved to {fn_trades_csv}")

        if plot_equity and cerebro_obj is not None:
            fn_plot = results_dir / f"{name_prefix}_equity_plot_{ts}.png"
            try:
                # cerebro.plot() returns a list of figures. We'll take the first one.
                # Ensure matplotlib backend is suitable for non-interactive saving
                plt.ioff() 
                fig = cerebro_obj.plot(iplot=False, style='candlestick', barup='green', bardown='red')[0][0]
                fig.savefig(fn_plot, dpi=300)
                logger.info(f"Equity plot saved to {fn_plot}")
                plt.close(fig) # Close the figure to free memory
            except Exception as e_plot:
                logger.error(f"Failed to save plot: {e_plot}", exc_info=True)
    except Exception as e_save:
        logger.error(f"save_results error: {e_save}", exc_info=True)

# ----------------------------------------
# Main CLI
# ----------------------------------------
def main():
    epilog_text = f"""
Examples:
  python %(prog)s --pair ADA/USDT --data-path path/to/your/ADAusdt_data.parquet --model path/to/your/model.keras --plot
  python %(prog)s --pair ADA/USDT --data-dir path/to/data_directory --model path/to/model.keras --loglevel DEBUG
"""
    parser = argparse.ArgumentParser(
        description="Backtest Morningstar model.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pair", type=str, required=True, help="Trading pair, e.g., ADA/USDT")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_FILENAME, help=f"Path to the trained Keras model file (default: {DEFAULT_MODEL_FILENAME})")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=str, help=f"Directory containing data files (e.g., {DEFAULT_DATA_DIR_NAME}/). Script will search for <pair>_data.parquet or .csv.")
    group.add_argument("--data-path", type=str, help="Direct path to the Parquet/CSV data file.")
    
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR_NAME, help=f"Directory to save backtest results (default: {DEFAULT_RESULTS_DIR_NAME})")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help=f"Initial capital for the backtest (default: {DEFAULT_INITIAL_CAPITAL})")
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_FEE, help=f"Commission fee percentage (e.g., 0.001 for 0.1%) (default: {DEFAULT_COMMISSION_FEE})")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE_PERC, help=f"Slippage percentage (e.g., 0.0005 for 0.05%) (default: {DEFAULT_SLIPPAGE_PERC})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIGNAL_THRESHOLD, help=f"Signal probability threshold (0.0 to 1.0) for taking trades (default: {DEFAULT_SIGNAL_THRESHOLD})")
    parser.add_argument("--plot", action="store_true", help="Generate and save equity curve plot.")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level (default: INFO).")
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be between 0.0 and 1.0.", file=sys.stderr)
        sys.exit(10) # Custom exit code for invalid argument
    if args.commission < 0:
        print("Error: --commission cannot be negative.", file=sys.stderr)
        sys.exit(10)
    if args.slippage < 0:
        print("Error: --slippage cannot be negative.", file=sys.stderr)
        sys.exit(10)
    if args.initial_capital <= 0:
        print("Error: --initial-capital must be positive.", file=sys.stderr)
        sys.exit(10)

    # --- Setup Logging (centralized) ---
    # Ensure logs directory exists
    log_dir_path = Path("logs/backtest") # Central log directory
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_path / f"backtest_{args.pair.replace('/','').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Remove existing handlers if any were set up by other modules or previous basicConfig calls
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig( # This sets up the root logger
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.handlers.RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5), 
            logging.StreamHandler(sys.stdout) # Also log to console
        ],
    )
    tf.get_logger().setLevel("ERROR") # Keep TensorFlow's own logger quiet unless it's an ERROR

    logger.info(f"Starting backtest for {args.pair} with arguments: {vars(args)}")

    # Determine data source
    data_file_path = Path(args.data_path) if args.data_path else None
    data_directory = Path(args.data_dir) if args.data_dir else None # data_dir is relative to CWD or absolute

    # --- Main Execution Block ---
    try:
        logger.info(f"Loading model from {args.model}...")
        model_path_obj = Path(args.model)
        if not model_path_obj.exists():
            logger.error(f"Model file not found: {args.model}")
            return 2 # Specific exit code for model file error
        model = tf.keras.models.load_model(model_path_obj)
        
        df_ohlcv = load_data(args.pair, data_dir=data_directory, data_path=data_file_path)
        if df_ohlcv is None: 
            logger.error(f"Failed to load data for {args.pair}.")
            return 3 # Data loading error

        features_dict, df_with_all_cols = prepare_features(df_ohlcv, args.pair)
        if features_dict is None or df_with_all_cols is None: 
            logger.error(f"Failed to prepare features for {args.pair}.")
            return 4 # Feature preparation error
        
        df_signals = generate_signals(df_with_all_cols, model, features_dict, args.threshold)
        if df_signals is None: 
            logger.error(f"Failed to generate signals for {args.pair}.")
            return 5 # Signal generation error

        metrics, cerebro_obj, df_equity, df_trades = run_backtest(
            df_signals, args.pair,
            initial_capital=args.initial_capital,
            commission_fee=args.commission, # Pass validated commission
            slippage_perc=args.slippage    # Pass validated slippage
        )
        if metrics is None: 
            logger.error(f"Backtest execution failed for {args.pair}.")
            return 6 # Backtest run error
        
        results_output_dir = Path(args.results_dir)
        save_results(metrics, cerebro_obj, df_equity, df_trades, args.pair, results_output_dir, plot_equity=args.plot)
        logger.info(f"Backtest for {args.pair} completed successfully. Results in {results_output_dir.resolve()}")
        return 0 # Success
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        return 3 # Consistent with data loading error
    except Exception as e:
        logger.error(f"Main error during backtest execution: {e}", exc_info=True)
        return 1 # Generic error

if __name__=="__main__":
    # Ensure the script can find modules in 'ultimate' if run from project root
    # This is more robust if the script is in a subdirectory like 'ultimate'
    # and utils/config are relative to the project root.
    # current_script_path = Path(__file__).resolve()
    # project_root = current_script_path.parent.parent # Assuming script is in ultimate/
    # if str(project_root) not in sys.path:
    #    sys.path.insert(0, str(project_root))
    #    logger.debug(f"Added {project_root} to sys.path")
        
    sys.exit(main())
