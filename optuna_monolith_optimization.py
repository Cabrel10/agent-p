import os
# Pour forcer TensorFlow à n'utiliser qu'un thread par essai Optuna (pour un vrai parallélisme)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import optuna
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from monolith_implementation.monolith_model import MonolithModel
from monolith_implementation.contrastive_utils import jitter, scaling, time_masking

# Définition des colonnes selon l'exemple fourni
technical_cols = [
    # Indicateurs classiques et techniques
    'open', 'high', 'low', 'close', 'volume',
    'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'RSI', 'MACD', 'MACDs', 'MACDh',
    'BBU', 'BBM', 'BBL', 'ATR', 'STOCHk', 'STOCHd', 'ADX', 'CCI', 'Momentum', 'ROC',
    'Williams_%R', 'TRIX', 'Ultimate_Osc', 'DPO', 'OBV', 'VWMA', 'CMF', 'MFI', 'Parabolic_SAR',
    'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SenkouA', 'Ichimoku_SenkouB', 'Ichimoku_Chikou',
    'KAMA', 'VWAP', 'STOCHRSIk', 'CMO', 'PPO', 'FISHERt'
]
embedding_cols = ['llm_embedding']
mcp_cols = [f'mcp_feature_{i:03d}' for i in range(128)]  # 128 features MCP
instrument_col = 'symbol'
target_signal_col = 'hmm_regime'
target_sl_tp_cols = ['level_sl', 'level_tp']
datetime_col = 'timestamp'

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True)
    dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
    lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
    transformer_blocks = trial.suggest_int("transformer_blocks", 1, 3)
    transformer_heads = trial.suggest_categorical("transformer_heads", [2, 4, 8])
    transformer_ff_dim_factor = trial.suggest_int("transformer_ff_dim_factor", 2, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    latent_dim = trial.suggest_int("latent_dim", 16, 128, step=16)

    # Utiliser multi_crypto_dataset.parquet et splitter en train/val
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "ultimate/data/processed/multi_crypto_dataset.parquet")
    df = pd.read_parquet(data_path)
    df = df.sort_values(datetime_col if datetime_col in df.columns else df.columns[0])
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    metadata_for_trial = {}
    metadata_for_trial['technical_cols'] = technical_cols
    metadata_for_trial['embedding_cols'] = embedding_cols
    metadata_for_trial['mcp_cols'] = mcp_cols
    metadata_for_trial['instrument_col'] = instrument_col
    metadata_for_trial['target_signal_col'] = target_signal_col
    metadata_for_trial['target_sl_tp_cols'] = target_sl_tp_cols
    metadata_for_trial['datetime_col'] = datetime_col

    tech_scaler = StandardScaler()
    df_train[technical_cols] = tech_scaler.fit_transform(df_train[technical_cols])
    df_val[technical_cols] = tech_scaler.transform(df_val[technical_cols])
    metadata_for_trial['scalers'] = {'technical': tech_scaler}

    if mcp_cols:
        mcp_scaler = StandardScaler()
        df_train[mcp_cols] = mcp_scaler.fit_transform(df_train[mcp_cols])
        df_val[mcp_cols] = mcp_scaler.transform(df_val[mcp_cols])
        metadata_for_trial['scalers']['mcp'] = mcp_scaler

    instrument_map = {instrument: i for i, instrument in enumerate(df_train[instrument_col].unique())}
    metadata_for_trial['instrument_map'] = instrument_map
    metadata_for_trial['num_instruments'] = len(instrument_map)

    emb_sample = df_train[embedding_cols[0]].iloc[0]
    if isinstance(emb_sample, str):
        emb_dim = len(json.loads(emb_sample))
    else:
        emb_dim = len(emb_sample)
    metadata_for_trial['embeddings_input_dim'] = emb_dim
    metadata_for_trial['mcp_input_dim'] = len(mcp_cols)
    metadata_for_trial['technical_input_dim'] = len(technical_cols)

    def prepare_X_y(df):
        X = {}
        X['technical_input'] = df[technical_cols].values.astype(np.float32)
        X['embeddings_input'] = np.stack(df[embedding_cols[0]].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else np.array(x)).values).astype(np.float32)
        X['mcp_input'] = df[mcp_cols].values.astype(np.float32)
        X['instrument_input'] = df[instrument_col].map(instrument_map).values.astype(np.int32)
        y = {}
        # Pour l'auto-encodeur, la cible est la reconstruction des features techniques
        y['reconstruction_output'] = X['technical_input']
        return X, y

    X_train_dict, y_train_dict = prepare_X_y(df_train)
    X_val_dict, y_val_dict = prepare_X_y(df_val)

    model_config = {
        "model_type": "monolith",
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "l2_reg": l2_reg,
        "dense_units": dense_units,
        "lstm_units": lstm_units,
        "transformer_blocks": transformer_blocks,
        "transformer_heads": transformer_heads,
        "transformer_ff_dim_factor": transformer_ff_dim_factor,
        "use_batch_norm": use_batch_norm,
        "latent_dim": latent_dim,
        "tech_input_shape": (len(technical_cols),),
        "input_dims": {
            "technical": len(technical_cols),
            "embeddings": metadata_for_trial['embeddings_input_dim'],
            "mcp": len(mcp_cols),
            "instrument": len(instrument_map)
        },
        "reconstruction_target_dim": len(technical_cols)
    }

    monolith_model = MonolithModel(config=model_config)

    # Hyperparamètres contrastifs à optimiser
    contrastive_temperature = trial.suggest_float("contrastive_temperature", 0.05, 0.5, log=True)
    augmentation_type = trial.suggest_categorical("augmentation_type", ["jitter", "scaling", "time_masking"])

    # Mapping augmentation_type -> fonction
    augmentation_fn = {
        "jitter": jitter,
        "scaling": scaling,
        "time_masking": time_masking
    }[augmentation_type]

    # Entraînement contrastif
    history = monolith_model.train(
        X_train_dict,
        outputs=None,
        validation_data=None,  # Pour l’instant, pas de validation contrastive
        epochs=10,  # Réduit pour l’optimisation rapide
        batch_size=batch_size,
        contrastive_training=True,
        contrastive_augmentation_fn=augmentation_fn,
        contrastive_temperature=contrastive_temperature
    )

    # Retourne la meilleure perte d'entraînement contrastive
    return min(history.history['loss'])

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    # Lancer plusieurs essais en parallèle (ici 6 jobs, pour 6 cœurs à 80%+)
    study.optimize(objective, n_trials=50, n_jobs=6)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
