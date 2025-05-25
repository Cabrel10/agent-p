import pandas as pd
import numpy as np
import os

# Créer des données de base pour 60 lignes
N_ROWS = 60
base_time = pd.to_datetime("2023-01-01 00:00:00")
timestamps = [base_time + pd.Timedelta(minutes=i) for i in range(N_ROWS)]

np.random.seed(42)  # Pour la reproductibilité
data = {
    "timestamp": timestamps,
    "open": 100 + np.random.randn(N_ROWS).cumsum() * 0.1,
    "high": 0,  # Sera calculé après
    "low": 0,  # Sera calculé après
    "close": 0,  # Sera calculé après
    "volume": np.random.randint(500, 2000, size=N_ROWS),
    "symbol": ["BTC/USDT"] * N_ROWS,
}
df = pd.DataFrame(data)

# Générer close, puis high et low de manière réaliste
df["close"] = df["open"] + np.random.randn(N_ROWS) * 0.5
df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(N_ROWS) * 0.2
df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(N_ROWS) * 0.2
# S'assurer que low <= open/close et high >= open/close
df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))


# Ajouter des features techniques simples (placeholders, car apply_feature_pipeline les recalculera)
# Ces colonnes sont ajoutées ici principalement pour que le fichier golden ait une structure
# similaire à ce que le modèle pourrait attendre si les features étaient pré-calculées.
# Pour le test E2E de run_backtest.py, apply_feature_pipeline sera appelé.
df["feature_SMA_10"] = df["close"].rolling(window=min(10, N_ROWS)).mean().fillna(method="bfill").fillna(method="ffill")
df["feature_RSI_14"] = np.random.uniform(30, 70, size=N_ROWS)
df["feature_MACD_12_26_9"] = np.random.uniform(-0.5, 0.5, size=N_ROWS)
df["feature_MACD_signal_12_26_9"] = np.random.uniform(-0.3, 0.3, size=N_ROWS)
df["feature_MACD_hist_12_26_9"] = df["feature_MACD_12_26_9"] - df["feature_MACD_signal_12_26_9"]

# Ajouter des features LLM et MCP placeholder
# Ces colonnes ne sont pas utilisées par le modèle factice mais pourraient être attendues par certains pipelines
num_llm_features = 10  # Réduit pour la fixture, 768 est beaucoup pour un fichier de test
for i in range(num_llm_features):
    df[f"bert_{i}"] = np.random.rand(N_ROWS) * 0.01

num_mcp_features = 5  # Réduit pour la fixture
for i in range(num_mcp_features):
    df[f"mcp_{i}"] = np.random.rand(N_ROWS) * 0.01

# Créer le répertoire fixtures s'il n'existe pas
output_dir = "tests/fixtures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "golden_backtest.parquet")

# Sauvegarder en Parquet
df.to_parquet(output_path, index=False)

print(f"Fichier '{output_path}' créé avec {len(df)} lignes et {len(df.columns)} colonnes.")
print(f"Colonnes : {df.columns.tolist()[:10]}...")  # Afficher les 10 premières colonnes
print("\nAperçu des données (premières 3 lignes) :")
print(df.head(3))
