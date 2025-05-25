import pandas as pd, numpy as np, gc
from pathlib import Path

DATA_DIR = Path("/…/market")
OUTPUT = DATA_DIR / "dataset_enriched.parquet"
parquet_files = list(DATA_DIR.glob("*.parquet"))

# On traitera chaque paire indépendamment
all_dfs = []
for f in parquet_files:
    print(f"–– Traitement de {f.name} ––")
    df = pd.read_parquet(f, columns=[
        'timestamp','open','high','low','close','volume','pair','timeframe','symbol'
    ])
    # Découpe en sous‑chunks temporels de 1 mois
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for start in pd.date_range(df['timestamp'].min(),
                               df['timestamp'].max(),
                               freq='MS'):  # Monthly start
        end = start + pd.DateOffset(months=1)
        chunk = df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()
        if chunk.empty: 
            continue

        # 1) Features
        from ultimate.utils.feature_engineering import apply_feature_pipeline
        chunk = apply_feature_pipeline(chunk, include_llm=True)

        # 2) MCP
        try:
            from ultimate.utils.mcp_integration import compute_mcp_features
            chunk = compute_mcp_features(chunk)
        except:
            pass

        # 3) HMM
        try:
            from ultimate.data_processors.hmm_regime_detector import HMMRegimeDetector
            hmm = HMMRegimeDetector()
            r, p = hmm.detect_regimes(chunk['close'].pct_change().fillna(0).values)
            chunk['hmm_regime'] = r
            for i in range(p.shape[1]):
                chunk[f'hmm_prob_{i}'] = p[:, i]
            chunk['market_regime'] = chunk['hmm_regime']
        except:
            pass

        all_dfs.append(chunk)
        del chunk
        gc.collect()

# Fusion finale
df_final = pd.concat(all_dfs, ignore_index=True)
df_final = df_final.sort_values(['pair','timestamp']).reset_index(drop=True)
df_final.to_parquet(OUTPUT, index=False)
print("✅ Enrichissement terminé :", OUTPUT)

