# Rapport de Fin de Sprint 1 — Auto-encodeur MonolithModel

## Objectif du Sprint 1

Pré-entraîner un auto-encodeur profond (MonolithModel) sur l’ensemble des features techniques, embeddings LLM et MCP, afin d’obtenir une représentation latente informative pour les tâches ultérieures (apprentissage contrastif, régimes de marché, etc.).

## Architecture du Modèle

- **MonolithModel** : auto-encodeur multi-branches
  - Entrées : 
    - 43 indicateurs techniques
    - Embeddings LLM (par ex. 768 dim)
    - MCP features (128 dim)
    - Instrument (embedding)
  - Encoder profond (Dense, LSTM, Transformer blocks)
  - Couche latente : `latent_dim=112`
  - Decoder symétrique pour la reconstruction des features techniques

## Hyperparamètres retenus (Optuna)

```json
{
  "learning_rate": 0.00039497810376922887,
  "dropout_rate": 0.1173889599996675,
  "l2_reg": 1.866186463803556e-06,
  "dense_units": 64,
  "lstm_units": 128,
  "transformer_blocks": 2,
  "transformer_heads": 2,
  "transformer_ff_dim_factor": 2,
  "batch_size": 32,
  "use_batch_norm": false,
  "latent_dim": 112
}
```

## Performances de Reconstruction

- **Validation MSE** : 0.0045
- **Validation MAE** : 0.0322

Ces faibles valeurs de perte indiquent que le modèle capture efficacement l’information essentielle des features techniques et parvient à les reconstruire fidèlement. La représentation latente (`latent_dim=112`) est donc jugée informative et adaptée pour les prochaines étapes.

## Artefacts sauvegardés

Tous les artefacts du sprint sont disponibles dans :
```
/home/morningstar/Desktop/crypto_robot/Morningstar/models/sprint1_autoencoder/
```
- Modèle entraîné (`autoencoder_monolith_model.keras`)
- Configuration (`model_config.json`)
- Métadonnées de traitement (`data_processing_metadata.json`)
- Scalers (`scalers.pkl`)
- Historique d’entraînement (`training_history.csv`)

## (Optionnel) Illustration qualitative

Pour illustrer la qualité de reconstruction, il est possible de charger le modèle et les scalers, puis de comparer quelques échantillons de l’ensemble de validation :

```python
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Charger artefacts
model = load_model("autoencoder_monolith_model.keras")
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
tech_scaler = scalers["tech_scaler"]

# Charger X_val_dict (cf. script d’entraînement)
# X_val_dict = ...

# Reconstruction
recon = model.predict(X_val_dict["technical_input"])
orig = tech_scaler.inverse_transform(X_val_dict["technical_input"])
recon_inv = tech_scaler.inverse_transform(recon)

# Comparaison sur quelques colonnes
df_compare = pd.DataFrame({
    "original_close": orig[:, 0],
    "reconstructed_close": recon_inv[:, 0]
})
print(df_compare.head())
```

---

# Préparation Sprint 2 (Apprentissage contrastif & neuro-évolution)

- Définir des stratégies d’augmentation pour générer des paires positives/negatives (bruit, masquage, décalage temporel…)
- Adapter MonolithModel : ajouter une tête de projection pour l’apprentissage contrastif
- Implémenter une perte contrastive (ex : InfoNCE)
- Planifier l’intégration de la neuro-évolution pour l’optimisation ultérieure

---

**Sprint 1 terminé avec succès. Prêt pour le kick-off du Sprint 2.**
