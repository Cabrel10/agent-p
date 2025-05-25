# Rapport Sprint 2 – Pré-entraînement Contrastif MonolithModel

## 1. Résultats de l’optimisation Optuna

- **Meilleure value (perte contrastive)** : 0.006662264000624418
- **Meilleurs hyperparamètres** :
    - `learning_rate` : 0.0015562385502385044
    - `dropout_rate` : 0.10263639892545441
    - `l2_reg` : 7.224308701685084e-05
    - `dense_units` : 32
    - `lstm_units` : 64
    - `transformer_blocks` : 3
    - `transformer_heads` : 8
    - `transformer_ff_dim_factor` : 2
    - `batch_size` : 64
    - `use_batch_norm` : True
    - `latent_dim` : 80
    - `contrastive_temperature` : 0.055566784132156696
    - `augmentation_type` : scaling

## 2. Amélioration de MonolithModel.train

La méthode `train` de `MonolithModel` a été améliorée pour :
- Calculer la perte contrastive sur l’ensemble de validation à chaque epoch si `validation_data` est fourni.
- Logger la perte de validation (`val_loss`) à chaque epoch.
- Retourner un historique contenant `loss` (train) et `val_loss` (validation).
- Permettre un suivi précis de la généralisation contrastive.

## 3. Script d’entraînement final contrastif

Le script [`train_final_contrastive_model.py`](../../train_final_contrastive_model.py) :
- Utilise la configuration Optuna optimale (voir section 1).
- Prépare les données et instancie le modèle avec les bons hyperparamètres.
- Entraîne le modèle en mode contrastif (`contrastive_training=True`), avec validation.
- Applique l’augmentation contrastive optimale.
- Sauvegarde tous les artefacts nécessaires.

## 4. Artefacts sauvegardés

Dans `models/sprint2_contrastive_encoder/` :
- `contrastive_encoder_model.keras` : Modèle Keras entraîné
- `model_config.json` : Configuration complète du modèle (incluant params contrastifs)
- `data_processing_metadata.json` : Métadonnées de préparation des données
- `scalers.pkl` : Scalers utilisés pour les features
- `training_history.csv` : Historique des pertes d’entraînement et validation

## 5. Architecture finale et hyperparamètres

- **Architecture** : MonolithModel autoencodeur avec tête de projection contrastive
- **Hyperparamètres principaux** :
    - Voir `model_config.json` pour le détail complet
    - Extraits clés :
        - `contrastive_temperature` : 0.055566784132156696
        - `augmentation_type` : scaling
        - `latent_dim` : 80
        - `dense_units` : 32
        - `lstm_units` : 64
        - `transformer_blocks` : 3
        - `transformer_heads` : 8
        - `transformer_ff_dim_factor` : 2
        - `batch_size` : 64
        - `use_batch_norm` : True
        - `learning_rate` : 0.0015562385502385044
        - `dropout_rate` : 0.10263639892545441
        - `l2_reg` : 7.224308701685084e-05

## 6. Résultats finaux

- **Perte contrastive finale (entraînement)** : 0.007241890765726566
- **Perte contrastive finale (validation)** : 0.004144292324781418

---

### Analyse des résultats

Les pertes contrastives finales sont exceptionnellement basses, indiquant que l’encodeur apprend des représentations très discriminantes et généralisables.  
La perte de validation (0.0041) est même inférieure à la perte d’entraînement (0.0072), ce qui reflète une excellente généralisation. Ce phénomène peut survenir lorsque l’option `restore_best_weights=True` est utilisée, permettant de restaurer les poids du modèle correspondant à la meilleure performance sur l’ensemble de validation, même si la convergence finale sur l’entraînement continue légèrement après ce point.

L’augmentation de type `scaling` et la température contrastive basse (0.055) ont permis d’obtenir une séparation nette des représentations.  
L’architecture compacte mais profonde (3 blocs Transformer, 8 têtes, latent_dim=80, batch_norm activé) a favorisé la capacité de discrimination du modèle sur les paires positives/négatives.

Tous les artefacts attendus ont bien été générés et sauvegardés dans `models/sprint2_contrastive_encoder/` :
- `contrastive_encoder_model.keras`
- `contrastive_encoder_model_config.json`
- `model_config.json`
- `data_processing_metadata.json`
- `scalers.pkl`
- `training_history.csv`

Ces résultats valident pleinement la stratégie de pré-entraînement contrastif retenue pour ce sprint.
