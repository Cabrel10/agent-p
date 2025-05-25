# Étape 1: Utiliser une image Python officielle comme image de base
FROM python:3.11-slim AS base

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Répertoire de travail système pour l'installation des paquets
WORKDIR /wheels

# Copier uniquement les fichiers de dépendances de production
COPY requirements.txt .

# Installer les dépendances de production dans un répertoire de wheels
# Cela peut aider à garder la couche finale plus propre si on utilise des builds multi-stages plus complexes,
# mais pour un Dockerfile simple, une installation directe est aussi ok.
# Pour l'instant, on installe directement dans l'environnement global de l'image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Étape 2: Créer l'image finale de production
FROM python:3.11-slim AS final

# Récupérer les variables d'environnement de l'étape de base
ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}

# Créer un utilisateur et un groupe non-root pour l'application
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup -s /bin/bash appuser

# Créer le répertoire de l'application et donner les permissions
RUN mkdir -p /home/appuser/app && chown -R appuser:appgroup /home/appuser/app

# Définir le répertoire de travail
WORKDIR /home/appuser/app

# Copier le code source de l'application nécessaire à l'exécution
# .dockerignore s'assurera que seuls les fichiers nécessaires sont copiés
COPY --chown=appuser:appgroup . .
# Si vous préférez copier des répertoires spécifiques :
# COPY --chown=appuser:appgroup telegram_bot.py .
# COPY --chown=appuser:appgroup predict_with_reasoning.py .
# COPY --chown=appuser:appgroup config config/
# COPY --chown=appuser:appgroup model model/
# COPY --chown=appuser:appgroup utils utils/
# COPY --chown=appuser:appgroup app_modules app_modules/ # Si utilisé

# S'assurer que les scripts exécutables le sont (si nécessaire)
# RUN chmod +x entrypoint.sh # Si vous utilisez un script d'entrée

# Changer d'utilisateur
USER appuser

# Commande par défaut pour exécuter le bot Telegram
# Les variables d'environnement comme TELEGRAM_BOT_TOKEN, paths pour les modèles/données
# seront injectées au runtime (par ex. via docker run -e ou le service systemd)
CMD ["python", "telegram_bot.py"]

# Note sur les modèles et données:
# Ce Dockerfile suppose que les fichiers comme:
# - outputs/enhanced/best_model.keras
# - outputs/enhanced/feature_names.json
# - data/cache/signal_cache.parquet
# seront montés en tant que volumes lors de l'exécution du conteneur.
# Si vous devez les inclure dans l'image, ajoutez des instructions COPY appropriées
# avant de changer d'utilisateur (USER appuser), par exemple:
# USER root # Temporairement pour créer les répertoires et copier
# RUN mkdir -p /home/appuser/app/outputs/enhanced && mkdir -p /home/appuser/app/data/cache
# COPY --chown=appuser:appgroup outputs/enhanced/best_model.keras /home/appuser/app/outputs/enhanced/best_model.keras
# COPY --chown=appuser:appgroup outputs/enhanced/feature_names.json /home/appuser/app/outputs/enhanced/feature_names.json
# COPY --chown=appuser:appgroup data/cache/signal_cache.parquet /home/appuser/app/data/cache/signal_cache.parquet
# USER appuser
