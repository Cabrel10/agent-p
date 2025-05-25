import os
import yaml
from dotenv import load_dotenv

from pathlib import Path  # Importer Path


class Config:
    _instance = None
    _config_data = None
    _project_root = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance._load_config()
        return cls._instance

    def _find_project_root(self):
        """Trouve le répertoire racine du projet ('Morningstar')."""
        current_path = Path(__file__).resolve()
        # Remonter jusqu'à trouver le dossier 'Morningstar' ou la racine du système
        while current_path.name != "Morningstar" and current_path.parent != current_path:
            current_path = current_path.parent
        if current_path.name == "Morningstar":
            return current_path
        else:
            # Fallback: utiliser le répertoire de travail actuel si 'Morningstar' n'est pas trouvé
            print("WARN: Répertoire racine 'Morningstar' non trouvé en remontant depuis config.py. Utilisation de CWD.")
            return Path(os.getcwd())

    def _load_config(self):
        """Charge la configuration depuis les fichiers .env et .yaml."""
        if Config._config_data is None:
            Config._project_root = self._find_project_root()
            print(f"DEBUG: Project root identified as: {Config._project_root}")

            secrets_path = Config._project_root / "config" / "secrets.env"
            yaml_path = Config._project_root / "config" / "config.yaml"

            print(f"DEBUG: Attempting to load .env from: {secrets_path}")
            load_dotenv(secrets_path, override=True)

            print(f"DEBUG: Attempting to load YAML from: {yaml_path}")
            try:
                with open(yaml_path, "r") as f:
                    Config._config_data = yaml.safe_load(f) or {}
                print(f"DEBUG: Config YAML loaded successfully from: {yaml_path}")
            except FileNotFoundError:
                print(f"ERROR: Configuration file not found at {yaml_path}")
                Config._config_data = {}
            except Exception as e:
                print(f"ERROR: Failed to load or parse YAML configuration from {yaml_path}: {e}")
                Config._config_data = {}

            # Charger les variables d'environnement après le YAML pour priorité
            self._load_env_vars()

    def _load_env_vars(self):
        """Charge les variables d'environnement spécifiques."""
        # Config Redis
        self.redis = type("", (), {})()
        self.redis.host = os.getenv("REDIS_HOST", "localhost")
        self.redis.port = int(os.getenv("REDIS_PORT", 6379))
        self.redis.db = int(os.getenv("REDIS_DB", 0))
        # Lire cache_ttl depuis les données chargées
        self.redis.cache_ttl = self.get_config("llm.cache_ttl", 86400)

        # Config LLM
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    @property
    def yaml_config(self):
        """Retourne les données de configuration chargées."""
        # Assurer que la config est chargée si ce n'est pas déjà fait
        if Config._config_data is None:
            self._load_config()
        return Config._config_data

    def get_config(self, key_path, default=None):
        """
        Récupère une valeur de configuration en utilisant un chemin de clé de type 'a.b.c'.
        Retourne une valeur par défaut si la clé n'est pas trouvée.
        """
        keys = key_path.split(".")
        value = self.yaml_config
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:  # Gérer le cas où un segment de chemin n'est pas un dictionnaire
                    return default
            return value
        except (KeyError, TypeError):
            return default
