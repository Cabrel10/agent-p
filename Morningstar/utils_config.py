import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger("utils_config")

def load_config(config_path=None, required=True):
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path (str ou Path, optional): Chemin vers le fichier de configuration.
            Si None, cherche automatiquement config.yaml dans le répertoire du projet.
        required (bool, optional): Si True, lève une exception si la configuration ne peut pas être chargée.
            Sinon, retourne None. Par défaut à True.
            
    Returns:
        dict: Dictionnaire contenant la configuration chargée, ou None en cas d'erreur si required=False.
        
    Raises:
        FileNotFoundError: Si le fichier de configuration n'est pas trouvé et required=True.
        Exception: Si une autre erreur se produit lors du chargement et required=True.
    """
    try:
        if config_path is None:
            # Tenter de trouver le fichier config.yaml dans le répertoire du projet
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            project_root = script_dir  # Le script utils_config.py est à la racine du projet
            config_path = project_root / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            error_msg = f"Fichier de configuration non trouvé: {config_path}"
            logger.error(error_msg)
            if required:
                raise FileNotFoundError(error_msg)
            return None
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration chargée depuis : {config_path}")
        return config
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement de la configuration depuis {config_path}: {e}"
        logger.error(error_msg)
        if required:
            raise
        return None

def get_project_root(config=None):
    """
    Obtient le chemin racine du projet à partir de la configuration
    ou en le déduisant à partir de l'emplacement du script utils_config.py.
    
    Args:
        config (dict, optional): Dictionnaire de configuration contenant project_root.
            Si None, la configuration est chargée automatiquement.
            
    Returns:
        Path: Chemin absolu vers la racine du projet.
    """
    if config is None:
        config = load_config()
        
    if config and "project_root" in config:
        return Path(config["project_root"])
    else:
        # Fallback: déduire la racine du projet à partir de l'emplacement du script
        return Path(os.path.dirname(os.path.abspath(__file__)))

def get_path(config, path_key, relative_to_root=True):
    """
    Récupère un chemin à partir de la configuration.
    
    Args:
        config (dict): Dictionnaire de configuration.
        path_key (str): Clé du chemin dans config["paths"].
        relative_to_root (bool, optional): Si True, convertit le chemin relatif en absolu
            par rapport à la racine du projet. Par défaut à True.
            
    Returns:
        Path: Chemin absolu si relative_to_root=True, sinon chemin relatif.
        
    Raises:
        KeyError: Si path_key n'existe pas dans config["paths"].
    """
    if "paths" not in config or path_key not in config["paths"]:
        raise KeyError(f"Clé de chemin '{path_key}' non trouvée dans la configuration.")
    
    path = config["paths"][path_key]
    
    if relative_to_root:
        project_root = get_project_root(config)
        return project_root / path
    else:
        return Path(path)

def ensure_directory_exists(path):
    """
    S'assure qu'un répertoire existe, en le créant si nécessaire.
    
    Args:
        path (str ou Path): Chemin du répertoire à créer.
        
    Returns:
        Path: Chemin du répertoire créé.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_script_config(config, script_name):
    """
    Récupère la configuration spécifique à un script.
    
    Args:
        config (dict): Dictionnaire de configuration.
        script_name (str): Nom du script (clé dans config["scripts"]).
        
    Returns:
        dict: Configuration spécifique au script, ou un dictionnaire vide si non trouvée.
    """
    if "scripts" in config and script_name in config["scripts"]:
        return config["scripts"][script_name]
    return {}

# Point d'entrée pour tester le module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Tester le chargement de la configuration
    config = load_config()
    if config:
        print(f"Configuration chargée avec succès.")
        print(f"Racine du projet: {get_project_root(config)}")
        
        # Tester la récupération de quelques chemins
        try:
            data_dir = get_path(config, "data_base_dir")
            print(f"Répertoire de données: {data_dir}")
            
            models_dir = get_path(config, "models_base_dir")
            print(f"Répertoire des modèles: {models_dir}")
            
            # Tester la récupération de la configuration d'un script
            train_rl_config = get_script_config(config, "train_rl_agent")
            print(f"Configuration de train_rl_agent: {train_rl_config}")
        except KeyError as e:
            print(f"Erreur: {e}")
    else:
        print("Échec du chargement de la configuration.")