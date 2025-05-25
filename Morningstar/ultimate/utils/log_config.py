import logging
import os
import sys
import time
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Union, List, Any
from datetime import datetime


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 5,
    enable_console: bool = True,
    module_levels: Optional[Dict[str, int]] = None,
    env: str = "production"
):
    """
    Configure le système de logging pour l'application.

    Args:
        log_dir: Répertoire où sauvegarder les fichiers de log
        log_level: Niveau de logging (par défaut INFO)
        log_format: Format personnalisé pour les logs
        log_file: Nom du fichier de log
        max_size_mb: Taille maximale d'un fichier de log en Mo
        backup_count: Nombre de fichiers de backup à conserver
        enable_console: Activer les logs dans la console
        module_levels: Dictionnaire {nom_module: niveau} pour configurer des niveaux spécifiques
        env: Environnement (development, testing, production)
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        log_file = "trading_workflow.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configurer le format selon l'environnement
    if log_format is None:
        if env == "development":
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        elif env == "testing":
            log_format = "%(asctime)s - %(levelname)s - %(message)s"
        else:  # production
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(thread)d - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Configurer le handler de fichier avec rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Configurer les handlers
    handlers = [file_handler]
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Configurer le root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
    )
    
    # Configurer les niveaux pour les modules spécifiques
    default_module_levels = {
        "matplotlib": logging.WARNING,
        "ccxt": logging.INFO,
        "urllib3": logging.WARNING,
        "tensorflow": logging.WARNING,
        "numpy": logging.WARNING
    }
    
    if module_levels:
        default_module_levels.update(module_levels)
    
    for module, level in default_module_levels.items():
        logging.getLogger(module).setLevel(level)


def get_logger(
    name: str,
    level: int = logging.INFO,
    propagate: bool = True,
    add_handler: bool = False
) -> logging.Logger:
    """
    Obtient un logger configuré pour l'application.
    
    Cette fonction est à utiliser dans tous les modules pour garantir
    une gestion cohérente des logs à travers l'application.
    
    Args:
        name: Nom du logger (généralement __name__ du module)
        level: Niveau de logging spécifique (par défaut INFO)
        propagate: Si True, les messages sont propagés aux loggers parents
        add_handler: Si True, ajoute un handler même si setup_logging() a été appelé
        
    Returns:
        Le logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Vérifier si le logger a déjà des handlers (pour éviter les doublons)
    if not logger.handlers and add_handler:
        # Si aucun handler n'existe, on ajoute un handler pour le logger individuel
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def archive_logs(
    log_dir: Optional[str] = None,
    archive_dir: Optional[str] = None,
    days_to_keep: int = 30,
    compress: bool = True
) -> None:
    """
    Archive les anciens fichiers de log.
    
    Args:
        log_dir: Répertoire contenant les logs
        archive_dir: Répertoire où archiver les logs
        days_to_keep: Nombre de jours pendant lesquels conserver les logs dans le répertoire principal
        compress: Si True, compresse les logs archivés
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    
    if archive_dir is None:
        archive_dir = os.path.join(log_dir, "archives")
    
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    
    # Timestamp actuel en secondes
    now = time.time()
    # Seuil en secondes (jours * 24h * 60min * 60s)
    threshold = now - days_to_keep * 86400
    
    logger = get_logger("log_archiver")
    logger.info(f"Archivage des logs antérieurs à {days_to_keep} jours")
    
    # Parcourir tous les fichiers du répertoire de logs
    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        
        # Vérifier si c'est un fichier (pas un répertoire)
        if os.path.isfile(filepath):
            # Vérifier si le fichier est un log (se termine par .log)
            if filename.endswith('.log') or filename.endswith('.log.1') or filename.endswith('.log.2'):
                # Obtenir la date de dernière modification
                file_mod_time = os.path.getmtime(filepath)
                
                # Si le fichier est plus ancien que le seuil
                if file_mod_time < threshold:
                    # Créer un nom d'archive avec timestamp
                    archive_name = f"{filename}_{datetime.fromtimestamp(file_mod_time).strftime('%Y%m%d')}"
                    archive_path = os.path.join(archive_dir, archive_name)
                    
                    # Compresser si demandé
                    if compress:
                        try:
                            import zipfile
                            zip_path = archive_path + ".zip"
                            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                zipf.write(filepath, arcname=filename)
                            logger.info(f"Log archivé et compressé: {filename} -> {zip_path}")
                            os.remove(filepath)
                        except Exception as e:
                            logger.error(f"Erreur lors de la compression de {filename}: {e}")
                    else:
                        # Sinon, simplement déplacer le fichier
                        shutil.move(filepath, archive_path)
                        logger.info(f"Log archivé: {filename} -> {archive_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configurer le système de logging
    setup_logging(env="development")
    
    # Obtenir un logger pour le module courant
    logger = get_logger(__name__)
    
    # Écrire des messages de test
    logger.debug("Message de debug")
    logger.info("Message d'information")
    logger.warning("Message d'avertissement")
    logger.error("Message d'erreur")
    
    # Archiver les anciens logs
    archive_logs(days_to_keep=7)
