#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'initialisation pour l'environnement trading_env
-------------------------------------------------------

Ce script vérifie et installe les dépendances nécessaires pour 
le modèle monolithique Morningstar dans l'environnement conda trading_env.
"""

import os
import sys
import subprocess
import importlib

def check_and_install(package):
    """Vérifie si un package est installé et l'installe si nécessaire avec conda."""
    package_name = package.split('>=')[0] if '>=' in package else package
    
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} est déjà installé.")
    except ImportError:
        print(f"⏳ Installation de {package} avec conda...")
        try:
            subprocess.check_call(["conda", "install", "-y", package])
            print(f"✅ {package} installé avec succès via conda.")
        except subprocess.CalledProcessError:
            print(f"⚠️ Échec d'installation via conda. Tentative avec pip...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installé avec succès via pip.")
            except subprocess.CalledProcessError:
                print(f"❌ Échec d'installation de {package}. Installation manuelle requise.")

def init_environment():
    """Initialise l'environnement trading_env avec les dépendances nécessaires."""
    print("🔄 Initialisation de l'environnement trading_env pour le modèle monolithique...")
    
    # Liste des dépendances
    dependencies = [
        "tensorflow",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "ta-lib",  # Bibliothèque d'indicateurs techniques
        "backtrader",  # Pour le backtesting
    ]
    
    # Vérifier et installer chaque dépendance
    for dep in dependencies:
        check_and_install(dep)
    
    print("\n✅ Environnement trading_env configuré avec succès pour le modèle monolithique!")
    print("\nPour commencer à utiliser le modèle monolithique:")
    print("1. Importez le modèle: `from ultimate.model.architecture.monolith_model import MonolithModel`")
    print("2. Créez une instance: `model = MonolithModel()`")
    print("3. Pour l'entraînement: `python -m ultimate.model.training.train_monolith --help`")
    print("4. Pour le backtest: `python -m ultimate.model.run_backtest --help`")

if __name__ == "__main__":
    init_environment() 