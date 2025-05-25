#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour nettoyer les notebooks Jupyter en supprimant les cellules exploratoires
et en optimisant la taille des fichiers.

Les cellules marquées avec un commentaire '# EXPLORATORY' seront supprimées
pour rendre les notebooks plus propres et plus légers.

Usage:
    python clean_notebooks.py [--dir <directory>] [--output <output_dir>] [--dry-run]
"""

import os
import sys
import argparse
import json
import nbformat
from nbformat.v4 import new_notebook
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
import re
import shutil


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Nettoie les notebooks en supprimant les cellules exploratoires.")
    parser.add_argument("--dir", type=str, default=".", help="Répertoire contenant les notebooks")
    parser.add_argument("--output", type=str, default=None, 
                      help="Répertoire où sauvegarder les notebooks nettoyés (si différent du répertoire d'origine)")
    parser.add_argument("--dry-run", action="store_true", help="Exécution à blanc, sans modification réelle")
    parser.add_argument("--clear-outputs", action="store_true", help="Effacer les outputs des cellules")
    parser.add_argument("--clear-execution-count", action="store_true", help="Réinitialiser les compteurs d'exécution")
    return parser.parse_args()


def is_exploratory_cell(cell: Dict[str, Any]) -> bool:
    """
    Détermine si une cellule est marquée comme exploratoire.
    
    Args:
        cell: Cellule du notebook
        
    Returns:
        True si la cellule est exploratoire, False sinon
    """
    if cell["cell_type"] == "code":
        if "source" in cell and cell["source"]:
            # Vérifier si le tag # EXPLORATORY apparaît dans le code
            if re.search(r'#\s*EXPLORATORY', cell["source"]):
                return True
            
    # Vérifier si les métadonnées de la cellule contiennent des tags 'exploratory'
    if "metadata" in cell and "tags" in cell["metadata"]:
        tags = cell["metadata"]["tags"]
        if isinstance(tags, list) and any(tag.lower() == "exploratory" for tag in tags):
            return True
    
    return False


def clean_notebook(notebook_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, 
                  dry_run: bool = False, clear_outputs: bool = False, 
                  clear_execution_count: bool = False) -> int:
    """
    Nettoie un notebook en supprimant les cellules exploratoires.
    
    Args:
        notebook_path: Chemin vers le notebook à nettoyer
        output_path: Chemin où sauvegarder le notebook nettoyé (optionnel)
        dry_run: Si True, n'écrit pas le résultat mais affiche les actions prévues
        clear_outputs: Si True, supprime les sorties des cellules
        clear_execution_count: Si True, réinitialise les compteurs d'exécution
        
    Returns:
        Nombre de cellules supprimées
    """
    notebook_path = Path(notebook_path)
    
    if output_path is None:
        output_path = notebook_path
    else:
        output_path = Path(output_path)
    
    print(f"Traitement de {notebook_path}...")
    
    try:
        # Charger le notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        # Compter les cellules originales
        original_cell_count = len(nb.cells)
        
        # Filtrer les cellules non-exploratoires
        new_cells = []
        removed_cells = 0
        
        for cell in nb.cells:
            if is_exploratory_cell(cell):
                removed_cells += 1
                print(f"  - Suppression cellule exploratoire: {cell['source'][:50].strip()}...")
                continue
            
            # Traiter la cellule selon les options
            if clear_outputs and cell["cell_type"] == "code":
                cell["outputs"] = []
                if "execution_count" in cell and clear_execution_count:
                    cell["execution_count"] = None
            
            new_cells.append(cell)
        
        # Créer un nouveau notebook avec les cellules filtrées
        new_nb = new_notebook(
            metadata=nb.metadata,
            cells=new_cells
        )
        
        # Sauvegarder le notebook nettoyé
        if not dry_run:
            with open(output_path, "w", encoding="utf-8") as f:
                nbformat.write(new_nb, f)
            
            print(f"  → Notebook sauvegardé: {output_path}")
            print(f"  → {removed_cells} cellules supprimées sur {original_cell_count} ({removed_cells/original_cell_count*100:.1f}%)")
        else:
            print(f"  [DRY RUN] {removed_cells} cellules seraient supprimées sur {original_cell_count} ({removed_cells/original_cell_count*100:.1f}%)")
        
        return removed_cells
    
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {notebook_path}: {e}")
        return 0


def process_directory(directory: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, 
                     dry_run: bool = False, clear_outputs: bool = False, 
                     clear_execution_count: bool = False) -> Dict[str, int]:
    """
    Traite tous les notebooks dans un répertoire.
    
    Args:
        directory: Répertoire à traiter
        output_dir: Répertoire où sauvegarder les notebooks nettoyés
        dry_run: Si True, n'écrit pas les résultats
        clear_outputs: Si True, supprime les sorties des cellules
        clear_execution_count: Si True, réinitialise les compteurs d'exécution
        
    Returns:
        Statistiques des notebooks traités
    """
    directory = Path(directory)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "notebooks_processed": 0,
        "notebooks_modified": 0,
        "total_cells_removed": 0
    }
    
    # Parcourir les fichiers du répertoire
    for file_path in directory.glob("**/*.ipynb"):
        # Ignorer les fichiers dans .ipynb_checkpoints
        if ".ipynb_checkpoints" in str(file_path):
            continue
        
        # Déterminer le chemin de sortie
        if output_dir is not None:
            # Conserver la structure relative des répertoires
            rel_path = file_path.relative_to(directory)
            target_path = output_dir / rel_path
            os.makedirs(target_path.parent, exist_ok=True)
        else:
            target_path = file_path
        
        # Nettoyer le notebook
        cells_removed = clean_notebook(
            file_path, target_path, 
            dry_run=dry_run,
            clear_outputs=clear_outputs,
            clear_execution_count=clear_execution_count
        )
        
        stats["notebooks_processed"] += 1
        if cells_removed > 0:
            stats["notebooks_modified"] += 1
            stats["total_cells_removed"] += cells_removed
    
    return stats


def main():
    """Fonction principale."""
    args = parse_args()
    
    print(f"Nettoyage des notebooks dans {args.dir}")
    if args.dry_run:
        print("[MODE DRY RUN] Aucune modification ne sera effectuée")
    
    stats = process_directory(
        args.dir, args.output,
        dry_run=args.dry_run,
        clear_outputs=args.clear_outputs,
        clear_execution_count=args.clear_execution_count
    )
    
    print("\nRésumé:")
    print(f"- {stats['notebooks_processed']} notebooks traités")
    print(f"- {stats['notebooks_modified']} notebooks modifiés")
    print(f"- {stats['total_cells_removed']} cellules exploratoires supprimées")


if __name__ == "__main__":
    main() 