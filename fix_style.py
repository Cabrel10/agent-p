#!/usr/bin/env python
# Script pour corriger les problèmes de style dans multi_asset_env.py

import re
import sys

def fix_style(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Corriger les variables non utilisées
    content = content.replace("old_tier = self._get_current_tier()", "# Obtenir le tier avant action")
    content = content.replace("execution_price =", "# execution_price =")
    
    # Corriger les importations non utilisées
    content = content.replace("from rich.console import Console", "# from rich.console import Console")
    
    # Corriger les redéfinitions
    content = re.sub(r'import numpy as np\n\s+import pandas as pd\n\s+import os', 
                    "# Utiliser les importations déjà définies", content)
    
    # Supprimer les espaces en fin de ligne
    content = re.sub(r' +$', '', content, flags=re.MULTILINE)
    
    # Corriger les f-strings sans placeholders
    content = re.sub(r'f"([^{]*)"', r'"\1"', content)
    
    # Écrire le contenu corrigé
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"Corrections de style appliquées à {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_style(sys.argv[1])
    else:
        print("Usage: python fix_style.py <filename>")
