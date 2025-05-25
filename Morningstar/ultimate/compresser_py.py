import os
import zipfile


def compress_python_files(root_dir, output_zip):
    """
    Compresse tous les fichiers .py trouvés dans les répertoires et sous-répertoires
    du répertoire racine spécifié, en conservant la structure de la hiérarchie.

    Args:
        root_dir (str): Le chemin du répertoire racine du projet.
        output_zip (str): Le chemin du fichier ZIP de sortie.
    """
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    zipf.write(file_path, relative_path)
    print(f"Les fichiers .py ont été compressés dans : {output_zip}")


if __name__ == "__main__":
    projet_courant = os.getcwd()
    nom_archive = "python_scripts.zip"
    compress_python_files(projet_courant, nom_archive)
