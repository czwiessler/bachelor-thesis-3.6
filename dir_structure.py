import os

def print_directory_structure(startpath, excluded_folders=None):
    if excluded_folders is None:
        excluded_folders = ['.venv', '.venv311', '.git', '.idea', 'data']  # Standardmäßig ausgeschlossene Verzeichnisse

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in excluded_folders]  # Entferne ausgeschlossene Verzeichnisse aus der Suche
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# Verwende das aktuelle Verzeichnis als Startpunkt und schließe spezifische Ordner aus
print_directory_structure('.')
