import os
import shutil

def kopiere_alle_dateien(quellordner, zielordner):
    # Erstelle den Zielordner, falls er nicht existiert
    if not os.path.exists(zielordner):
        os.makedirs(zielordner)

    # Gehe durch alle Dateien und Unterordner im Quellordner
    for ordnername, unterordner, dateinamen in os.walk(quellordner):
        for dateiname in dateinamen:
            dateipfad = os.path.join(ordnername, dateiname)
            ziel_dateipfad = os.path.join(zielordner, dateiname)

            # Kopiere jede Datei in den Zielordner
            shutil.copy(dateipfad, ziel_dateipfad)
            print(f'Kopiert: {dateipfad} nach {ziel_dateipfad}')

# Setze den Quell- und Zielordner
# schreibe alle \ als / oder doppelt \\ für das: C:\Users\christian.zwiessler\PycharmProjects\bachelor-thesis\data\full_data\train_data_one_folder
# quellordner = 'C:/Users/christian.zwiessler/PycharmProjects/bachelor-thesis/data/full_data/train_data_one_folder'

quellordner = 'C:/Users/christian.zwiessler/PycharmProjects/bachelor-thesis/data/train'
zielordner = 'C:/Users/christian.zwiessler/PycharmProjects/bachelor-thesis/data/full_data/train_data_one_folder'

kopiere_alle_dateien(quellordner, zielordner)
