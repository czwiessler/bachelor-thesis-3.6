import os
import shutil

def rename_files_in_range(directory, old_object_name, new_object_name, start_number, end_number):
    # Durchlaufe alle Dateien im angegebenen Verzeichnis
    for filename in os.listdir(directory):
        # Prüfe, ob die Datei dem Muster entspricht: old_object_name + Zahl
        if filename.startswith(old_object_name):
            # Extrahiere die Zahl aus dem Dateinamen
            number_part = filename[len(old_object_name):].split('.')[0]
            if number_part.isdigit():
                number = int(number_part)
                # Prüfe, ob die Zahl im angegebenen Bereich liegt
                if start_number <= number <= end_number:
                    # Generiere den neuen Dateinamen
                    new_filename = f"{new_object_name}{number_part}{filename[len(old_object_name)+len(number_part):]}"
                    # Vollständiger Pfad der alten und neuen Datei
                    old_file_path = os.path.join(directory, filename)
                    new_file_path = os.path.join(directory, new_filename)
                    # Benenne die Datei um
                    shutil.move(old_file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")

# Beispiel für den Aufruf der Methode
directory = "C:\\Users\\christian.zwiessler\\PycharmProjects\\bachelor-thesis\\data\\train\\RGB"
rename_files_in_range(directory, "Beermug", "Beermug_0_", 1, 38)
