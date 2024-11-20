import os

def extract_png_filenames(input_folder, output_file):
    try:
        # Überprüfen, ob der Ordner existiert
        if not os.path.exists(input_folder):
            print(f"Der Ordner '{input_folder}' existiert nicht.")
            return

        # Liste aller .png-Dateien im Ordner
        png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

        # Schreiben der Dateinamen in die Ausgabedatei
        with open(output_file, 'w') as file:
            for png_file in png_files:
                file.write(png_file + '\n')

        print(f"Die Namen der .png-Dateien wurden in '{output_file}' gespeichert.")
    except Exception as e:
        print(f"Es ist ein Fehler aufgetreten: {e}")

# Beispielaufruf
input_folder = "input_folder"  # Ersetze dies durch den Pfad zu deinem Ordner
output_file = "output.txt"                 # Name der Ausgabedatei
extract_png_filenames(input_folder, output_file)
