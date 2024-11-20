import os
from PIL import Image  # Zum Auslesen der Bildabmessungen

def convert_to_yolo_format(input_folder, output_folder, class_id=1):
    """
    Wandelt Textdateien mit `position_vehicle`-Informationen in YOLO-Format um.
    Verwendet die Abmessungen der zugehörigen Bilder, um die Werte zu normalisieren.
    Speichert die neuen Dateien in einem Ausgabeordner.

    :param input_folder: Ordner mit den Eingabe-Textdateien und zugehörigen Bildern.
    :param output_folder: Ordner, in dem die Ausgabedateien gespeichert werden.
    :param class_id: Klassen-ID (Standard: 1 für 'car').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gehe durch jede Datei im Eingabeordner
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Erwarteter Bildpfad basierend auf dem Textdateinamen
            image_name = os.path.splitext(filename)[0]  # Dateiname ohne Endung
            image_path = None

            # Suche nach einer passenden Bilddatei
            for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                potential_image_path = os.path.join(input_folder, image_name + ext)
                if os.path.exists(potential_image_path):
                    image_path = potential_image_path
                    break

            if not image_path:
                print(f"Warnung: Kein passendes Bild für {filename} gefunden. Überspringe diese Datei.")
                continue

            # Lese die Bildabmessungen
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            with open(input_path, 'r') as file:
                lines = file.readlines()

            # Suche nach `position_vehicle`
            yolo_lines = []
            for line in lines:
                if "position_vehicle:" in line:
                    # Extrahiere die Bounding-Box-Koordinaten
                    coords = line.split(":")[1].strip().split()
                    x_min, y_min, width, height = map(int, coords)

                    # Konvertiere in YOLO-Format
                    x_center = (x_min + width / 2) / image_width
                    y_center = (y_min + height / 2) / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height

                    # Bereite die YOLO-Format-Zeile vor
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                    yolo_lines.append(yolo_line)

            # Schreibe das Ergebnis in eine neue Datei
            with open(output_path, 'w') as output_file:
                output_file.writelines(yolo_lines)

    print(f"Konvertierung abgeschlossen. Dateien sind im Ordner '{output_folder}' gespeichert.")

# Beispielaufruf
input_folder = "input_folder"  # Ordner mit Original-Textdateien und Bildern
output_folder = "output_texts"  # Ordner für umgewandelte Dateien

convert_to_yolo_format(input_folder, output_folder)
