import os

def convert_to_yolo_format(input_folder, output_folder, image_width, image_height, class_id=1):
    """
    Wandelt Textdateien mit `position_vehicle`-Informationen in YOLO-Format um.
    Speichert die neuen Dateien in einem Ausgabeordner.

    :param input_folder: Ordner mit den Eingabe-Textdateien.
    :param output_folder: Ordner, in dem die Ausgabedateien gespeichert werden.
    :param image_width: Breite der Bilder in Pixel.
    :param image_height: Höhe der Bilder in Pixel.
    :param class_id: Klassen-ID (Standard: 1 für 'car').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gehe durch jede Datei im Eingabeordner
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as file:
                lines = file.readlines()

            # Suche nach `position_vehicle`
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

                    # Schreibe das Ergebnis ins YOLO-Format
                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"

                    # Speichere die neue Datei
                    with open(output_path, 'w') as output_file:
                        output_file.write(yolo_line)

    print(f"Konvertierung abgeschlossen Dateien sind im Ordner '{output_folder}' gespeichert.")

# Beispielaufruf
input_folder = "input_folder"  # Ordner mit Original-Textdateien
output_folder = "output_texts"  # Ordner für umgewandelte Dateien
image_width = 1920  # Beispiel-Bildbreite
image_height = 1080  # Beispiel-Bildhöhe

convert_to_yolo_format(input_folder, output_folder, image_width, image_height)
