from PIL import Image

def read_label(filepath: str) -> dict:
    with open(filepath, "r") as file:
        image_data_lines = file.readlines()

    label = {
        "license_plate": "",
        "camera": "",
        "position_vehicle": [],
        "license_plate_corners": [],
        "vehicle_type": "",
        "vehicle_make": "",
        "vehicle_model": "",
        "vehicle_year": "",
        "char_positions": [],
    }
    for line in image_data_lines:
        line = line.strip()
        if line.startswith("plate:"):
            label["license_plate"] = line[7:]
        if line.startswith("camera:"):
            label["camera"] = line[8:]
        if line.startswith("position_vehicle:"):
            positions = line[18:].strip().split(" ")
            label["position_vehicle"] = [int(p) for p in positions]
        if line.startswith("corners:"):
            corners = []
            corners_text = line[9:].split(" ")
            for corner in corners_text:
                c0, c1 = corner.split(",")
                corners.append((int(c0), int(c1)))
            label["license_plate_corners"] = corners
        if line.startswith("type:"):
            label["vehicle_type"] = line[6:]
        if line.startswith("make:"):
            label["vehicle_make"] = line[6:]
        if line.startswith("model:"):
            label["vehicle_model"] = line[7:]
        if line.startswith("year:"):
            label["vehicle_year"] = line[6:]
        if line.startswith("char "):
            positions = line[9:].split(" ")
            positions = tuple([int(p) for p in positions])
            label["char_positions"].append(positions)

    return label