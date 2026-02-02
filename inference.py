# inference.py
from ultralytics import YOLO

model = YOLO("my_model.pt")
labels = model.names


def detect_objects(image, conf=0.5):
    results = model(image, conf=conf, verbose=False)
    boxes = results[0].boxes

    response = {
        "tree_count": 0,
        "human_count": 0,
        "trees": [],
        "humans": []
    }

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        class_id = int(box.cls[0])
        class_name = labels[class_id].lower()
        confidence = float(box.conf[0])

        pixel_height = y2 - y1
        bottom_pixel = y2

        item = {
            "confidence": round(confidence, 3),
            "pixel_height": int(pixel_height),
            "bottom_pixel": int(bottom_pixel),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        }

        if class_name == "tree":
            response["tree_count"] += 1
            response["trees"].append(item)

        elif class_name == "human":
            response["human_count"] += 1
            response["humans"].append(item)

    return response
