from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="YOLO Tree & Human Detection API")

# -------- CONFIG --------
MODEL_PATH = "my_model.pt"
CONFIDENCE_THRESHOLD = 0.25
# ------------------------

model = YOLO(MODEL_PATH)

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # Response structure
    response = {
        "tree_pixel_heights": [],
        "human_pixel_heights": [],
        "tree_count": 0,
        "human_count": 0,
        "tree_bottom_pixels": [],
        "human_bottom_pixels": []
    }

    for r in results:
        boxes = r.boxes

        for box, cls in zip(boxes.xyxy, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = r.names[int(cls)]

            pixel_height = y2 - y1
            bottom_pixel = y2

            if label.lower() == "tree":
                response["tree_count"] += 1
                response["tree_pixel_heights"].append(pixel_height)
                response["tree_bottom_pixels"].append(bottom_pixel)

            elif label.lower() in ["person", "human"]:
                response["human_count"] += 1
                response["human_pixel_heights"].append(pixel_height)
                response["human_bottom_pixels"].append(bottom_pixel)

    return response
