import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

# -------------------------------
# FORCE CPU (Render has no GPU)
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "my_model.pt"
CONFIDENCE_THRESHOLD = 0.25
MAX_IMAGE_SIZE = 1280  # resize large images
# -------------------------------

app = FastAPI(title="YOLO Tree & Human Detection API")

# -------------------------------
# LOAD MODEL (ONCE)
# -------------------------------
model = YOLO(MODEL_PATH)


# -------------------------------
# ROOT ENDPOINT (IMPORTANT)
# -------------------------------
@app.get("/")
def root():
    return {"status": "alive", "service": "YOLO Detection API"}


# -------------------------------
# WARM-UP (CRITICAL FOR RENDER)
# -------------------------------
@app.on_event("startup")
def warmup():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)
    print("✅ YOLO model warmed up")


# -------------------------------
# DETECT ENDPOINT
# -------------------------------
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    # Resize large images (VERY IMPORTANT)
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Run YOLO
    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # Response template
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
            label = r.names[int(cls)].lower()

            pixel_height = y2 - y1
            bottom_pixel = y2

            if label == "tree":
                response["tree_count"] += 1
                response["tree_pixel_heights"].append(pixel_height)
                response["tree_bottom_pixels"].append(bottom_pixel)

            elif label in ["person", "human"]:
                response["human_count"] += 1
                response["human_pixel_heights"].append(pixel_height)
                response["human_bottom_pixels"].append(bottom_pixel)

    return response
