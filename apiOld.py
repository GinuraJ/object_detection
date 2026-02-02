from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import traceback
import os

app = FastAPI(title="Tree & Human Detection API")

# ---------- CONFIG ----------
MODEL_PATH = "my_model.pt"  # make sure this file is in the same folder as api.py
CONFIDENCE_THRESHOLD = 0.5
# -----------------------------

# Load YOLO model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)
labels = model.names  # class names


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            img_path = tmp.name

        # Load image
        frame = cv2.imread(img_path)
        if frame is None:
            return JSONResponse({"error": "Failed to read image"}, status_code=400)

        # Run inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Initialize outputs
        tree_heights = []
        human_heights = []
        tree_bottom_pixels = []
        human_bottom_pixels = []

        tree_count = 0
        human_count = 0

        # Process detections
        for det in detections:
            xyxy = det.xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(det.cls.item())
            classname = labels[classidx].lower()
            conf = float(det.conf.item())

            if conf >= CONFIDENCE_THRESHOLD:
                pixel_height = int(ymax - ymin)
                bottom_pixel = int(ymax)

                if classname == "tree":
                    tree_heights.append(pixel_height)
                    tree_bottom_pixels.append(bottom_pixel)
                    tree_count += 1

                elif classname == "human":
                    human_heights.append(pixel_height)
                    human_bottom_pixels.append(bottom_pixel)
                    human_count += 1

        # Build JSON response (always include keys)
        response = {
            "tree_pixel_heights": tree_heights,
            "human_pixel_heights": human_heights,
            "tree_count": tree_count,
            "human_count": human_count,
            "tree_bottom_pixels": tree_bottom_pixels,
            "human_bottom_pixels": human_bottom_pixels
        }

        return JSONResponse(response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
