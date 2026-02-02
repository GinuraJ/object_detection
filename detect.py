import sys
from ultralytics import YOLO
import cv2
import os

# -------- CONFIG --------
MODEL_PATH = "my_model.pt"
CONFIDENCE_THRESHOLD = 0.25
SAVE_RESULTS = True
# ------------------------

# Check image path argument
if len(sys.argv) < 2:
    print("❌ Usage: python detect.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    sys.exit(1)

# Load model
model = YOLO(MODEL_PATH)

# Run prediction
results = model.predict(
    source=IMAGE_PATH,
    conf=CONFIDENCE_THRESHOLD,
    save=False,
    show=False
)

# Process results
for r in results:
    boxes = r.boxes
    img = r.orig_img

    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{r.names[int(cls)]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if SAVE_RESULTS:
        output_dir = "runs/detect/manual"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(IMAGE_PATH))
        cv2.imwrite(output_path, img)
        print(f"✅ Saved: {output_path}")
