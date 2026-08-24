import io
from pathlib import Path
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # <-- Add this line
from PIL import Image
from ultralytics import YOLO
app = FastAPI(title="YOLO Bike Classifier API")

# Allow Frontend access via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load your custom trained YOLO model weights
# Replace 'path/to/best.pt' with your relative or absolute .pt file path
MODEL_PATH = r"runs\runs\train\superbike_detector\weights\best.pt"  

try:
    model = YOLO(MODEL_PATH, task="classify")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model from {MODEL_PATH}: {e}")

FRONTEND_PATH = r"D:\Document\Python\Bike_clasification\frontend\index.html"

@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read image bytes directly into PIL
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLO inference
        results = model(image)
        result = results[0]
    
        # Case A: YOLO Classification Model (.predict / -cls)
        if hasattr(result, "probs") and result.probs is not None:
            top1_idx = int(result.probs.top1)
            top1_conf = float(result.probs.top1conf)
            class_name = result.names[top1_idx]

            # Construct dictionary of all class confidence scores
            all_scores = {
                result.names[idx]: f"{float(conf) * 100:.2f}%"
                for idx, conf in enumerate(result.probs.data)
            }

            return {
                "success": True,
                "prediction": class_name,
                "confidence": f"{top1_conf * 100:.2f}%",
                "all_scores": all_scores,
            }

        # Case B: YOLO Detection Model (.pt with bounding boxes)
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            # Take highest confidence detection box
            top_box = sorted(result.boxes, key=lambda b: float(b.conf[0]), reverse=True)[0]
            cls_id = int(top_box.cls[0])
            conf_val = float(top_box.conf[0])
            class_name = result.names[cls_id]

            return {
                "success": True,
                "prediction": class_name,
                "confidence": f"{conf_val * 100:.2f}%",
                "all_scores": {class_name: f"{conf_val * 100:.2f}%"},
            }

        else:
            return {
                "success": False,
                "prediction": "No object detected",
                "confidence": "0.00%",
                "all_scores": {},
            }

    except Exception as e:
        traceback.print_exc()  # Prints exact error details to your terminal!
        raise HTTPException(status_code=500, detail=str(e))