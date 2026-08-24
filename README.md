# 🏍️ Superbike & Motorcycle Classifier API

A lightweight web app for classifying **60 motorcycle models** in Vietnam—from everyday street commuters to rare, exotic superbikes.

---

### 📌 Project Overview
* **Dataset:** ~12,000 images across 60 classes (Hosted on [Roboflow Universe](https://universe.roboflow.com/xe-i5j8x/xe-may-qhsd0).
* **Model:** `YOLO26n` (Nano)—optimized for fast inference under hardware constraints.
* **Training:** 200 epochs.
* **Stack:** FastAPI (Backend) + HTML/JS (Frontend) + PyTorch & Ultralytics.

---

### 🚀 Quickstart

1. **Install Dependencies**
   ```powershell
   uv sync
2. **Run server**
    ```powershell
    uv run fastapi dev
3. **Open http://127.0.0.1:8000 in your browser.**