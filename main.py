# =========================================================
# IMPORTS
# =========================================================

import os
import uuid
import io
import re
import datetime
import numpy as np
import tensorflow as tf
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from PIL import Image
import gdown

from database import Base, engine, get_db
import models


# =========================================================
# DATABASE
# =========================================================

Base.metadata.create_all(bind=engine)

# =========================================================
# FASTAPI INIT
# =========================================================

app = FastAPI(
    title="SafeSkin AI API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 224
THRESHOLD = 0.173

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "generated_reports"))

os.makedirs(REPORT_FOLDER, exist_ok=True)

app.mount(
    "/generated_reports",
    StaticFiles(directory=REPORT_FOLDER),
    name="generated_reports"
)

# =========================================================
# GLOBAL MODEL VARIABLES
# =========================================================

model1 = None
model2 = None


# =========================================================
# MODEL DOWNLOAD + LOAD
# =========================================================

def valid_model(path):
    return os.path.exists(path) and os.path.getsize(path) > 50000000


@app.on_event("startup")
def load_models():

    global model1, model2

    url1 = "https://drive.google.com/uc?id=1YC8k5q6gMWFJvznVvZSgyyCsBvLRCEF2"
    url2 = "https://drive.google.com/uc?id=1i67AdZVtzoduCI0iSDMrcLoZV9R3uGeV"

    model1_path = os.path.join(BASE_DIR, "densenet_model.keras")
    model2_path = os.path.join(BASE_DIR, "inception_model.keras")

    if not valid_model(model1_path):
        print("⬇ Downloading DenseNet model...")
        gdown.download(url1, model1_path, quiet=False, fuzzy=True)

    if not valid_model(model2_path):
        print("⬇ Downloading Inception model...")
        gdown.download(url2, model2_path, quiet=False, fuzzy=True)

    print("🔄 Loading DenseNet...")
    model1 = tf.keras.models.load_model(model1_path, compile=False)

    print("🔄 Loading Inception...")
    model2 = tf.keras.models.load_model(model2_path, compile=False)

    print("✅ Models loaded successfully")


# =========================================================
# IMAGE PREPROCESS
# =========================================================

def preprocess_image(image: Image.Image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(prob):

    if prob < 0.10:
        return "Very Low"
    elif prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Moderate"
    elif prob < 0.75:
        return "High"
    else:
        return "Very High"


# =========================================================
# GRADCAM
# =========================================================

# def make_gradcam_heatmap(img_array, model, layer_name):

#     last_conv_layer = model.get_layer(layer_name)

#     # Fix model output (Keras sometimes returns list)
#     model_output = model.output
#     if isinstance(model_output, list):
#         model_output = model_output[0]

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[last_conv_layer.output, model_output]
#     )

#     with tf.GradientTape() as tape:

#         conv_outputs, predictions = grad_model(img_array)

#         if isinstance(predictions, (list, tuple)):
#             predictions = predictions[0]

#         class_channel = predictions[:, 0]

#     grads = tape.gradient(class_channel, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

#     conv_outputs = conv_outputs[0]

#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = tf.maximum(heatmap, 0)
#     heatmap /= (tf.reduce_max(heatmap) + 1e-8)

#     return heatmap.numpy()


def make_gradcam_heatmap(img_array, model, layer_name):

    last_conv_layer = model.get_layer(layer_name)

    # Handle models whose output is wrapped in list
    model_output = model.output
    if isinstance(model_output, (list, tuple)):
        model_output = model_output[0]

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model_output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(

    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    doctor_name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)

):

    if not re.match(r"^[A-Za-z ]+$", patient_name.strip()):
        raise HTTPException(400, "Invalid patient name")

    if not re.match(r"^[A-Za-z ]+$", doctor_name.strip()):
        raise HTTPException(400, "Invalid doctor name")

    if age < 1 or age > 120:
        raise HTTPException(400, "Invalid age")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Invalid image format")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")

    img_array = preprocess_image(image)

    p1 = float(model1.predict(img_array, verbose=0)[0][0])
    p2 = float(model2.predict(img_array, verbose=0)[0][0])

    prob = (p1 + p2) / 2

    diagnosis = "Malignant" if prob > THRESHOLD else "Benign"
    risk = get_risk_level(prob)

    heatmap1 = make_gradcam_heatmap(img_array, model1, "conv5_block32_concat")
    heatmap2 = make_gradcam_heatmap(img_array, model2, "mixed10")

    heatmap1 = cv2.resize(heatmap1, image.size)
    heatmap2 = cv2.resize(heatmap2, image.size)

    combined = 0.6 * heatmap1 + 0.4 * heatmap2
    combined = np.maximum(combined, 0)
    combined /= (np.max(combined) + 1e-8)

    heatmap_uint8 = np.uint8(255 * combined)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)

    original_np = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(original_np, 0.65, heatmap_color, 0.55, 0)

    report_id = f"SSAI-{datetime.datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"

    original_path = os.path.join(REPORT_FOLDER, f"{report_id}_original.jpg")
    gradcam_path = os.path.join(REPORT_FOLDER, f"{report_id}_gradcam.jpg")
    comparison_path = os.path.join(REPORT_FOLDER, f"{report_id}_comparison.jpg")

    cv2.imwrite(original_path, original_np)
    cv2.imwrite(gradcam_path, overlay)

    comparison = np.hstack((original_np, overlay))
    cv2.imwrite(comparison_path, comparison)

    report = models.Report(
        report_id=report_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        doctor_name=doctor_name,
        diagnosis=diagnosis,
        probability=round(prob * 100, 2),
        risk=risk,
        original_path=original_path,
        gradcam_path=gradcam_path,
        comparison_path=comparison_path
    )

    db.add(report)
    db.commit()

    return JSONResponse({
        "report_id": report_id,
        "diagnosis": diagnosis,
        "probability": round(prob * 100, 2),
        "risk": risk
    })


# =========================================================
# PDF GENERATION
# =========================================================

@app.get("/generate-report/{report_id}")
def generate_report(report_id: str, db: Session = Depends(get_db)):

    report = db.query(models.Report).filter(
        models.Report.report_id == report_id
    ).first()

    if not report:
        raise HTTPException(404, "Report not found")

    file_path = os.path.join(REPORT_FOLDER, f"{report_id}.pdf")

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>SafeSkin AI Medical Report</b>", styles["Title"]))
    elements.append(Spacer(1,20))

    data = [
        ["Report ID", report.report_id],
        ["Patient", report.patient_name],
        ["Age", str(report.age)],
        ["Gender", report.gender],
        ["Doctor", report.doctor_name],
        ["Diagnosis", report.diagnosis],
        ["Risk Level", report.risk],
        ["Confidence", f"{report.probability}%"]
    ]

    elements.append(Table(data, colWidths=[150,300]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph("<b>Original vs GradCAM</b>", styles["Heading2"]))
    elements.append(Spacer(1,10))

    elements.append(RLImage(report.comparison_path, width=6*inch, height=3*inch))

    doc.build(elements)

    return FileResponse(file_path, media_type="application/pdf")


# =========================================================
# HISTORY
# =========================================================

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(models.Report).all()


# =========================================================
# DELETE REPORT
# =========================================================

@app.delete("/delete-report/{report_id}")
def delete_report(report_id: str, db: Session = Depends(get_db)):

    report = db.query(models.Report).filter(
        models.Report.report_id == report_id
    ).first()

    if not report:
        raise HTTPException(404, "Report not found")

    for path in [
        report.original_path,
        report.gradcam_path,
        report.comparison_path,
        os.path.join(REPORT_FOLDER, f"{report_id}.pdf")
    ]:
        if os.path.exists(path):
            os.remove(path)

    db.delete(report)
    db.commit()

    return {"message": "Report deleted successfully"}