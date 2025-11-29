import os
import io
import time
import joblib
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from math import exp

MODEL_PATH = "models/svm_rice.pkl"   
CAPTURE_DIR = "captures"            
IMG_SIZE = (128, 128)               

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
st.set_page_config(page_title="Deteksi Penyakit Daun Padi - Kamera", layout="centered")
st.title("Deteksi Penyakit Daun Padi (SVM) — Live Camera")
st.write("Ambil foto daun padi menggunakan kamera, lalu tekan Predict.")

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model tidak ditemukan di: {path}. Pastikan kamu sudah menjalankan training dan menyimpan model di lokasi ini.")
        return None
    model = joblib.load(path)
    return model

model = load_model(MODEL_PATH)

def preprocess_image_to_hog(pil_img: Image.Image):
    """
    Input: PIL image (RGB)
    Output: 1D numpy array feature HOG
    """
    img = np.array(pil_img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    img_resized = cv2.resize(img, IMG_SIZE)
    img_float = img_resized.astype("float32") / 255.0
    fd = hog(img_float, orientations=9, pixels_per_cell=(8,8),
            cells_per_block=(2,2), channel_axis=-1, feature_vector=True)
    return fd

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

camera_image = st.camera_input("Ambil foto daun padi (kamera)", help="Pastikan daun memenuhi frame dan pencahayaan cukup")
st.write("— atau —")
uploaded_file = st.file_uploader("Upload gambar jika tidak ingin pakai kamera", type=["jpg","jpeg","png"])

if st.button("Predict"):

    pil_img = None
    if camera_image is not None:
        try:
            pil_img = Image.open(camera_image)
        except Exception as e:
            st.error("Gagal membaca gambar dari kamera: " + str(e))
    elif uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
        except Exception as e:
            st.error("Gagal membaca file upload: " + str(e))
    else:
        st.warning("Belum ada gambar: ambil foto dengan kamera atau upload gambar.")
    
    if pil_img is not None and model is not None:
        st.image(pil_img, caption="Input image", use_column_width=True)
        timestamp = int(time.time())
        save_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")

        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(save_path)

        try:
            feat = preprocess_image_to_hog(pil_img).reshape(1, -1)
        except Exception as e:
            st.error("Gagal preprocessing gambar (HOG): " + str(e))
            feat = None

        if feat is not None:
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feat)[0]
                    idx = np.argmax(probs)
                    label = model.classes_[idx]
                    confidence = probs[idx] * 100
                    st.success(f"Prediksi: **{label}** — Confidence: **{confidence:.2f}%**")
                else:
                    if hasattr(model, "decision_function"):
                        scores = model.decision_function(feat)
                        if scores.ndim == 1:
                            scores = np.vstack([-scores, scores]).T
                        probs = softmax(scores[0])
                        idx = np.argmax(probs)
                        label = model.classes_[idx]
                        confidence = probs[idx] * 100
                        st.success(f"Prediksi: **{label}** — Confidence (pseudo): **{confidence:.2f}%**")
                    else:
                        label = model.predict(feat)[0]
                        st.success(f"Prediksi: **{label}**")
            except Exception as e:
                st.error("Gagal melakukan prediksi: " + str(e))

            try:
                if hasattr(model, "predict_proba"):
                    probs_all = model.predict_proba(feat)[0]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(feat)
                    if scores.ndim == 1:
                        scores = np.vstack([-scores, scores]).T
                    probs_all = softmax(scores[0])
                else:
                    probs_all = None

                if probs_all is not None:
                    classes = list(model.classes_)
                    st.write("Confidence per kelas:")
                    for c, p in zip(classes, probs_all):
                        st.write(f"- {c}: {p*100:.2f}%")
            except Exception as e:
                pass

st.markdown("---")
st.caption("Catatan: Jika model SVM kamu dilatih tanpa `probability=True`, confidence yang ditampilkan adalah pseudo-probability (dari decision_function). Untuk hasil confidence yang lebih valid, re-train SVM dengan `SVC(probability=True)`.")