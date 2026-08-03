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
import plotly.graph_objects as go

MODEL_PATH = "models/svm_rice.pkl"
CAPTURE_DIR = "captures"
IMG_SIZE = (128, 128)

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Deteksi Penyakit Daun Padi",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_css("apps/style.css")

st.markdown("""
<div class="main-header">
    <h1>🌾 Deteksi Penyakit Daun Padi</h1>
    <p style="font-size: 1.1rem; margin-top: 0.5rem;">Sistem Deteksi Otomatis Menggunakan Machine Learning (SVM)</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        st.error(f"⚠️ Model tidak ditemukan di: {path}")
        return None
    return joblib.load(path)

with st.spinner("Memuat model..."):
    model = load_model(MODEL_PATH)

if model is None:
    st.stop()

st.success("Model berhasil dimuat!")

def preprocess_image_to_hog(pil_img: Image.Image):
    img = np.array(pil_img)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    img_resized = cv2.resize(img, IMG_SIZE)
    img_float = img_resized.astype("float32") / 255.0

    fd = hog(
        img_float,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
        feature_vector=True
    )
    return fd

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def create_confidence_chart(classes, probs):
    colors = ['#667eea' if p == max(probs) else '#a8b3cf' for p in probs]

    fig = go.Figure(data=[
        go.Bar(
            x=[p * 100 for p in probs],
            y=classes,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Confidence Score per Kelas",
        xaxis_title="Confidence (%)",
        yaxis_title="Kelas Penyakit",
        height=max(300, len(classes) * 60),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_xaxes(range=[0, 100])

    return fig

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("Input Gambar")
    st.markdown('<div class="info-box"><strong>Tips:</strong> Pastikan daun memenuhi frame</div>', unsafe_allow_html=True)

    camera_image = st.camera_input("Ambil Foto dengan Kamera", label_visibility="collapsed")

    st.markdown("**— atau —**")
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

with col2:
    st.markdown("Hasil Prediksi")
    result_placeholder = st.empty()
    with result_placeholder.container():
        st.info("Ambil foto atau upload gambar, lalu tekan PREDICT")

st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("PREDICT", use_container_width=True)

if predict_button:
    pil_img = None

    if camera_image:
        pil_img = Image.open(camera_image)
    elif uploaded_file:
        pil_img = Image.open(uploaded_file)
    else:
        st.warning("Belum ada gambar.")
        st.stop()

    with col1:
        st.markdown("#### 🖼️ Gambar Input")
        st.image(pil_img, use_column_width=True)

    with col2:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("⏳ Menyimpan gambar...")
        progress_bar.progress(20)
        time.sleep(0.3)

        timestamp = int(time.time())
        save_path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(save_path)

        status_text.text("Ekstraksi fitur HOG...")
        progress_bar.progress(40)
        time.sleep(0.3)

        feat = preprocess_image_to_hog(pil_img).reshape(1, -1)
        
        status_text.text("Melakukan prediksi...")
        progress_bar.progress(70)
        time.sleep(0.3)

        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(feat)[0]
            idx = np.argmax(probs_all)
            label = model.classes_[idx]
            confidence = probs_all[idx] * 100

        else:
            scores = model.decision_function(feat)
            probs_all = softmax(scores[0])
            idx = np.argmax(probs_all)
            label = model.classes_[idx]
            confidence = probs_all[idx] * 100

        progress_bar.progress(100)
        time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        with result_placeholder.container():
            color_class = "success-card" if confidence >= 70 else "warning-card"
            st.markdown(f"""
            <div class="result-card {color_class}">
                <h2>Hasil Deteksi</h2>
                <h1>{label}</h1>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(model.classes_)}</div>
                    <div class="metric-label">Total Kelas</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                icon = "✅" if confidence >= 70 else "⚠️"
                stat = "Tinggi" if confidence >= 70 else "Rendah"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{icon}</div>
                    <div class="metric-label">Status: {stat}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("Detail Confidence Score")
            fig = create_confidence_chart(list(model.classes_), probs_all)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div class="info-box">
    <strong>Catatan:</strong> Model SVM harus dilatih dengan <code>probability=True</code> agar confidence lebih akurat.
</div>
""", unsafe_allow_html=True)
