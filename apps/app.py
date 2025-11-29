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

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Deteksi Penyakit Daun Padi", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css('apps/style.css')
except FileNotFoundError:
    st.warning("File style.css tidak ditemukan. Aplikasi akan berjalan tanpa styling custom.")

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
        st.info("💡 Pastikan kamu sudah menjalankan training dan menyimpan model di lokasi ini.")
        return None
    model = joblib.load(path)
    return model

def preprocess_image_to_hog(pil_img: Image.Image):
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

def create_confidence_chart(classes, probs):
    """Membuat bar chart interaktif untuk confidence scores"""
    colors = ['#667eea' if p == max(probs) else '#a8b3cf' for p in probs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[p*100 for p in probs],
            y=classes,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in probs],
            textposition='auto',
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
    
    fig.update_xaxes(range=[0, 100], gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    return fig

with st.spinner('Memuat model...'):
    model = load_model(MODEL_PATH)

if model is not None:
    st.success("Model berhasil dimuat!")
else:
    st.stop()
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("Input Gambar")
    st.markdown('<div class="info-box"><strong>Tips:</strong> Pastikan daun memenuhi frame dan pencahayaan cukup untuk hasil terbaik</div>', unsafe_allow_html=True)
    
    camera_image = st.camera_input("Ambil Foto dengan Kamera", label_visibility="collapsed")
    
    st.markdown("**— atau —**")
    
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg","jpeg","png"], label_visibility="collapsed")

with col2:
    st.markdown("Hasil Prediksi")
    result_placeholder = st.empty()
    
    with result_placeholder.container():
        st.info("Ambil foto atau upload gambar, lalu tekan tombol Predict")

st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_button = st.button("PREDICT", use_container_width=True)

if predict_button:
    pil_img = None
    if camera_image is not None:
        try:
            pil_img = Image.open(camera_image)
        except Exception as e:
            st.error(f"Gagal membaca gambar dari kamera: {str(e)}")
    elif uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file upload: {str(e)}")
    else:
        st.warning("Belum ada gambar: ambil foto dengan kamera atau upload gambar.")
    
    if pil_img is not None:
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
            
            try:
                feat = preprocess_image_to_hog(pil_img).reshape(1, -1)
            except Exception as e:
                st.error(f"Gagal preprocessing gambar (HOG): {str(e)}")
                feat = None
            
            if feat is not None:
                status_text.text("Melakukan prediksi...")
                progress_bar.progress(70)
                time.sleep(0.3)
                
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(feat)[0]
                        idx = np.argmax(probs)
                        label = model.classes_[idx]
                        confidence = probs[idx] * 100
                        probs_all = probs
                    else:
                        if hasattr(model, "decision_function"):
                            scores = model.decision_function(feat)
                            if scores.ndim == 1:
                                scores = np.vstack([-scores, scores]).T
                            probs = softmax(scores[0])
                            idx = np.argmax(probs)
                            label = model.classes_[idx]
                            confidence = probs[idx] * 100
                            probs_all = probs
                        else:
                            label = model.predict(feat)[0]
                            confidence = None
                            probs_all = None
                    
                    status_text.text("Selesai!")
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    with result_placeholder.container():
                        if confidence is not None:
                            confidence_color = "success-card" if confidence >= 70 else "warning-card"
                            st.markdown(f"""
                            <div class="result-card {confidence_color}">
                                <h2 style="margin: 0;">Hasil Deteksi</h2>
                                <h1 style="margin: 1rem 0; font-size: 2.5rem;">{label}</h1>
                                <p style="font-size: 1.3rem; margin: 0;">Confidence: {confidence:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{len(model.classes_)}</div>
                                    <div class="metric-label">Total Kelas</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{confidence:.1f}%</div>
                                    <div class="metric-label">Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col3:
                                status_emoji = "✅" if confidence >= 70 else "⚠️"
                                status_text_display = "Tinggi" if confidence >= 70 else "Rendah"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{status_emoji}</div>
                                    <div class="metric-label">Status: {status_text_display}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if probs_all is not None:
                                st.markdown("---")
                                st.markdown("Detail Confidence Score")
                                fig = create_confidence_chart(list(model.classes_), probs_all)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card success-card">
                                <h2 style="margin: 0;">Hasil Deteksi</h2>
                                <h1 style="margin: 1rem 0; font-size: 2.5rem;">{label}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Gagal melakukan prediksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <strong>ℹ Catatan Teknis:</strong><br>
    Jika model SVM dilatih tanpa <code>probability=True</code>, confidence yang ditampilkan adalah pseudo-probability 
    dari decision function. Untuk hasil yang lebih akurat, re-train model dengan <code>SVC(probability=True)</code>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>Dibuat dengan menggunakan Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)