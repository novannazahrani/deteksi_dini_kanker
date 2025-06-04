import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image


def equalize_hist_manual(channel):
    hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_normalized = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)
    return cdf_normalized[channel]

def segment_serviks_manual(img):
    hasil = np.zeros(img.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    red, green, blue = cv2.split(img)

    sat_f = sat.astype(np.float32) / 255.0
    val_f = val.astype(np.float32) / 255.0

    mask = (~((red > 238) & (green > 238) & (blue > 238))) & \
           (((hue < 10) | (hue > 160)) & ~((sat_f < 0.2) & (val_f < 0.95))) & \
           (val_f > 0.15)

    hasil[mask] = 255
    return hasil

def segment_lesi_manual(img, cluster_id):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    hue = hue.astype(np.float32) * 2  # konversi ke derajat 0-360

    if cluster_id == 0:
        mask1 = (hue >= 290) & (hue <= 360) & (sat >= 20) & (sat <= 100) & (val >= 180) & (val <= 240)
        mask2 = (hue >= 0) & (hue <= 100) & (sat >= 0) & (sat <= 80) & (val >= 250) & (val <= 255)
    elif cluster_id == 1:
        mask1 = (hue >= 290) & (hue <= 360) & (sat >= 70) & (sat <= 110) & (val >= 190) & (val <= 255)
        mask2 = (hue >= 0) & (hue <= 100) & (sat >= 0) & (sat <= 80) & (val >= 250) & (val <= 255)
    elif cluster_id == 2:
        mask1 = (hue >= 290) & (hue <= 360) & (sat >= 20) & (sat <= 70) & (val >= 220) & (val <= 250)
        mask2 = (hue >= 0) & (hue <= 100) & (sat >= 0) & (sat <= 40) & (val >= 250) & (val <= 255)
    else:
        mask1 = (hue >= 290) & (hue <= 360) & (sat >= 20) & (sat <= 100) & (val >= 180) & (val <= 255)
        mask2 = (hue >= 0) & (hue <= 50) & (sat >= 0) & (sat <= 130) & (val >= 240) & (val <= 255)

    mask = mask1 | mask2
    return (mask.astype(np.uint8) * 255)

def bitwise_and_manual(mask1, mask2):
    return ((mask1 > 0) & (mask2 > 0)).astype(np.uint8) * 255


st.title("Deteksi Dini Kanker Serviks")

st.markdown("""  
Aplikasi ini digunakan untuk mendeteksi dini kanker serviks yang dapat dilihat dari acetowhite ephitalium area yang ada di sekitar daerah serviks.  
""")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert ke HSV dan equalize V channel
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = equalize_hist_manual(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    # Segmentasi serviks
    mask_serviks = segment_serviks_manual(img_eq)

    # Hitung warna rata-rata HSV dari area serviks
    h_eq, s_eq, v_eq = cv2.split(hsv_eq)
    mask = mask_serviks > 0
    mean_color = np.array([[np.mean(h_eq[mask]), np.mean(s_eq[mask]), np.mean(v_eq[mask])]])

    # Load model KMeans (.joblib)
    kmeans = joblib.load("kmeans_model.joblib")

    # Prediksi cluster untuk gambar ini
    cluster_id = kmeans.predict(mean_color)[0]

    # Segmentasi lesi berdasarkan cluster
    lesi_mask = segment_lesi_manual(bgr, cluster_id)
    lesi_mask_area_serviks = bitwise_and_manual(lesi_mask, mask_serviks)

    # Overlay lesi pada gambar asli dengan warna hijau
    lesi_overlay = bgr.copy()
    lesi_overlay[lesi_mask_area_serviks > 0] = [0, 255, 0]

    st.image(cv2.cvtColor(lesi_overlay, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi Acetowhite Ephitalium Area ", use_column_width=True)
