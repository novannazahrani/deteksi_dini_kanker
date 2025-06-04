import streamlit as st
import cv2
import numpy as np
import joblib

def equalize_hist_manual(channel):
    hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_mask = cdf != 0
    cdf_min = cdf[cdf_mask].min()
    cdf_max = cdf[cdf_mask].max()
    cdf_normalized = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)
    return cdf_normalized[channel]

def erosi(mask, kernel):
    output = np.zeros_like(mask)
    pad = kernel.shape[0] // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def dilasi(mask, kernel):
    output = np.zeros_like(mask)
    pad = kernel.shape[0] // 2
    padded = np.pad(mask, pad, mode='constant', constant_values=0)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def opening(mask, kernel):
    return dilasi(erosi(mask, kernel), kernel)

def closing(mask, kernel):
    return erosi(dilasi(mask, kernel), kernel)

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

    # Kernel bulat 5x5
    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ], dtype=np.uint8)

    hasil = opening(hasil, kernel)
    hasil = closing(hasil, kernel)
    return hasil

def segment_lesi_manual(img, cluster_id):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    hue = hue.astype(np.float32) * 2

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

    mask = (mask1 | mask2).astype(np.uint8) * 255
    return mask

def bitwise_and_manual(mask1, mask2):
    return ((mask1 > 0) & (mask2 > 0)).astype(np.uint8) * 255

st.title("👩🏻‍⚕ Deteksi Dini Kanker Serviks 👩🏻‍⚕")
st.markdown("""
Aplikasi ini mendeteksi **acetowhite ephitelium area** dari citra serviks menggunakan segmentasi manual dan clustering.
""")

uploaded_file = st.file_uploader("📤 Upload gambar serviks", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = equalize_hist_manual(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    mask_serviks = segment_serviks_manual(img_eq)

    h_eq, s_eq, v_eq = cv2.split(hsv_eq)
    mask = mask_serviks > 0
    mean_color = np.array([[np.mean(h_eq[mask]), np.mean(s_eq[mask]), np.mean(v_eq[mask])]])

    kmeans = joblib.load("kmeans_model.joblib")
    cluster_id = kmeans.predict(mean_color)[0]

    mask_lesi = segment_lesi_manual(bgr, cluster_id)
    lesi_akhir = bitwise_and_manual(mask_lesi, mask_serviks)

    hasil_overlay = bgr.copy()
    hasil_overlay[lesi_akhir > 0] = [0, 255, 0]

    st.image(cv2.cvtColor(hasil_overlay, cv2.COLOR_BGR2RGB),
             caption="Hasil Deteksi Acetowhite Ephitelium Area",
             use_column_width=True)
