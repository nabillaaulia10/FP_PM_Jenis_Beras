import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# =====================
# LOAD MODEL
# =====================

model = YOLO(
    "runs/classify/hasil_training/beras_cls/weights/best.pt"
)


# =====================
# HALAMAN
# =====================

st.title("Klasifikasi Jenis Beras Menggunakan YOLOv11")

st.write(
    "Upload gambar beras untuk melakukan prediksi jenis beras"
)


# =====================
# UPLOAD GAMBAR
# =====================

file = st.file_uploader(
    "Upload gambar beras",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if file:


    # tampilkan gambar
    image = Image.open(file)

    st.image(
        image,
        caption="Gambar Input",
        width=400
    )


    # ubah ke array
    img_array = np.array(image)


    # =====================
    # PREDIKSI
    # =====================

    hasil = model.predict(
        img_array,
        imgsz=224
    )


    for h in hasil:


        kelas = h.names[
            h.probs.top1
        ]


        confidence = h.probs.top1conf.item()



        st.success(
            f"Jenis Beras : {kelas}"
        )


        st.info(
            f"Confidence : {confidence*100:.2f}%"
        )