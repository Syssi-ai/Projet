import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

st.set_page_config(page_title="Classification d'images - MobileNetV2", page_icon="📷")

st.title("📷 Classification d'images avec MobileNetV2")
st.write(
    "Prends une photo avec ta caméra ou upload une image, "
    "et le modèle te dira ce qu'il reconnaît."
)


@st.cache_resource
def load_my_model(path="mobilenetv2.keras"):
    return load_model(path)


def predict_image(model, pil_image, top_k=3):
    # Convertit l'image PIL en RGB puis la redimensionne (remplace cv2.resize)
    resized = pil_image.convert("RGB").resize((224, 224))
    frame = np.array(resized)

    x = np.expand_dims(frame.astype(np.float32), axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    return decode_predictions(preds, top=top_k)[0]


# Chargement du modèle (mis en cache pour éviter de le recharger à chaque interaction)
with st.spinner("Chargement du modèle..."):
    try:
        model = load_my_model()
        st.success("Modèle chargé avec succès ✅")
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()

st.divider()

# Choix de la source de l'image
source = st.radio("Source de l'image :", ["📷 Caméra", "📁 Upload"], horizontal=True)

image = None

if source == "📷 Caméra":
    camera_photo = st.camera_input("Prends une photo")
    if camera_photo is not None:
        image = Image.open(camera_photo)
else:
    uploaded_file = st.file_uploader(
        "Choisis une image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

if image is not None:
    st.image(image, caption="Image sélectionnée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        predictions = predict_image(model, image, top_k=5)

    st.subheader("Résultats de la prédiction")
    for _, label, prob in predictions:
        label_clean = label.replace("_", " ").title()
        st.write(f"**{label_clean}** — {prob:.2%}")
        st.progress(float(prob))
