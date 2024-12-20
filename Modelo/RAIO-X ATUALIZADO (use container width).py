from PIL import Image
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuração da página do Streamlit
st.set_page_config(
    page_title=" (PROTOTIPO) Classificação de Raios-X", layout="centered")

# Título da aplicação
st.title("Classificação de Raios-X - Tuberculose ou Normal")

# Carrega o modelo treinado
@st.cache_resource
def load_trained_model():
    return load_model('modelo_raiox.h5')


model = load_trained_model()

# Função de pré-processamento
def preprocess_input(image_file, target_size=(96, 96)):
    img = Image.open(image_file).convert("RGB").resize(target_size)
    img_array = np.asarray(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Índices das classes
class_indices = {0: 'Normal', 1: 'Tuberculose'}

# Carregar imagem do usuário
uploaded_file = st.file_uploader(
    "Envie uma imagem de raio-x no formato PNG ou JPG", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Exibe a imagem carregada
    st.image(uploaded_file, caption="Imagem Carregada", use_container_width=True)  # Alterado aqui

    # Processa a imagem
    with st.spinner("Processando a imagem..."):
        preprocessed_image = preprocess_input(uploaded_file)
        predictions = model.predict(preprocessed_image)
        predicted_class = class_indices[np.argmax(predictions[0])]

    # Mostra o resultado da previsão
    st.subheader(f"A classe prevista é: **{predicted_class}**")
    st.bar_chart(predictions[0])
else:
    st.info("Por favor, carregue uma imagem para começar a classificação.")
