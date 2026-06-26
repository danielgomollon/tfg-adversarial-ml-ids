import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import time
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

# --- IMPORTAR TU ARQUITECTURA (Ajusta la ruta si es necesario) ---
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cnn import AdvancedNIDSModel 

# --- CONFIGURACIÓN DE RUTAS ---
MODEL_PATH = "outputs/models/ArcFace_Titan_PoC_best.pth" # <--- TU MODELO
DATA_PATH = "data/processed/demo_samples.npy"  # <--- TUS DATOS DE MUESTRA

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="TFG: IDS Adversarial Lab", page_icon="🛡️", layout="wide")

# Estilo CSS "Hacker" (El tuyo, que mola mucho)
st.markdown("""
    <style>
    .stApp {background-color: #0e1117;}
    .metric-card {background-color: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #374151;}
    h1, h2, h3 {color: #00ff00 !important; font-family: 'Courier New', Courier, monospace;}
    div[data-testid="stMetricValue"] {color: #ffffff;}
    .stButton>button {color: #000; background-color: #00ff00; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CARGA (MOTOR REAL) ---
@st.cache_resource
def load_model():
    # Instanciamos la arquitectura (52 features, 7 clases)
    model = AdvancedNIDSModel(num_features=52, num_classes=7)
    # Cargamos pesos en CPU
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        return np.load(DATA_PATH)
    except FileNotFoundError:
        return None

# --- CARGAR RECURSOS ---
model = load_model()
data_samples = load_data()

# --- CABECERA ---
st.title("🛡️ IDS DEFENSE SYSTEM // ADVERSARIAL LAB")
st.markdown("### Simulación de Ataques de Evasión en Tiempo Real (Motor PyTorch + ART)")

if model is None or data_samples is None:
    st.error("⚠️ ERROR CRÍTICO: No encuentro 'outputs/models/ArcFace_Titan_PoC_best.pth' o 'data/processed/demo_samples.npy'. Descárgalos de Drive.")
    st.stop()

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("💀 Panel de Control")
    
    # Seleccionar muestra real
    sample_id = st.number_input("ID del Flujo de Red", 0, len(data_samples)-1, 0)
    
    attack_type = st.selectbox("Vector de Ataque", ["Ninguno", "FGSM (Rápido)", "PGD (Potente)"])
    
    st.subheader("Parámetros del Exploit")
    epsilon = st.slider("Magnitud del Ruido (Epsilon)", 0.0, 0.5, 0.1, 0.01)
    
    steps = 10
    if attack_type == "PGD":
        steps = st.slider("Iteraciones PGD", 1, 20, 10)
    
    launch_btn = st.button("EJECUTAR ATAQUE 🚀", type="primary")

# --- LÓGICA PRINCIPAL ---

# 1. Preparar datos reales
original_flow = data_samples[sample_id].reshape(1, 52).astype(np.float32)
tensor_orig = torch.FloatTensor(original_flow)

# 2. Predicción Inicial (Real)
with torch.no_grad():
    logits = model(tensor_orig)
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_label_orig = np.argmax(probs)
    confidence_orig = probs[pred_label_orig]

# Variables para visualización (por defecto, estado limpio)
current_conf = confidence_orig
current_status = "TRÁFICO ANALIZADO"
color_status = "gray"
display_flow = original_flow
diff = np.zeros(52)

# 3. Ejecución del Ataque (Si se pulsa el botón)
if launch_btn and attack_type != "Ninguno":
    with st.spinner(f'Calculando gradientes ({attack_type})...'):
        # Envolver modelo para ART
        classifier = PyTorchClassifier(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            input_shape=(1, 52),
            nb_classes=7,
            clip_values=(np.min(data_samples), np.max(data_samples))
        )
        
        # Generar Ataque Real
        if attack_type == "FGSM":
            attacker = FastGradientMethod(estimator=classifier, eps=epsilon)
        elif attack_type == "PGD":
            attacker = ProjectedGradientDescent(estimator=classifier, eps=epsilon, max_iter=steps, verbose=False)
            
        adv_flow = attacker.generate(x=original_flow)
        
        # Predicción sobre el dato hackeado
        tensor_adv = torch.FloatTensor(adv_flow)
        with torch.no_grad():
            logits_adv = model(tensor_adv)
            probs_adv = torch.softmax(logits_adv, dim=1).numpy()[0]
            pred_label_adv = np.argmax(probs_adv)
            current_conf = probs_adv[pred_label_adv] # Confianza de la nueva clase
        
        # Calcular diferencias para graficar
        display_flow = adv_flow
        diff = adv_flow - original_flow
        
        # Lógica de Evasión
        # Asumimos que la clase original era la correcta (ej: Maligno). 
        # Si la etiqueta cambia, hubo evasión.
        if pred_label_adv != pred_label_orig:
            current_status = f"EVASIÓN EXITOSA (Clase {pred_label_adv})"
            color_status = "#00ff00" # Verde Hacker
        else:
            current_status = "ATAQUE RESISTIDO (Bloqueado)"
            color_status = "orange"
else:
    if confidence_orig > 0.5: # Asumiendo que detectó algo
        current_status = "DETECTADO (Original)"
        color_status = "red"
    else:
        current_status = "BENIGNO (Original)"
        color_status = "#00ff00"

# --- VISUALIZACIÓN ---

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Estado del IDS")
    
    # Velocímetro (Gauge) - Ahora con datos REALES
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_conf * 100,
        title = {'text': f"Confianza (Clase {np.argmax(probs)})"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color_status},
            'steps': [
                {'range': [0, 50], 'color': "gray"},
                {'range': [50, 100], 'color': "#262626"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(f"<h3 style='text-align: center; color: {color_status} !important;'>{current_status}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Forense: Perturbación Inyectada")
    
    # Gráfico de la perturbación REAL
    # Mostramos solo las features con cambios
    features_idx = [f"F{i}" for i in range(52)]
    
    if np.sum(np.abs(diff)) > 0:
        fig_bar = go.Figure(data=[
            go.Bar(name='Ruido Inyectado', x=features_idx, y=diff.flatten(), marker_color='#00ff00')
        ])
        fig_bar.update_layout(
            title=f"Gradiente Adversario (Epsilon={epsilon})",
            yaxis_title="Cambio de Valor",
            xaxis_title="Features (0-51)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.warning(f"⚠️ El ataque ha inyectado ruido matemático real en el flujo #{sample_id}.")
    else:
        st.info("Esperando ejecución del ataque... (Selecciona FGSM o PGD)")
        # Placeholder vacío bonito
        st.plotly_chart(go.Figure(), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# pie de págiona
st.markdown("---")
st.caption("TFG Ciberseguridad & IA | Ejecución REAL de ataques de evasión ejecutando en CPU local usando PyTorch & ART")