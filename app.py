import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Configuración de la página
st.set_page_config(
    page_title="Supply Chain Copilot",
    layout="wide"
)

# Cargar variables de entorno
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Sidebar
st.sidebar.title("⚙️ Configuración")
uploaded_file = st.sidebar.file_uploader("📂 Cargar tu propio CSV", type="csv")

# Cargar dataset
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv('data/supply_chain_clean.csv')
    except FileNotFoundError:
        return None

# Determinar qué dataset usar
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {uploaded_file.name} ({len(df):,} filas)")
else:
    df = load_default_data()
    if df is not None:
        st.sidebar.info("📊 Usando dataset por defecto")

# Resumen del dataset para el LLM
def get_dataset_summary():
    total = len(df)
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(include='object').columns.tolist()
    nulls = df.isnull().sum().sum()
    sample = df.head(3).to_string()

    return f"""
    Dataset con {total:,} filas y {len(cols)} columnas.
    - Columnas disponibles: {', '.join(cols)}
    - Columnas numéricas: {', '.join(numeric_cols)}
    - Columnas de texto: {', '.join(text_cols)}
    - Valores nulos totales: {nulls}
    - Muestra de los primeros 3 registros:
    {sample}
    """

# Inicializar LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

llm = get_llm()

# Función del copilot
def ask_copilot(question, chat_history):
    context = get_dataset_summary()
    messages = [
        SystemMessage(content=f"""Eres un experto en análisis de datos y Supply Chain.
        Tienes acceso a los siguientes datos reales:
        {context}
        Responde siempre en español, de forma clara y profesional.
        Si puedes dar números concretos del dataset, hazlo.""")
    ]
    for msg in chat_history:
        messages.append(msg)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content

st.title("Supply Chain Copilot")
st.caption("Asistente de IA para análisis de datos — carga cualquier CSV y hazle preguntas")

# Si no hay dataset mostrar mensaje
if df is None:
    st.warning("Carga un archivo CSV desde el panel izquierdo para comenzar")
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)
numeric_cols = df.select_dtypes(include='number').columns
with col1:
    st.metric("Total filas", f"{len(df):,}")
with col2:
    st.metric("Total columnas", f"{len(df.columns)}")
with col3:
    st.metric("Columnas numéricas", f"{len(numeric_cols)}")
with col4:
    st.metric("Valores nulos", f"{df.isnull().sum().sum():,}")

st.divider()

# Preguntas sugeridas
st.subheader("💡 Preguntas sugeridas")
suggested = [
    "¿Cuál es el principal problema de este dataset?",
    "¿Qué columnas son más importantes?",
    "Dame un resumen ejecutivo de los datos",
    "¿Qué recomendaciones me darías basándote en estos datos?"
]
cols = st.columns(4)
for i, q in enumerate(suggested):
    if cols[i].button(q, use_container_width=True):
        st.session_state.selected_question = q

st.divider()

# Chat
st.subheader("💬 Chat con el Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Pregunta algo sobre tus datos...")

if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            response = ask_copilot(question, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(HumanMessage(content=response))
    st.rerun()