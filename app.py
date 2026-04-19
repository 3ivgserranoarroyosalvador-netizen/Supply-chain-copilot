import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

#Página
st.set_page_config(
    page_title="Supply Chain Copilot",
    layout="wide"
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#Guardar la función
@st.cache_data
def load_data():
    return pd.read_csv('data/supply_chain_clean.csv')

st.sidebar.title("⚙️ Configuración")
uploaded_file = st.sidebar.file_uploader("📂 Cargar tu propio CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ {uploaded_file.name} ({len(df):,} filas)")
else:
    df = load_data()
    st.sidebar.info("📊 Usando dataset por defecto")

def get_dataset_summary():
    total = len(df)
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(include='object').columns.tolist()
    nulls = df.isnull().sum().sum()
    sample = df.head(3).to_string()

    summary = f"""
    Dataset con {total:,} filas y {len(cols)} columnas.
    - Columnas disponibles: {', '.join(cols)}
    - Columnas numéricas: {', '.join(numeric_cols)}
    - Columnas de texto: {', '.join(text_cols)}
    - Valores nulos totales: {nulls}
    - Muestra de los primeros 3 registros:
    {sample}
    """
    return summary
    # Info extra si es el dataset de supply chain
    if 'Late_delivery_risk' in df.columns:
        late = len(df[df['Late_delivery_risk'] == 1])
        late_pct = round(late / total * 100, 1)
        avg_delay = round(df['shipping_delay_days'].mean(), 2)
        worst_mode = df.groupby('Shipping Mode')['shipping_delay_days'].mean().idxmax()
        summary += f"""
    - Órdenes con riesgo de retraso: {late:,} ({late_pct}%)
    - Días promedio de retraso: {avg_delay}
    - Modo de envío con más retrasos: {worst_mode}
        """
    
    return summary

    return f"""
Dataset de Supply Chain con {total:,} órdenes reales.
-Órdenes con riesgo de retraso: {late:,}({late_pct}%)
-Días promedio de retraso: {avg_delay}
-Modo de envío con más retraso: {worst_mode}
-Categoría con más retrasos: {worst_cat}
-Mercado principal: {top_market}
-Modos de envío disponibles: {",".join(df["Shipping Mode"].unique())}
-Mercados:{",".join(df["Market"].unique())}
-Categorías de producto:{",".join(df["Category Name"].unique()[:10])}
"""

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

llm=get_llm()

def ask_copilot(question, chat_history):
    context = get_dataset_summary()
    messages = [
        SystemMessage(content=f"""Eres un Eres un experto en Supply Chain y análisis de datos.
                      Tienes acceso a datos reales de una cadena de suministro:
                      {context}
                      Responde siempre en español, de froma clara y profesional.
                      Si puede dar números concretos del dataset, hazlo.""")
    ]
    for msg in chat_history:
        messages.append(msg)
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content

st.title("Supply Chain Copilot")
st.caption("Asistente de IA para análisis de cadena de suministro")

total = len(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total filas", f"{total:,}")
with col2:
    st.metric("Total columnas", f"{len(df.columns)}")
with col3:
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        st.metric("Columnas numéricas", f"{len(numeric_cols)}")
    else:
        st.metric("Columnas numéricas", "0")
with col4:
    nulls = df.isnull().sum().sum()
    st.metric("Valores nulos", f"{nulls:,}")
st.divider()

st.subheader("💡 Preguntas sugeridas")
suggested = [
    "¿Cuál es el principal problema de esta supply chain?",
    "¿Qué categorías tienen más retrasos?",
    "¿Qué mercado es el más importante?",
    "¿Qué recomendaciones darías para reducir retrasos?"
]
cols = st.columns(4)
for i, q in enumerate(suggested):
    if cols[i].button(q, use_container_width=True):
        st.session_state.selected_question = q
st.divider()

#Chat
st.subheader("Chat con Copilot")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])

#Usuario
question = st.chat_input("Pregunta algo sobre tu supply chain...")

#Pregunta sugerida
if "selected_question" in st.session_state:
    question = st.session_state.selected_question
    del st.session_state.selected_question

#Pregunta en general
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            response = ask_copilot(question, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content":response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(HumanMessage(content=response))
    st.rerun()