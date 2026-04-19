import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

df = pd.read_csv('data/supply_chain_clean.csv')

def get_dataset_summary():
    total = len(df)
    late = len(df[df["Late_delivery_risk"] == 1])
    late_pct = round(late/total * 100,1)
    avg_delay = round(df["shipping_delay_days"].mean(),2)
    worst_mode = df.groupby("Shipping Mode")["shipping_delay_days"].mean().idxmax()
    worst_cat = df.groupby("Category Name")["shipping_delay_days"].mean().idxmax()
    top_market = df["Market"].value_counts().index[0]

    return f"""
Resumen del dataset de Supply Chain:
-Total de órdenes: {total:,}
-Órdenes con riesgo de retraso: {late:,} ({late_pct}%)
-Días promedio de retraso: {avg_delay}
-Modo de envío con más retrasos: {worst_mode}
-Categoría con más retrasos: {worst_cat}
-Mercado principal: {top_market}
-Columna disponibles: {",".join(df.columns.tolist())}
"""

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
)

def ask_copilot(question):
    context = get_dataset_summary()
    messages = [
        SystemMessage(content=f"""Eres un asistente experto en Supply Chain...
                      {context}
                      ..."""),
                      HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content

#IA
print("Supply Chain Copilot iniciado\n")
print("Pregunta 1:", "¿Cuál es el principal problema de esta supply chain?")
print("Respuesta:", ask_copilot("¿Cuál es el principal problema de esta supply chain?"))
print("\n" + "="*60 + "\n")
print("Pregunta 2:", "¿Qué modo de envío debería evitar la empresa?")
print("Respuesta:", ask_copilot("¿Qué modo de envío debería evitar la empresa?"))
if GROQ_API_KEY:
    print("API cargada correctamente")
else:
    print("No se encontró la API")