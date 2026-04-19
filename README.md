# Supply Chain Copilot
Asistente de IA para análisis de cadena de suministro usando RAG + LLMs.
# Funciones
- Responde preguntas en lenguaje natural sobre datos de supply chain
- Detecta patrones de retraso y riesgo operacional
- Funciona con cualquier dataset CSV
- Genera insights ejecutivos automáticamente
# Herramientas
- **Python** — procesamiento de datos
- **Pandas** — limpieza y análisis del dataset
- **LangChain** — orquestación del pipeline RAG
- **Groq (Llama 3.1)** — modelo de lenguaje
- **ChromaDB** — vector database
- **Streamlit** — interfaz web
# Dataset
[DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) — 180,519 órdenes reales
# Instalación local
```bash
git clone https://github.com/3ivgserranoarroyosalvador-netizen/Supply-chain-copilot.git
cd supply-chain-copilot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
# Ejecutar
```bash
streamlit run app.py
```
# Limitaciones
El LLM no accede al CSV directamente: Copilot recibe un resumen de texto del dataset, es decir no responde con exactitud.
Sin memoria entre sesiones: se pierde la memoria al momento de cerrar y volver abrir el navegador
Depende de internet
Límite de tokens
# Preview
![Supply Chain Copilot](https://i.imgur.com/placeholder.png)
