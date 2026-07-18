import os
import json
import time
from datetime import datetime
 
import pandas as pd
import numpy as np
import faiss
import cohere
import gradio as gr
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# ── Configuracion ─────────────────────────────────────────────────────────────
load_dotenv()

COHERE_API_KEY  = os.environ.get("COHERE_API_KEY")
AUTH_USER	= os.environ.get("AUTH_USER", "admin")
AUTH_PASS	= os.environ.get("AUTH_PASS", "changeme")
CSV_PATH        = os.environ.get("CSV_PATH", "archivo.csv")
RUTA_EMBEDDINGS = os.environ.get("RUTA_EMBEDDINGS", "embeddings.npy")
RUTA_INDEX      = os.environ.get("RUTA_INDEX", "index.faiss")
LOG_PATH        = os.environ.get("LOG_PATH", "logs.json")


# ── Carga del CSV ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# ── Clientes Cohere ───────────────────────────────────────────────────────────
co  = cohere.Client(COHERE_API_KEY)
llm = ChatCohere(model="command-r-08-2024", temperature=0)


# ── Embeddings e índice FAISS ─────────────────────────────────────────────────
def embeddings_desactualizados() -> bool:
    """Devuelve True si el CSV es más nuevo que los embeddings guardados."""
    if not os.path.exists(RUTA_EMBEDDINGS) or not os.path.exists(RUTA_INDEX):
        return True  # no existen → hay que generarlos
    fecha_csv = os.path.getmtime(CSV_PATH)
    fecha_embeddings = os.path.getmtime(RUTA_EMBEDDINGS)
    return fecha_csv > fecha_embeddings  # CSV más nuevo → regenerar

if embeddings_desactualizados():

    df["texto_busqueda"] = (
        df["Descripción"].fillna("") + " " +
        df["Marca"].fillna("") + " " +
        df["Categoría"].fillna("") + " " +
        df["Subcategoría"].fillna("")
    ).str.strip()

    response = co.embed(
        texts=df["texto_busqueda"].tolist(),
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    embeddings = np.array(response.embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Guardar para la próxima sesión
    np.save(RUTA_EMBEDDINGS, embeddings)
    faiss.write_index(index, RUTA_INDEX)

    
else:
    embeddings = np.load(RUTA_EMBEDDINGS)
    index = faiss.read_index(RUTA_INDEX)

def buscar_productos(pregunta: str, top_k: int = 5) -> pd.DataFrame:
    """Embeddea solo la pregunta y busca las filas más similares en FAISS."""
    emb_pregunta = co.embed(
        texts=[pregunta],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    ).embeddings
    emb_pregunta = np.array(emb_pregunta).astype("float32")
    _, indices = index.search(emb_pregunta, top_k)
    columnas = [c for c in df.columns if c != "texto_busqueda"]  # ← definida acá adentro
    return df.iloc[indices[0]][columnas]



def guardar_log(pregunta, productos, respuesta, tiempo):
    entrada = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pregunta": pregunta,
        "productos_recuperados": productos,
        "respuesta": respuesta,
        "tiempo_respuesta_seg": round(tiempo, 2)
    }
    logs = []
    try:
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    except:
        pass
    logs.append(entrada)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

def preguntar_con_historial(pregunta: str, historial: list) -> tuple:
    inicio = time.time()

    filas = buscar_productos(pregunta, top_k=5)
    contexto = filas.to_string(index=False)
    productos_recuperados = filas["Descripción"].tolist()

    mensajes = [
        SystemMessage(content=(
            "Eres un asistente que analiza datos de un catálogo de productos. "
            "Respondé en español, de forma clara y concisa. "
            "Solo usá la información del contexto proporcionado. "
            "Si la respuesta no está en el contexto, decilo explícitamente. "
            "Tenés memoria de la conversación actual."
        ))
    ]

    for msg in historial:
        if msg["role"] == "user":
            mensajes.append(HumanMessage(content=msg["content"]))
        else:
            mensajes.append(SystemMessage(content=msg["content"]))

    mensajes.append(
        HumanMessage(content=f"Contexto (productos relevantes):\n{contexto}\n\nPregunta: {pregunta}")
    )

    respuesta = llm.invoke(mensajes).content
    tiempo = time.time() - inicio

    guardar_log(pregunta, productos_recuperados, respuesta, tiempo)

    historial.append({"role": "user", "content": pregunta})
    historial.append({"role": "assistant", "content": respuesta})
    return "", historial, historial.copy()

with gr.Blocks(title="Agente de Productos") as demo:
    gr.Markdown("## 🛒 Agente de Productos\nHacé preguntas sobre el catálogo en lenguaje natural.")

    chatbot = gr.Chatbot(label="Conversación", height=400)
    historial_state = gr.State([])

    with gr.Row():
        txt_input = gr.Textbox(
            placeholder="Ej: ¿Qué precio tiene el arroz integral?",
            label="Tu pregunta",
            scale=4
        )
        btn_enviar = gr.Button("Enviar", scale=1)

    btn_limpiar = gr.Button("🗑️ Limpiar conversación")

    txt_input.submit(
        preguntar_con_historial,
        inputs=[txt_input, historial_state],
        outputs=[txt_input, chatbot, historial_state]
    )
    btn_enviar.click(
        preguntar_con_historial,
        inputs=[txt_input, historial_state],
        outputs=[txt_input, chatbot, historial_state]
    )

    btn_limpiar.click(lambda: ([], []), outputs=[chatbot, historial_state])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
	auth=(AUTH_USER, AUTH_PASS)
    )
