# app.py (versión sin OpenAI; embeddings locales con sentence-transformers)
import os
import time
import json
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
load_dotenv()

# Sentence Transformers (embeddings locales)
from sentence_transformers import SentenceTransformer

# Config
EMBED_MODEL_LOCAL = "all-MiniLM-L6-v2"   # ligero y efectivo
COMPLETION_STYLE = "template"            # indicativo: usamos plantilla
INDEX_FILE = "faiss.index"
DOCS_META_FILE = "docs_meta.json"
LOG_FILE = "chat_logs.jsonl"
DEFAULT_BATCH_SIZE = 128

# Load local SBERT model (download la primera vez)
@st.cache_resource(show_spinner=False)
def load_sbert_model(model_name: str = EMBED_MODEL_LOCAL):
    return SentenceTransformer(model_name)

sbert_model = load_sbert_model()

# ---------- Utilities ----------
def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    # Aproximación simple: cortar por caracteres (4 chars ~ 1 token)
    approx_char = max_tokens * 4
    text = text.replace("\r\n", "\n")
    return [text[i:i+approx_char] for i in range(0, len(text), approx_char)]

def save_index(index: faiss.IndexFlatIP, meta: List[dict]):
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index() -> Tuple[Optional[faiss.IndexFlatIP], List[dict]]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_META_FILE):
        return None, []
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def log_interaction(record: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------- Local embeddings (sentence-transformers) with batching ----------
def get_embedding_local(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE, st_progress: Optional[st.delta_generator] = None) -> List[List[float]]:
    """
    Devuelve embeddings (listas de floats) usando sentence-transformers.
    """
    embeddings = []
    total = len(texts)
    if total == 0:
        return embeddings

    progress = st_progress if st_progress is not None else (st.progress(0) if "st" in globals() else None)

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        embs = sbert_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        # convertir a lista por compatibilidad con FAISS + json
        embeddings.extend([e.tolist() for e in embs])
        if progress:
            progress.progress(min(1.0, (i + len(batch)) / total))
    if progress:
        progress.progress(1.0)
    return embeddings

# ---------- Index building/search (FAISS) ----------
def build_index_from_dataframe(df: pd.DataFrame, text_column: str, batch_size: int = DEFAULT_BATCH_SIZE):
    docs = []
    meta = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append(c)
            meta.append({
                "source": f"row_{idx}",
                "row_index": int(idx),
                "chunk_id": i,
                "orig_preview": c[:400]
            })

    st.info(f"Generando embeddings locales para {len(docs)} fragmentos... (esto puede tardar)")
    progress_bar = st.progress(0)
    embeddings = get_embedding_local(docs, batch_size=batch_size, st_progress=progress_bar)

    if not embeddings or len(embeddings) != len(docs):
        st.error("Embeddings fallaron o no se generaron todos los embeddings.")
        return None, []

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    arr = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(arr)
    index.add(arr)
    save_index(index, meta)
    st.success("Índice local creado y guardado.")
    return index, meta

def search_index(index: faiss.IndexFlatIP, query: str, meta: List[dict], k: int = 4):
    try:
        q_emb = get_embedding_local([query], batch_size=1, st_progress=None)
        if not q_emb:
            return []
        q = np.array(q_emb).astype('float32')
        faiss.normalize_L2(q)
        D, I = index.search(q, k)
        results = []
        for idx in I[0]:
            if idx < len(meta):
                results.append(meta[idx])
        return results
    except Exception as e:
        st.error(f"Error en búsqueda del índice: {e}")
        return []

# ---------- Simple template-based answer generator ----------
def generate_answer_template(user_question: str, context_snippets: List[str]) -> str:
    """
    Generador simple: resume los snippets y sugiere acciones heurísticas.
    Útil cuando no se dispone de un LLM externo.
    """
    if not context_snippets:
        return ("No se encontró información relevante en los documentos indexados. "
                "Sugerencias:\n- Subir más datos/contexto.\n- Probar reformular la pregunta.\n- Hacer seguimiento con encuestas directas.")
    # Hacer un pequeño 'resumen' concatenando previews (limitar longitud)
    resumen = "\n\n".join([f"- {s[:400].strip()}" for s in context_snippets[:5]])
    sugerencias = (
        "\n\nSugerencias heurísticas para oportunidades de mejora:\n"
        "1) Priorizar problemas que se repitan en los fragmentos.\n"
        "2) Investigar la causa raíz (revisar logs / tickets asociados).\n"
        "3) Probar una solución de baja inversión (MVP) para validar hipótesis.\n"
        "4) Recopilar métricas antes/después (NPS, tasa de conversión, errores).\n"
    )
    retorno = f"Contexto relevante (fragmentos):\n{resumen}\n\n{sugerencias}"
    # Añadir nota de confidencialidad
    retorno += ("\nNota: Esta respuesta fue generada con un motor local basado en plantillas. "
                "Si quieres respuestas más creativas o detalladas, conecta un LLM (por ejemplo OpenAI).")
    return retorno

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chatbot Mejora Producto (Local RAG)", layout="centered")
st.title("Chatbot (RAG) — Embeddings locales y respuestas por plantilla")

st.sidebar.markdown("### Opciones")
tab = st.sidebar.radio("Navegación", ["Chat", "Indexar datos", "Logs y feedback", "Instrucciones", "Configuración"])

if tab == "Instrucciones":
    st.header("Instrucciones rápidas")
    st.markdown("""
- Esta versión usa **embeddings locales** con `sentence-transformers` y FAISS.
- No requiere clave de OpenAI.
- Flujo:
  1. Subir CSV/Excel con una columna de texto (comentarios/FAQ/etc).
  2. Indexar desde la pestaña *Indexar datos*.
  3. Ir a *Chat* y hacer preguntas; el sistema devolverá fragmentos relevantes + sugerencias heurísticas.
""")
    st.markdown("Archivos locales creados: `faiss.index`, `docs_meta.json`, `chat_logs.jsonl`.")

elif tab == "Configuración":
    st.header("Configuración")
    st.markdown("Modelos y parámetros (ajusta si lo deseas)")
    st.text_input("Modelo de embeddings local", value=EMBED_MODEL_LOCAL, disabled=True)
    batch_size_cfg = st.number_input("Batch size para embeddings", min_value=8, max_value=1024, value=DEFAULT_BATCH_SIZE)

elif tab == "Indexar datos":
    st.header("Construir/Actualizar índice (local)")
    uploaded = st.file_uploader("Sube un CSV o Excel con textos (comentarios, FAQ, etc.)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.type == "text/csv" or uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.write("Vista previa:", df.head())
            col = st.selectbox("Columna de texto para indexar", options=list(df.columns))
            batch_size = st.number_input("Batch size para embeddings", min_value=8, max_value=1024, value=batch_size_cfg)
            if st.button("Construir índice desde este archivo"):
                with st.spinner("Indexando (embeddings locales)..."):
                    index, meta = build_index_from_dataframe(df, col, batch_size=batch_size)
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
    else:
        st.info("Sube un archivo para indexar. Si ya tienes un índice guardado, no es necesario subir uno nuevo.")

elif tab == "Chat":
    st.header("Chat")
    st.info("El sistema recupera fragmentos relevantes usando FAISS + embeddings locales y genera una respuesta basada en plantilla.")
    index, meta = load_index()
    if index is None:
        st.warning("No hay índice guardado. Ve a 'Indexar datos' y sube tus documentos.")
    user_input = st.chat_input("Escribe la pregunta del cliente (ej. '¿Qué mejoras sugieres para el onboarding?')")

    if user_input:
        if index is None:
            st.error("No se puede responder sin índice. Indexa tus datos primero.")
        else:
            with st.spinner("Buscando contexto..."):
                results_meta = search_index(index, user_input, meta, k=6)
                snippets = [r["orig_preview"] for r in results_meta] if results_meta else []
                answer = generate_answer_template(user_input, snippets)
                st.chat_message("user").write(user_input)
                st.chat_message("assistant").write(answer)

                # feedback UI
                col1, col2, col3 = st.columns([1,1,2])
                useful_selection = None
                with col1:
                    if st.button("👍 Útil"):
                        useful_selection = True
                with col2:
                    if st.button("👎 No útil"):
                        useful_selection = False
                with col3:
                    feedback_txt = st.text_input("Comentario adicional (opcional)")

                record = {
                    "timestamp": time.time(),
                    "question": user_input,
                    "answer": answer,
                    "context_snippets": snippets,
                    "useful": useful_selection,
                    "feedback": feedback_txt
                }
                log_interaction(record)
                st.success("Interacción registrada.")

elif tab == "Logs y feedback":
    st.header("Logs de conversaciones")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-200:]
        records = [json.loads(l) for l in lines]
        df_logs = pd.DataFrame(records)
        st.dataframe(df_logs)
        st.markdown("**Descargar logs**")
        st.download_button("Descargar logs (JSONL)", data="".join(lines), file_name="chat_logs.jsonl", mime="text/plain")
    else:
        st.info("No hay logs aún. Interactúa con el chatbot para generar datos.")

