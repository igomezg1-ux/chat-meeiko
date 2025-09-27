# app.py
import os
import time
import json
import math
import hashlib
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
load_dotenv()

# OpenAI modern client
from openai import OpenAI

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"   # ajusta si necesitas otro
COMPLETION_MODEL = "gpt-4o-mini"         # cambia si no lo soportas (ej: "gpt-3.5-turbo")
INDEX_FILE = "faiss.index"
DOCS_META_FILE = "docs_meta.json"
LOG_FILE = "chat_logs.jsonl"
DEFAULT_BATCH_SIZE = 256

# If user didn't set key in env, allow pasting in UI (session-only)
if not OPENAI_API_KEY:
    st.warning("No se encontró OPENAI_API_KEY. Puedes ponerla en un archivo .env o pegarla a continuación (solo en esta sesión).")
    secret_key = st.text_input("Pega tu OpenAI API key (se mantendrá solo en la sesión)", type="password")
    if secret_key:
        OPENAI_API_KEY = secret_key

if not OPENAI_API_KEY:
    st.error("Falta la API key. Define OPENAI_API_KEY en .env o pégala arriba.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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

# ---------- OpenAI Embeddings with batching and retries ----------
def _call_embeddings_with_retry(batch: List[str], max_retries: int = 5, backoff_base: float = 1.0):
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            # resp.data is a list of embedding objects
            return [r.embedding for r in resp.data]
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep = backoff_base * (2 ** (attempt - 1))
            time.sleep(sleep)

def get_embedding(texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE, st_progress: Optional[st.delta_generator] = None) -> List[List[float]]:
    embeddings: List[List[float]] = []
    total = len(texts)
    if total == 0:
        return embeddings

    progress = st_progress if st_progress is not None else (st.progress(0) if "st" in globals() else None)

    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_emb = _call_embeddings_with_retry(batch)
            embeddings.extend(batch_emb)
        except Exception as e:
            # bubble error to UI
            st.error(f"Fallo en embeddings (batch {i}..{i+len(batch)-1}): {e}")
            return []
        if progress:
            progress.progress(min(1.0, (i + len(batch)) / total))
    if progress:
        progress.progress(1.0)
    return embeddings

# ---------- Index building/search ----------
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
                "orig_preview": c[:200]
            })

    st.info(f"Generando embeddings para {len(docs)} fragmentos... (esto puede tardar)")
    progress_bar = st.progress(0)
    embeddings = get_embedding(docs, batch_size=batch_size, st_progress=progress_bar)

    if not embeddings or len(embeddings) != len(docs):
        st.error("Embeddings fallaron o no se generaron todos los embeddings.")
        return None, []

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim)
    arr = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(arr)
    index.add(arr)
    save_index(index, meta)
    st.success("Índice creado y guardado.")
    return index, meta

def search_index(index: faiss.IndexFlatIP, query: str, meta: List[dict], k: int = 4):
    try:
        q_emb = get_embedding([query], batch_size=1, st_progress=None)
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

# ---------- Chat / Generation ----------
def generate_answer(system_prompt: str, user_question: str, context_snippets: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_snippets)
    prompt = f"""
Eres un asistente para clientes cuya tarea es ayudar a identificar oportunidades de mejora de productos y servicios.
- Usa únicamente la información provista en CONTEXTO cuando sea posible.
- Si debes inferir algo, dilo claramente.
- Si la pregunta no está en el contexto, responde de manera útil y sugiere pasos para obtener más datos.

CONTEXTO:
{context_text}

PREGUNTA DEL USUARIO:
{user_question}

RESPONDE:
"""
    try:
        resp = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        # resp.choices is a list
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error en generación: {e}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chatbot Mejora Producto (RAG)", layout="centered")
st.title("Chatbot para identificar oportunidades de mejora — Demo")

tab = st.sidebar.radio("Navegación", ["Chat", "Indexar datos", "Logs y feedback", "Instrucciones", "Configuración"])

if tab == "Instrucciones":
    st.header("Instrucciones rápidas")
    st.markdown("""
1. Subir CSV/Excel con columna de texto (p.ej. `descripcion`, `comentarios_usuarios`).
2. Ir a *Indexar datos* y elegir la columna. Click en *Construir índice*.
3. En *Chat*, escribir preguntas. Las interacciones se guardan en logs.
4. Pedir a testers que prueben y dejen feedback.
""")
    st.markdown("Archivos locales generados: `faiss.index`, `docs_meta.json`, `chat_logs.jsonl`.")

elif tab == "Configuración":
    st.header("Configuración")
    st.markdown("Modelos y parámetros (cambia si tu cuenta no soporta alguno)")
    st.text_input("EMBED_MODEL", value=EMBED_MODEL, disabled=True)
    st.text_input("COMPLETION_MODEL", value=COMPLETION_MODEL, disabled=True)
    st.number_input("Batch size para embeddings", min_value=16, max_value=1024, value=DEFAULT_BATCH_SIZE)

elif tab == "Indexar datos":
    st.header("Construir/Actualizar índice")
    uploaded = st.file_uploader("Sube un CSV o Excel con textos (comentarios, FAQ, etc.)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.type == "text/csv" or uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.write("Vista previa:", df.head())
            col = st.selectbox("Columna de texto para indexar", options=list(df.columns))
            batch_size = st.number_input("Batch size para embeddings", min_value=32, max_value=1024, value=DEFAULT_BATCH_SIZE)
            if st.button("Construir índice desde este archivo"):
                with st.spinner("Indexando..."):
                    index, meta = build_index_from_dataframe(df, col, batch_size=batch_size)
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
    else:
        st.info("Sube un archivo para indexar. Si ya tienes un índice guardado, no es necesario subir.")

elif tab == "Chat":
    st.header("Chat")
    system_prompt = st.text_area("Prompt de sistema (instrucciones al modelo)", value="Eres un asistente que ayuda a identificar oportunidades de mejora de productos/servicios. Sé claro, conciso y orientado al cliente.", height=120)

    index, meta = load_index()
    if index is None:
        st.warning("No hay índice guardado. Ve a 'Indexar datos' y sube tus documentos.")
    user_input = st.chat_input("Escribe la pregunta del cliente (p. ej. '¿Qué mejoras sugieres para el onboarding?')")

    if user_input:
        if index is None:
            st.error("No se puede responder sin índice. Indexa tus datos primero.")
        else:
            with st.spinner("Buscando contexto y generando respuesta..."):
                results_meta = search_index(index, user_input, meta, k=4)
                snippets = [r["orig_preview"] for r in results_meta] if results_meta else []
                answer = generate_answer(system_prompt, user_input, snippets)
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
