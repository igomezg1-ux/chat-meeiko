
import os
import streamlit as st
from typing import List, Tuple
import pandas as pd
import json
import time

# OpenAI imports
import openai
# ✅ nuevo
from openai import OpenAIError


# FAISS + utilities
import faiss
import numpy as np

# For text splitting
import tiktoken
import math
import hashlib

# ---------- CONFIG ----------
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini"
INDEX_FILE = "faiss.index"
DOCS_META_FILE = "docs_meta.json"
LOG_FILE = "chat_logs.jsonl"
# ----------------------------

if not OPENAI_API_KEY:
    st.warning("No se encontró OPENAI_API_KEY. Puedes ponerlo en un .env o pegarlo aquí.")
    secret_key = st.text_input("Pega tu OpenAI API key", type="password")
    if secret_key:
        OPENAI_API_KEY = secret_key

if not OPENAI_API_KEY:
    st.error("Falta la API key. Define OPENAI_API_KEY antes de ejecutar.")
    st.stop()

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Utilities ----------
def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    # simple chunker by characters approximating tokens
    approx_char = max_tokens * 4
    return [text[i:i+approx_char] for i in range(0, len(text), approx_char)]

def get_embedding(texts: List[str]) -> List[List[float]]:
    try:
        resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
        return [r["embedding"] for r in resp["data"]]
    except OpenAIError as e:
        st.error(f"Error embedding: {e}")
        return []

def save_index(index: faiss.IndexFlatIP, meta: List[dict]):
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index() -> Tuple[faiss.IndexFlatIP, List[dict]]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_META_FILE):
        return None, []
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def build_index_from_dataframe(df: pd.DataFrame, text_column: str):
    # Flatten text from dataframe rows into documents
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
    st.info(f"Generando embeddings para {len(docs)} fragmentos... (esto tarda)")
    embeddings = get_embedding(docs)
    if not embeddings:
        st.error("Embeddings fallaron.")
        return None, []
    dim = len(embeddings[0])
    # FAISS index (inner product over L2-normalized vectors = cosine)
    index = faiss.IndexFlatIP(dim)
    arr = np.array(embeddings).astype('float32')
    # normalize
    faiss.normalize_L2(arr)
    index.add(arr)
    save_index(index, meta)
    st.success("Index creado y guardado.")
    return index, meta

def search_index(index: faiss.IndexFlatIP, query: str, meta: List[dict], k: int = 4):
    emb = get_embedding([query])
    if not emb:
        return []
    q = np.array(emb).astype('float32')
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results

def generate_answer(system_prompt: str, user_question: str, context_snippets: List[str]) -> str:
    # Construye prompt y llama a OpenAI Completion
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
        resp = openai.ChatCompletion.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return resp["choices"][0]["message"]["content"].strip()
    except OpenAIError as e:
        return f"Error en generación: {e}"

def log_interaction(record: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chatbot Mejora Producto", layout="centered")
st.title("Chatbot para identificar oportunidades de mejora — Demo")

tab = st.sidebar.radio("Navegación", ["Chat", "Indexar datos", "Logs y feedback", "Instrucciones"])

if tab == "Instrucciones":
    st.markdown("""
### Pasos rápidos
1. Subir CSV/Excel con columna de texto (p.ej. `descripcion`, `comentarios_usuarios`).
2. Ir a *Indexar datos* y elegir la columna. Click en *Indexar*.
3. En *Chat*, escribir preguntas. Las interacciones se guardan en logs.
4. Pedir a 5 testers que prueben y dejen feedback en el chat.
    """)
    st.markdown("**Archivos guardados localmente:** `faiss.index`, `docs_meta.json`, `chat_logs.jsonl`.")

elif tab == "Indexar datos":
    st.header("Construir/Actualizar índice")
    uploaded = st.file_uploader("Sube un CSV o Excel con textos (comentarios, FAQ, etc.)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.type == "text/csv":
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.write("Vista previa:", df.head())
            col = st.selectbox("Columna de texto para indexar", options=list(df.columns))
            if st.button("Construir índice desde este archivo"):
                with st.spinner("Indexando..."):
                    index, meta = build_index_from_dataframe(df, col)
        except Exception as e:
            st.error(f"Error leyendo archivo: {e}")
    else:
        st.info("Sube un archivo para indexar. Si ya tienes un índice guardado, no necesitas subir.")

elif tab == "Chat":
    st.header("Chat")
    system_prompt = st.text_area("Prompt de sistema (instrucciones al modelo)", value="Eres un asistente que ayuda a identificar oportunidades de mejora de productos/servicios. Sé claro, conciso y orientado al cliente.", height=120)
    # cargar index
    index, meta = load_index()
    if index is None:
        st.warning("No hay índice guardado. Ve a 'Indexar datos' y sube tus documentos.")
    user_input = st.chat_input("Escribe aquí la pregunta del cliente (p. ej. '¿Qué mejoras sugieres para el onboarding?')")

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

                # feedback widget
                col1, col2, col3 = st.columns([1,1,2])
                liked = None
                with col1:
                    if st.button("👍 Útil"):
                        liked = True
                with col2:
                    if st.button("👎 No útil"):
                        liked = False
                with col3:
                    feedback_txt = st.text_input("Comentario adicional (opcional)")

                record = {
                    "timestamp": time.time(),
                    "question": user_input,
                    "answer": answer,
                    "context_snippets": snippets,
                    "useful": liked,
                    "feedback": feedback_txt
                }
                log_interaction(record)
                st.success("Interacción registrada.")

elif tab == "Logs y feedback":
    st.header("Logs de conversaciones")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-200:]  # últimos 200 registros
        records = [json.loads(l) for l in lines]
        df_logs = pd.DataFrame(records)
        st.dataframe(df_logs)
        st.markdown("**Descargar logs**")
        st.download_button("Descargar logs (JSONL)", data="".join(lines), file_name="chat_logs.jsonl", mime="text/plain")
    else:
        st.info("No hay logs aún. Interactúa con el chatbot para generar datos.")
