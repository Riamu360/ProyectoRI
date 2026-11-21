import os
import re
import gc
import json
import math
import time
import hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import faiss
from huggingface_hub import InferenceClient
import gradio as gr

for recurso_ruta, recurso_id in [("corpora/stopwords", "stopwords")]:
    try:
        nltk.data.find(recurso_ruta)
    except LookupError:
        nltk.download(recurso_id, quiet=True)

RUTA_PARQUET_LOCAL = "corpus-00000-of-00001.parquet"
RUTA_QUERIES_CACHE = "fiqa_queries.parquet"
RUTA_QRELS_CACHE   = "fiqa_qrels_test.parquet"
RUTA_FAISS_INDEX   = "fiqa_faiss.index"
RUTA_FAISS_META    = "fiqa_faiss_meta.json"
RUTA_EVAL_RESUMEN  = "fiqa_eval_resumen.csv"
RUTA_EVAL_DETALLE  = "fiqa_eval_detalle.parquet"

MAX_DOCS = None
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"
TOP_K_EVAL = 10
PROFUNDIDAD_AP = 100


# ============================================================================
# 1. CARGA DEL CORPUS, QUERIES Y QRELS
# ============================================================================
def _descargar_tabla_hf(repo_id: str, palabras_clave: list):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for revision in ["refs/convert/parquet", "main"]:
        try:
            archivos = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        except Exception:
            continue

        candidatos = [
            f for f in archivos
            if any(kw in f.lower() for kw in palabras_clave)
            and f.lower().endswith((".parquet", ".jsonl", ".jsonl.gz", ".tsv"))
        ]
        candidatos.sort(key=lambda x: (0 if x.endswith(".parquet") else 1, x))
        if candidatos:
            ruta_local = hf_hub_download(
                repo_id=repo_id,
                filename=candidatos[0],
                repo_type="dataset",
                revision=revision,
            )
            if candidatos[0].endswith(".parquet"):
                return pd.read_parquet(ruta_local)
            if candidatos[0].endswith((".jsonl", ".jsonl.gz")):
                return pd.read_json(ruta_local, lines=True)
            return pd.read_csv(ruta_local, sep="\t")

    raise FileNotFoundError(f"No se encontraron archivos ({palabras_clave}) en '{repo_id}'.")


def cargar_fiqa_corpus(ruta_local=RUTA_PARQUET_LOCAL, max_docs=MAX_DOCS):
    if os.path.exists(ruta_local):
        df_corpus = pd.read_parquet(ruta_local)
    else:
        print("Descargando corpus FiQA-2018 desde Hugging Face...")
        df_corpus = _descargar_tabla_hf("BeIR/fiqa", ["corpus"])
        df_corpus.to_parquet(ruta_local, index=False)

    if max_docs is not None:
        df_corpus = df_corpus.head(max_docs)

    corpus_dict = {}
    ids_col = df_corpus["_id"].tolist()
    titulos_col = df_corpus["title"].tolist() if "title" in df_corpus.columns else [""] * len(df_corpus)
    textos_col = df_corpus["text"].tolist()

    for doc_id, titulo, cuerpo in zip(ids_col, titulos_col, textos_col):
        doc_id_str = str(doc_id)
        t_str = str(titulo).strip() if pd.notna(titulo) else ""
        c_str = str(cuerpo).strip() if pd.notna(cuerpo) else ""
        corpus_dict[doc_id_str] = f"{t_str} {c_str}".strip() if t_str else c_str

    del df_corpus, ids_col, titulos_col, textos_col
    gc.collect()
    return corpus_dict


def cargar_fiqa_queries_y_qrels(
    corpus_ids_validos: set,
    ruta_queries_cache: str = RUTA_QUERIES_CACHE,
    ruta_qrels_cache: str = RUTA_QRELS_CACHE,
):
    if not (os.path.exists(ruta_queries_cache) and os.path.exists(ruta_qrels_cache)):
        return {}, {}

    df_queries = pd.read_parquet(ruta_queries_cache)
    df_qrels = pd.read_parquet(ruta_qrels_cache)

    todas_queries = {
        str(qid): str(txt)
        for qid, txt in zip(df_queries["_id"], df_queries["text"])
    }

    qrels_dict = defaultdict(set)
    for q_id, d_id, score in zip(df_qrels["query-id"], df_qrels["corpus-id"], df_qrels["score"]):
        q_str, d_str = str(q_id), str(d_id)
        if int(score) > 0 and d_str in corpus_ids_validos and q_str in todas_queries:
            qrels_dict[q_str].add(d_str)

    queries_eval = {q_id: todas_queries[q_id] for q_id in qrels_dict.keys()}
    del df_queries, df_qrels, todas_queries
    gc.collect()
    return queries_eval, dict(qrels_dict)


STOPWORDS_EN = frozenset(stopwords.words("english"))
TOKENIZADOR_NLTK = RegexpTokenizer(r"[a-z0-9]+")


def preprocesar_texto(texto: str):
    if not texto:
        return []
    tokens = TOKENIZADOR_NLTK.tokenize(texto.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS_EN]


# ============================================================================
# 2. ÍNDICE INVERTIDO PROPIO (SIN DICCIONARIO INTERMEDIO DE TOKENS)
# ============================================================================
class IndiceInvertido:
    def __init__(self):
        self.indice = defaultdict(lambda: {"df": 0, "postings": {}})
        self.longitudes_doc = {}
        self.total_docs = 0
        self.longitud_promedio = 0.0
        self.vocabulario = []
        self.total_postings = 0

    def construir(self, corpus_textos: dict):
        self.total_docs = len(corpus_textos)
        suma_longitudes = 0

        for doc_id, texto in corpus_textos.items():
            tokens = preprocesar_texto(texto)
            dl = len(tokens)
            self.longitudes_doc[doc_id] = dl
            suma_longitudes += dl
            conteo = Counter(tokens)

            for termino, tf in conteo.items():
                entrada = self.indice[termino]
                entrada["postings"][doc_id] = tf
                entrada["df"] += 1
                self.total_postings += 1

        self.longitud_promedio = suma_longitudes / self.total_docs if self.total_docs > 0 else 0.0
        self.vocabulario = sorted(self.indice.keys())

    def obtener_postings(self, termino: str):
        entrada = self.indice.get(termino)
        return entrada["postings"] if entrada is not None else {}

    def obtener_df(self, termino: str):
        entrada = self.indice.get(termino)
        return entrada["df"] if entrada is not None else 0


# ============================================================================
# 3. MODELOS CLÁSICOS (JACCARD, TF-IDF, BM25)
# ============================================================================
class ModelosClasicosRI:
    def __init__(self, indice: IndiceInvertido):
        self.indice = indice
        self.doc_ids = list(indice.longitudes_doc.keys())
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.vocab_idx = {term: i for i, term in enumerate(self.indice.vocabulario)}
        self.cardinalidad_doc = np.zeros(len(self.doc_ids), dtype=np.int32)
        self._construir_matriz_dispersa_tfidf()

    def _construir_matriz_dispersa_tfidf(self):
        N = self.indice.total_docs
        V = len(self.indice.vocabulario)
        self.idf_vector = np.zeros(V, dtype=np.float32)

        if N == 0 or V == 0:
            self.matriz_tfidf = csr_matrix((N, V), dtype=np.float32)
            self.normas_docs = np.zeros(N, dtype=np.float32)
            return

        filas, columnas, valores = [], [], []
        for term, col_idx in self.vocab_idx.items():
            df = self.indice.obtener_df(term)
            idf = math.log10((N + 1) / (df + 1)) + 1.0
            self.idf_vector[col_idx] = idf

            for doc_id, tf in self.indice.obtener_postings(term).items():
                row_idx = self.doc_id_to_idx[doc_id]
                self.cardinalidad_doc[row_idx] += 1
                tf_log = 1.0 + math.log10(tf) if tf > 0 else 0.0
                filas.append(row_idx)
                columnas.append(col_idx)
                valores.append(tf_log * idf)

        self.matriz_tfidf = csr_matrix(
            (np.array(valores, dtype=np.float32), (filas, columnas)),
            shape=(N, V),
        )
        del filas, columnas, valores
        gc.collect()
        self.normas_docs = np.sqrt(self.matriz_tfidf.multiply(self.matriz_tfidf).sum(axis=1)).A1

    def buscar_jaccard(self, consulta: str, top_k: int = 10):
        if top_k <= 0:
            return []
        tokens_q = set(preprocesar_texto(consulta))
        if not tokens_q:
            return []

        intersecciones = defaultdict(int)
        for term in tokens_q:
            for doc_id in self.indice.obtener_postings(term):
                intersecciones[self.doc_id_to_idx[doc_id]] += 1

        if not intersecciones:
            return []

        len_q = len(tokens_q)
        resultados = []
        for row_idx, inter in intersecciones.items():
            union = len_q + int(self.cardinalidad_doc[row_idx]) - inter
            score = inter / union if union > 0 else 0.0
            resultados.append((self.doc_ids[row_idx], float(score)))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return [(doc_id, round(score, 4)) for doc_id, score in resultados[:top_k]]

    def buscar_tfidf_coseno(self, consulta: str, top_k: int = 10):
        if top_k <= 0 or self.indice.total_docs == 0:
            return []
        tokens_q = preprocesar_texto(consulta)
        if not tokens_q:
            return []

        conteo_q = Counter(tokens_q)
        cols_q, vals_q = [], []
        for term, tf in conteo_q.items():
            idx = self.vocab_idx.get(term)
            if idx is not None:
                cols_q.append(idx)
                vals_q.append((1.0 + math.log10(tf)) * self.idf_vector[idx])

        if not cols_q:
            return []

        vec_q = csr_matrix(
            (np.array(vals_q, dtype=np.float32), (np.zeros(len(cols_q), dtype=int), cols_q)),
            shape=(1, len(self.indice.vocabulario)),
        )
        norma_q = float(np.sqrt(vec_q.multiply(vec_q).sum()))
        if norma_q == 0.0:
            return []

        productos = self.matriz_tfidf.dot(vec_q.T).toarray().ravel()
        denominadores = self.normas_docs * norma_q
        similitudes = np.divide(
            productos,
            denominadores,
            out=np.zeros_like(productos),
            where=denominadores != 0,
        )

        k = min(top_k, len(similitudes))
        top_indices = np.argpartition(-similitudes, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similitudes[top_indices])]

        return [
            (self.doc_ids[i], round(float(similitudes[i]), 4))
            for i in top_indices
            if similitudes[i] > 0
        ]

    def buscar_bm25(self, consulta: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        if top_k <= 0:
            return []
        tokens_q = preprocesar_texto(consulta)
        if not tokens_q:
            return []

        N = self.indice.total_docs
        avgdl = self.indice.longitud_promedio
        if N == 0 or avgdl == 0.0:
            return []

        conteo_q = Counter(tokens_q)
        scores = defaultdict(float)

        for term, tf_q in conteo_q.items():
            df = self.indice.obtener_df(term)
            if df == 0:
                continue
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
            for doc_id, tf in self.indice.obtener_postings(term).items():
                dl = self.indice.longitudes_doc[doc_id]
                numerador = tf * (k1 + 1.0)
                denominador = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[doc_id] += tf_q * idf * (numerador / denominador)

        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, round(float(score), 4)) for doc_id, score in ranking[:top_k]]


# ============================================================================
# 4. RECUPERADOR SEMÁNTICO LIGERO (FAISS EN DISCO + API HUGGING FACE)
# ============================================================================
class RecuperadorSemanticoFAISS:
    def __init__(self, modelo_nombre: str = MODELO_EMBEDDINGS):
        self.modelo_nombre = modelo_nombre
        self.client = InferenceClient(
            model=f"sentence-transformers/{modelo_nombre}",
            token=os.environ.get("HF_TOKEN")
        )
        self.dimension = 384
        self.index_faiss = None
        self.doc_ids = []

    def indexar_o_cargar(self, corpus_dict: dict, ruta_indice: str = RUTA_FAISS_INDEX):
        self.doc_ids = list(corpus_dict.keys())
        self.index_faiss = faiss.read_index(ruta_indice, faiss.IO_FLAG_MMAP)
        print(f"Índice FAISS cargado (mmap): {self.index_faiss.ntotal:,} vectores.")

    def buscar(self, consulta: str, top_k: int = 10):
        if top_k <= 0 or not consulta or not consulta.strip() or self.index_faiss is None:
            return []
        try:
            emb_q = np.array(self.client.feature_extraction(consulta), dtype="float32")
            if emb_q.ndim == 1:
                emb_q = emb_q.reshape(1, -1)
            elif emb_q.ndim == 3:
                emb_q = emb_q.mean(axis=1)
            faiss.normalize_L2(emb_q)

            k = min(top_k, len(self.doc_ids))
            distancias, indices = self.index_faiss.search(emb_q, k)
            return [
                (self.doc_ids[idx], round(float(score), 4))
                for score, idx in zip(distancias[0], indices[0])
                if idx != -1 and score > 0
            ]
        except Exception as e:
            print(f"Error consultando API de embeddings: {e}")
            return []


# ============================================================================
# 5. INICIALIZACIÓN DE MOTORES CON LIBERACIÓN DE MEMORIA
# ============================================================================
print("Iniciando carga del SRI FiQA-2018...")
corpus = cargar_fiqa_corpus()
queries_test, qrels_test = cargar_fiqa_queries_y_qrels(set(corpus.keys()))

indice_invertido = IndiceInvertido()
indice_invertido.construir(corpus)
gc.collect()

motor_clasico = ModelosClasicosRI(indice_invertido)
gc.collect()

motor_semantico = RecuperadorSemanticoFAISS(MODELO_EMBEDDINGS)
motor_semantico.indexar_o_cargar(corpus)
gc.collect()
print("Motores listos.")


# ============================================================================
# 6. GRÁFICOS E INTERFAZ WEB GRADIO
# ============================================================================
def graficar_metricas_modelos(df_resumen: pd.DataFrame, k: int = TOP_K_EVAL, profundidad_ap: int = PROFUNDIDAD_AP):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    modelos = df_resumen["Modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.25

    axes[0].bar(x - ancho, df_resumen[f"Mean Precision@{k}"], width=ancho, label=f"Precision@{k}", color="#4C78A8")
    axes[0].bar(x,         df_resumen[f"Mean Recall@{k}"], width=ancho, label=f"Recall@{k}", color="#F58518")
    axes[0].bar(x + ancho, df_resumen[f"MAP@{profundidad_ap}"], width=ancho, label=f"MAP@{profundidad_ap}", color="#54A24B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modelos, rotation=15, ha="right")
    axes[0].set_ylabel("Puntaje (0 - 1)")
    axes[0].set_title("Comparación de Calidad de Recuperación (FiQA-2018)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].bar(modelos, df_resumen["Latencia Media (ms/query)"], color="#E45756")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modelos, rotation=15, ha="right")
    axes[1].set_ylabel("Milisegundos por consulta")
    axes[1].set_title("Latencia Media de Búsqueda por Modelo")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def _cargar_metricas_seguro():
    res = pd.read_csv(RUTA_EVAL_RESUMEN) if os.path.exists(RUTA_EVAL_RESUMEN) else pd.DataFrame()
    det = pd.read_parquet(RUTA_EVAL_DETALLE) if os.path.exists(RUTA_EVAL_DETALLE) else pd.DataFrame()
    return res, det


def _buscar_qrels_por_texto(consulta: str):
    consulta_norm = consulta.strip().lower()
    for q_id, q_txt in queries_test.items():
        if q_txt.strip().lower() == consulta_norm:
            return q_id, qrels_test.get(q_id, set())
    return None, set()


def ejecutar_busqueda_web(consulta: str, modelo_elegido: str, top_n: int):
    cols_ind = ["Rank", "Doc ID (FiQA)", "Relevante (qrels)", "Score", "Contenido Completo del Documento"]
    if not consulta or not consulta.strip():
        return pd.DataFrame(columns=cols_ind), pd.DataFrame(), "Ingresa una consulta para buscar."

    top_n = int(top_n)
    q_id_match, relevantes = _buscar_qrels_por_texto(consulta)

    if modelo_elegido == "Binario (Jaccard)":
        res_sel = motor_clasico.buscar_jaccard(consulta, top_k=top_n)
    elif modelo_elegido == "Vectorial (TF-IDF + Coseno)":
        res_sel = motor_clasico.buscar_tfidf_coseno(consulta, top_k=top_n)
    elif modelo_elegido == "Probabilístico (BM25)":
        res_sel = motor_clasico.buscar_bm25(consulta, top_k=top_n)
    else:
        res_sel = motor_semantico.buscar(consulta, top_k=top_n)

    filas_ind = [
        {
            "Rank": i + 1,
            "Doc ID (FiQA)": doc_id,
            "Relevante (qrels)": "SÍ [REL]" if doc_id in relevantes else ("No" if relevantes else "-"),
            "Score": score,
            "Contenido Completo del Documento": corpus.get(doc_id, ""),
        }
        for i, (doc_id, score) in enumerate(res_sel)
    ]
    df_individual = pd.DataFrame(filas_ind, columns=cols_ind)

    def fmt_celdas(lista):
        salida = []
        for d, s in lista:
            marca = " [REL]" if d in relevantes else ""
            snippet = corpus.get(d, "")[:95].replace("\n", " ")
            salida.append(f"ID:{d}{marca} ({s}) — {snippet}...")
        while len(salida) < top_n:
            salida.append("-")
        return salida

    df_comp = pd.DataFrame({
        "Rank": [i + 1 for i in range(top_n)],
        "Jaccard (Binario)": fmt_celdas(motor_clasico.buscar_jaccard(consulta, top_k=top_n)),
        "TF-IDF (Coseno)": fmt_celdas(motor_clasico.buscar_tfidf_coseno(consulta, top_k=top_n)),
        "BM25 (Probabilístico)": fmt_celdas(motor_clasico.buscar_bm25(consulta, top_k=top_n)),
        "Embeddings (FAISS)": fmt_celdas(motor_semantico.buscar(consulta, top_k=top_n)),
    })

    if q_id_match:
        estado = f"Consulta detectada en Test Set (`query_id={q_id_match}`, {len(relevantes)} relevantes en qrels)."
    else:
        estado = f"Consulta libre ejecutada sobre {len(corpus):,} documentos."

    return df_individual, df_comp, estado


df_resumen_ui, df_detalle_ui = _cargar_metricas_seguro()
n_eval_ui = int(df_resumen_ui["Consultas Evaluadas"].iloc[0]) if not df_resumen_ui.empty else 0

ejemplos_ui = [
    ["Should I invest in a Roth IRA or traditional 401k for retirement?"],
    ["How can I improve my credit score quickly without paying fees?"],
    ["What is the difference between an index fund and an ETF?"],
]
if queries_test:
    primeras_test = list(queries_test.values())[:3]
    ejemplos_ui = [[q] for q in primeras_test] + ejemplos_ui[:1]

with gr.Blocks(title="SRI FiQA-2018 — Proyecto 1er Bimestre") as demo:
    gr.Markdown("# Sistema de Recuperación de Información — Corpus FiQA-2018 (BeIR)")
    gr.Markdown(
        f"Comparación de modelos clásicos (Jaccard, TF-IDF, BM25) y recuperación semántica "
        f"(`{MODELO_EMBEDDINGS}` + FAISS) sobre **{len(corpus):,}** documentos financieros."
    )

    with gr.Tab("1. Búsqueda y Comparación de Modelos"):
        with gr.Row():
            txt_consulta = gr.Textbox(
                label="Consulta financiera en texto libre",
                placeholder="Ej: Should I invest in a Roth IRA or traditional 401k for retirement?",
                scale=3,
            )
            dd_modelo = gr.Dropdown(
                choices=[
                    "Binario (Jaccard)",
                    "Vectorial (TF-IDF + Coseno)",
                    "Probabilístico (BM25)",
                    "Vectorial Semántico (Embeddings + FAISS)",
                ],
                value="Vectorial Semántico (Embeddings + FAISS)",
                label="Modelo Principal",
                scale=2,
            )
            sld_topn = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-N", scale=1)

        btn_buscar = gr.Button("Ejecutar Búsqueda", variant="primary")
        lbl_estado = gr.Markdown("")

        gr.Examples(examples=ejemplos_ui, inputs=[txt_consulta], label="Ejemplos rápidos (clic para probar)")

        gr.Markdown("### Ranking del Modelo Seleccionado")
        out_individual = gr.Dataframe(interactive=False, wrap=True)

        gr.Markdown("### Comparación Lado a Lado entre los 4 Modelos")
        out_comparacion = gr.Dataframe(interactive=False, wrap=True)

        btn_buscar.click(
            fn=ejecutar_busqueda_web,
            inputs=[txt_consulta, dd_modelo, sld_topn],
            outputs=[out_individual, out_comparacion, lbl_estado],
        )

    with gr.Tab("2. Evaluación Oficial (Precision, Recall, MAP)"):
        if not df_resumen_ui.empty:
            gr.Markdown(f"### Resumen Global de Métricas en FiQA-2018 ({n_eval_ui:,} consultas evaluadas)")
            gr.Dataframe(value=df_resumen_ui, interactive=False)
            gr.Plot(value=graficar_metricas_modelos(df_resumen_ui), label="Gráfico Comparativo")
            gr.Markdown("### Desglose por Consulta (Muestra de 100 filas)")
            gr.Dataframe(value=df_detalle_ui.head(100), interactive=False)
        else:
            gr.Markdown("**No se encontraron archivos de métricas en el repositorio.**")

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=puerto)