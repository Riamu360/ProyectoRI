import os
import re
import gc
import math
import platform
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import faiss
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import HfApi, hf_hub_download
import gradio as gr

RUTA_PARQUET_LOCAL = "corpus-00000-of-00001.parquet"
RUTA_QUERIES_CACHE = "fiqa_queries.parquet"
RUTA_QRELS_CACHE   = "fiqa_qrels_test.parquet"
RUTA_FAISS_INDEX   = "fiqa_faiss.index"
RUTA_EVAL_RESUMEN  = "fiqa_eval_resumen.csv"
RUTA_EVAL_DETALLE  = "fiqa_eval_detalle.parquet"

# Muestra de 12,000 documentos para Render Free (512 MB RAM)
MAX_DOCS = 12000
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"
MAX_SEQ_LEN = 256
TOP_K_EVAL = 10
PROFUNDIDAD_AP = 100


def _leer_dataframe_acotado(ruta_local: str, max_docs: int = None) -> pd.DataFrame:
    """Lee solo las primeras max_docs filas cuando el archivo es Parquet para evitar picos de RAM."""
    lower = ruta_local.lower()
    if lower.endswith(".parquet"):
        if max_docs is None:
            return pd.read_parquet(ruta_local)
        pf = pq.ParquetFile(ruta_local)
        tablas = []
        acumulado = 0
        for batch in pf.iter_batches(batch_size=max_docs):
            tablas.append(pa.Table.from_batches([batch]))
            acumulado += batch.num_rows
            if acumulado >= max_docs:
                break
        if not tablas:
            return pd.DataFrame()
        return pa.concat_tables(tablas).to_pandas().head(max_docs)

    if lower.endswith((".jsonl", ".jsonl.gz")):
        df = pd.read_json(ruta_local, lines=True)
    else:
        df = pd.read_csv(ruta_local, sep="\t")
    return df.head(max_docs) if max_docs is not None else df


def _descargar_tabla_hf(repo_id: str, palabras_clave: list, max_docs: int = None) -> pd.DataFrame:
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
            return _leer_dataframe_acotado(ruta_local, max_docs=max_docs)

    raise FileNotFoundError(f"No se encontraron archivos ({palabras_clave}) en '{repo_id}'.")


def cargar_fiqa_corpus(ruta_local=RUTA_PARQUET_LOCAL, max_docs=MAX_DOCS):
    if os.path.exists(ruta_local):
        df_corpus = _leer_dataframe_acotado(ruta_local, max_docs=max_docs)
    else:
        print("Descargando muestra del corpus FiQA-2018 desde Hugging Face...")
        df_corpus = _descargar_tabla_hf("BeIR/fiqa", ["corpus"], max_docs=max_docs)

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


def cargar_fiqa_queries_y_qrels(corpus_ids_validos: set):
    if not (os.path.exists(RUTA_QUERIES_CACHE) and os.path.exists(RUTA_QRELS_CACHE)):
        return {}, {}

    df_queries = pd.read_parquet(RUTA_QUERIES_CACHE)
    df_qrels = pd.read_parquet(RUTA_QRELS_CACHE)

    todas_queries = {str(qid): str(txt) for qid, txt in zip(df_queries["_id"], df_queries["text"])}
    qrels_dict = defaultdict(set)
    for q_id, d_id, score in zip(df_qrels["query-id"], df_qrels["corpus-id"], df_qrels["score"]):
        q_str, d_str = str(q_id), str(d_id)
        if int(score) > 0 and d_str in corpus_ids_validos and q_str in todas_queries:
            qrels_dict[q_str].add(d_str)

    queries_eval = {q_id: todas_queries[q_id] for q_id in qrels_dict.keys()}
    del df_queries, df_qrels, todas_queries
    gc.collect()
    return queries_eval, dict(qrels_dict)


# Lista exacta de 198 stopwords en inglés de NLTK (evita importar nltk y ahorra ~93 MB de RAM)
STOPWORDS_EN = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
    "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll", "it's", "its",
    "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most",
    "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not",
    "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she",
    "she'd", "she'll", "she's", "should", "should've", "shouldn", "shouldn't",
    "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn",
    "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
})
TOKENIZADOR_RE = re.compile(r"[a-z0-9]+")


def preprocesar_texto(texto: str):
    if not texto:
        return []
    tokens = TOKENIZADOR_RE.findall(texto.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS_EN]


class IndiceInvertido:
    def __init__(self):
        self.indice = defaultdict(lambda: {"df": 0, "postings": {}})
        self.longitudes_doc = {}
        self.total_docs = 0
        self.longitud_promedio = 0.0
        self.vocabulario = []

    def construir(self, corpus_textos: dict):
        self.total_docs = len(corpus_textos)
        suma_longitudes = 0

        for doc_id, texto in corpus_textos.items():
            tokens = preprocesar_texto(texto)
            dl = len(tokens)
            self.longitudes_doc[doc_id] = dl
            suma_longitudes += dl
            for termino, tf in Counter(tokens).items():
                entrada = self.indice[termino]
                entrada["postings"][doc_id] = tf
                entrada["df"] += 1

        self.longitud_promedio = suma_longitudes / self.total_docs if self.total_docs > 0 else 0.0
        self.vocabulario = sorted(self.indice.keys())

    def obtener_postings(self, termino: str):
        entrada = self.indice.get(termino)
        return entrada["postings"] if entrada is not None else {}

    def obtener_df(self, termino: str):
        entrada = self.indice.get(termino)
        return entrada["df"] if entrada is not None else 0


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

        filas, columnas, valores = [], [], []
        for term, col_idx in self.vocab_idx.items():
            df = self.indice.obtener_df(term)
            idf = math.log10((N + 1) / (df + 1)) + 1.0
            self.idf_vector[col_idx] = idf
            for doc_id, tf in self.indice.obtener_postings(term).items():
                row_idx = self.doc_id_to_idx[doc_id]
                self.cardinalidad_doc[row_idx] += 1
                filas.append(row_idx)
                columnas.append(col_idx)
                valores.append((1.0 + math.log10(tf)) * idf)

        self.matriz_tfidf = csr_matrix((np.array(valores, dtype=np.float32), (filas, columnas)), shape=(N, V))
        del filas, columnas, valores
        gc.collect()
        self.normas_docs = np.sqrt(self.matriz_tfidf.multiply(self.matriz_tfidf).sum(axis=1)).A1

    def buscar_jaccard(self, consulta: str, top_k: int = 10):
        tokens_q = set(preprocesar_texto(consulta))
        if not tokens_q:
            return []
        intersecciones = defaultdict(int)
        for term in tokens_q:
            for doc_id in self.indice.obtener_postings(term):
                intersecciones[self.doc_id_to_idx[doc_id]] += 1
        len_q = len(tokens_q)
        resultados = [
            (self.doc_ids[r], inter / (len_q + int(self.cardinalidad_doc[r]) - inter))
            for r, inter in intersecciones.items()
            if (len_q + int(self.cardinalidad_doc[r]) - inter) > 0
        ]
        resultados.sort(key=lambda x: x[1], reverse=True)
        return [(d, round(float(s), 4)) for d, s in resultados[:top_k]]

    def buscar_tfidf_coseno(self, consulta: str, top_k: int = 10):
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
        vec_q = csr_matrix((np.array(vals_q, dtype=np.float32), (np.zeros(len(cols_q), dtype=int), cols_q)), shape=(1, len(self.indice.vocabulario)))
        norma_q = float(np.sqrt(vec_q.multiply(vec_q).sum()))
        if norma_q == 0.0:
            return []
        productos = self.matriz_tfidf.dot(vec_q.T).toarray().ravel()
        denominadores = self.normas_docs * norma_q
        similitudes = np.divide(productos, denominadores, out=np.zeros_like(productos), where=denominadores != 0)
        k = min(top_k, len(similitudes))
        top_indices = np.argpartition(-similitudes, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similitudes[top_indices])]
        return [(self.doc_ids[i], round(float(similitudes[i]), 4)) for i in top_indices if similitudes[i] > 0]

    def buscar_bm25(self, consulta: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        tokens_q = preprocesar_texto(consulta)
        if not tokens_q:
            return []
        N = self.indice.total_docs
        avgdl = self.indice.longitud_promedio
        conteo_q = Counter(tokens_q)
        scores = defaultdict(float)
        for term, tf_q in conteo_q.items():
            df = self.indice.obtener_df(term)
            if df == 0:
                continue
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
            for doc_id, tf in self.indice.obtener_postings(term).items():
                dl = self.indice.longitudes_doc[doc_id]
                scores[doc_id] += tf_q * idf * ((tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl))))
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(d, round(float(s), 4)) for d, s in ranking[:top_k]]


class RecuperadorSemanticoFAISS:
    """Codifica consultas localmente con ONNX Runtime (sin PyTorch ni HF_TOKEN) y busca en FAISS."""

    def __init__(self, doc_ids_muestra: list):
        self.doc_ids = doc_ids_muestra
        self.index_faiss = faiss.read_index(RUTA_FAISS_INDEX, faiss.IO_FLAG_MMAP)
        self.ultimo_error = ""

        repo_modelo = f"sentence-transformers/{MODELO_EMBEDDINGS}"
        ruta_tok = hf_hub_download(repo_id=repo_modelo, filename="tokenizer.json")
        self.tokenizer = Tokenizer.from_file(ruta_tok)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

        self.session = self._cargar_sesion_onnx(repo_modelo)
        self._nombres_entrada = {inp.name for inp in self.session.get_inputs()}

    @staticmethod
    def _cargar_sesion_onnx(repo_modelo: str) -> ort.InferenceSession:
        arch = platform.machine().lower()
        if "arm" in arch or "aarch64" in arch:
            candidatos = ["onnx/model_qint8_arm64.onnx", "onnx/model.onnx"]
        else:
            candidatos = ["onnx/model_quint8_avx2.onnx", "onnx/model.onnx"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        ultimo_exc = None
        for nombre_onnx in candidatos:
            try:
                ruta_onnx = hf_hub_download(repo_id=repo_modelo, filename=nombre_onnx)
                return ort.InferenceSession(ruta_onnx, sess_options=opts, providers=["CPUExecutionProvider"])
            except Exception as exc:
                ultimo_exc = exc
        raise RuntimeError(f"No se pudo inicializar la sesión ONNX para '{repo_modelo}': {ultimo_exc}")

    def _codificar_consulta(self, consulta: str) -> np.ndarray:
        enc = self.tokenizer.encode(consulta)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._nombres_entrada:
            feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

        token_embeddings = self.session.run(None, feed)[0]
        mask_f32 = attention_mask[:, :, None].astype(np.float32)
        pooled = (token_embeddings * mask_f32).sum(axis=1) / np.clip(mask_f32.sum(axis=1), 1e-9, None)
        emb_q = pooled.astype("float32")
        faiss.normalize_L2(emb_q)
        return emb_q

    def buscar(self, consulta: str, top_k: int = 10):
        self.ultimo_error = ""
        if top_k <= 0 or not consulta or not consulta.strip():
            return []
        try:
            emb_q = self._codificar_consulta(consulta)
            n_muestra = len(self.doc_ids)
            if n_muestra == 0:
                return []

            k = min(top_k, n_muestra)
            if n_muestra < self.index_faiss.ntotal:
                params = faiss.SearchParameters(sel=faiss.IDSelectorRange(0, n_muestra))
                distancias, indices = self.index_faiss.search(emb_q, k, params=params)
            else:
                distancias, indices = self.index_faiss.search(emb_q, k)

            resultados = []
            for score, idx in zip(distancias[0], indices[0]):
                if 0 <= idx < n_muestra and score > 0:
                    resultados.append((self.doc_ids[idx], round(float(score), 4)))
            return resultados
        except Exception as e:
            self.ultimo_error = str(e)
            print(f"Error búsqueda semántica ONNX/FAISS: {e}")
            return []


# Carga directa y ligera de la muestra (12,000 docs)
print("Cargando muestra de FiQA-2018...")
corpus = cargar_fiqa_corpus()
doc_ids = list(corpus.keys())
queries_test, qrels_test = cargar_fiqa_queries_y_qrels(set(doc_ids))

indice_invertido = IndiceInvertido()
indice_invertido.construir(corpus)
gc.collect()

motor_clasico = ModelosClasicosRI(indice_invertido)
motor_semantico = RecuperadorSemanticoFAISS(doc_ids)
gc.collect()
print(f"SRI listo con muestra de {len(corpus):,} documentos.")


def graficar_metricas_modelos(df_resumen: pd.DataFrame, k: int = TOP_K_EVAL, profundidad_ap: int = PROFUNDIDAD_AP):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    modelos = df_resumen["Modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.25
    axes[0].bar(x - ancho, df_resumen[f"Mean Precision@{k}"], width=ancho, label=f"Precision@{k}", color="#4C78A8")
    axes[0].bar(x, df_resumen[f"Mean Recall@{k}"], width=ancho, label=f"Recall@{k}", color="#F58518")
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

    res_jaccard = motor_clasico.buscar_jaccard(consulta, top_k=top_n)
    res_tfidf = motor_clasico.buscar_tfidf_coseno(consulta, top_k=top_n)
    res_bm25 = motor_clasico.buscar_bm25(consulta, top_k=top_n)
    res_faiss = motor_semantico.buscar(consulta, top_k=top_n)

    if modelo_elegido == "Binario (Jaccard)":
        res_sel = res_jaccard
    elif modelo_elegido == "Vectorial (TF-IDF + Coseno)":
        res_sel = res_tfidf
    elif modelo_elegido == "Probabilístico (BM25)":
        res_sel = res_bm25
    else:
        res_sel = res_faiss

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
        "Jaccard (Binario)": fmt_celdas(res_jaccard),
        "TF-IDF (Coseno)": fmt_celdas(res_tfidf),
        "BM25 (Probabilístico)": fmt_celdas(res_bm25),
        "Embeddings (FAISS)": fmt_celdas(res_faiss),
    })

    estado = (
        f"Consulta detectada en Test Set (`query_id={q_id_match}`, {len(relevantes)} relevantes en qrels)."
        if q_id_match else f"Consulta libre ejecutada sobre muestra de {len(corpus):,} documentos."
    )
    if motor_semantico.ultimo_error:
        estado += f" | Aviso Embeddings: {motor_semantico.ultimo_error}"
    return df_individual, df_comp, estado


df_resumen_ui = pd.read_csv(RUTA_EVAL_RESUMEN) if os.path.exists(RUTA_EVAL_RESUMEN) else pd.DataFrame()
df_detalle_ui = pd.read_parquet(RUTA_EVAL_DETALLE) if os.path.exists(RUTA_EVAL_DETALLE) else pd.DataFrame()
n_eval_ui = int(df_resumen_ui["Consultas Evaluadas"].iloc[0]) if not df_resumen_ui.empty else 0

ejemplos_ui = [
    ["Should I invest in a Roth IRA or traditional 401k for retirement?"],
    ["How can I improve my credit score quickly without paying fees?"],
    ["What is the difference between an index fund and an ETF?"],
]
if queries_test:
    primeras_test = [[q] for q in list(queries_test.values())[:3]]
    ejemplos_ui = primeras_test + ejemplos_ui[:1]

with gr.Blocks(title="SRI FiQA-2018 — Proyecto 1er Bimestre") as demo:
    gr.Markdown("# Sistema de Recuperación de Información — Corpus FiQA-2018 (BeIR)")
    gr.Markdown(f"Comparación de modelos clásicos y recuperación semántica sobre muestra de **{len(corpus):,}** documentos (evaluación oficial sobre 57,638 documentos).")

    with gr.Tab("1. Búsqueda y Comparación de Modelos"):
        with gr.Row():
            txt_consulta = gr.Textbox(label="Consulta financiera en texto libre", placeholder="Ej: Should I invest in a Roth IRA or traditional 401k for retirement?", scale=3)
            dd_modelo = gr.Dropdown(
                choices=["Binario (Jaccard)", "Vectorial (TF-IDF + Coseno)", "Probabilístico (BM25)", "Vectorial Semántico (Embeddings + FAISS)"],
                value="Vectorial Semántico (Embeddings + FAISS)", label="Modelo Principal", scale=2
            )
            sld_topn = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top-N", scale=1)
        btn_buscar = gr.Button("Ejecutar Búsqueda", variant="primary")
        lbl_estado = gr.Markdown("")
        gr.Examples(examples=ejemplos_ui, inputs=[txt_consulta], label="Ejemplos rápidos (clic para probar)")
        gr.Markdown("### Ranking del Modelo Seleccionado")
        out_individual = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown("### Comparación Lado a Lado entre los 4 Modelos")
        out_comparacion = gr.Dataframe(interactive=False, wrap=True)
        btn_buscar.click(fn=ejecutar_busqueda_web, inputs=[txt_consulta, dd_modelo, sld_topn], outputs=[out_individual, out_comparacion, lbl_estado])

    with gr.Tab("2. Evaluación Oficial (Precision, Recall, MAP)"):
        if not df_resumen_ui.empty:
            gr.Markdown(f"### Resumen Global de Métricas en FiQA-2018 ({n_eval_ui:,} consultas evaluadas sobre el corpus completo)")
            gr.Dataframe(value=df_resumen_ui, interactive=False)
            gr.Plot(value=graficar_metricas_modelos(df_resumen_ui), label="Gráfico Comparativo")
            gr.Markdown("### Desglose por Consulta (Muestra de 100 filas)")
            gr.Dataframe(value=df_detalle_ui.head(100), interactive=False)

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 7860))
    compartir = os.environ.get("GRADIO_SHARE", "").lower() == "true"
    demo.launch(server_name="0.0.0.0", server_port=puerto, share=compartir)
