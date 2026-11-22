# Sistema de Recuperación de Información (SRI) — Corpus FiQA-2018 (BeIR)

Aplicación principal en **Gradio** (`app.py`) y cuadernos experimentales en **Jupyter Notebook** para consultar el corpus financiero **FiQA-2018 (BeIR)** mediante modelos clásicos y búsqueda semántica con **FAISS**, mostrando rankings, comparaciones lado a lado y evaluación oficial con métricas estándar de Recuperación de Información.

- **Despliegue activo en producción (Render):** [https://proyecto-1er-bimestre-sri-fiqa2018-v2.onrender.com](https://proyecto-1er-bimestre-sri-fiqa2018-v2.onrender.com)
- **Repositorio:** `Riamu360/ProyectoRI` (rama `main`)
- **Informe técnico:** [Informe_Tecnico_Proyecto_SRI_FiQA2018_BN.pdf](Informe/Informe_Tecnico_Proyecto_SRI_FiQA2018_BN.pdf)

---

## Descripción General

El proyecto implementa un Sistema de Recuperación de Información (SRI) completo sobre el conjunto de datos **FiQA-2018** (*Financial Opinion Mining and Question Answering*, benchmark BeIR). La interfaz web (`app.py`) y el notebook principal (`Proyecto_1er_Bimestre_SRI_FiQA2018_v2.ipynb`) permiten:

- Ejecutar consultas en texto libre en inglés sobre el dominio financiero;
- Comparar simultáneamente tres modelos clásicos (Jaccard, TF-IDF + Coseno y BM25) frente a recuperación vectorial densa (Embeddings + FAISS);
- Visualizar rankings con puntaje de similitud, contenido completo del documento e indicador de relevancia oficial (`qrels`);
- Consultar métricas globales de evaluación (`Precision@10`, `Recall@10`, `MAP@100`) y latencia media sobre las 648 consultas del conjunto de prueba (`test`);
- Reutilizar artefactos persistentes en disco (índice FAISS, caché de consultas/relevancias y resultados de evaluación) para arrancar de forma rápida tanto en entorno local como en el servidor web en Render.

El backend combina:
- **Modelos clásicos implementados desde cero:** Jaccard (Binario), TF-IDF + Similitud de Coseno y BM25 (Probabilístico) apoyados en un índice invertido propio y matrices dispersas (`scipy.sparse.csr_matrix`);
- **Búsqueda semántica:** embeddings densos de 384 dimensiones con `all-MiniLM-L6-v2` (`sentence-transformers` localmente / `huggingface_hub.InferenceClient` en despliegue ligero) + índice vectorial **FAISS** (`IndexFlatIP`);
- **Persistencia local:** archivos `.parquet`, `.csv`, `.json` y `.index` precalculados en el directorio raíz del proyecto.

---

## Características Principales de `app.py`

- Búsqueda por consulta financiera en texto libre desde un panel unificado.
- Selector de modelo principal: **Binario (Jaccard)**, **Vectorial (TF-IDF + Coseno)**, **Probabilístico (BM25)** o **Vectorial Semántico (Embeddings + FAISS)**.
- Control ajustable de **Top-N** documentos recuperados (de 1 a 15).
- Ranking detallado del modelo seleccionado con verificación automática de relevancia (`SÍ [REL]`) cuando la consulta pertenece al `Test Set` de FiQA-2018.
- Tabla de comparación lado a lado entre los 4 modelos para la misma consulta.
- Pestaña de **Evaluación Oficial** con tabla comparativa global, gráficos de calidad y latencia (`matplotlib`) y desglose por consulta.
- Lectura mapeada en disco del índice vectorial (`fiqa_faiss.index` con `faiss.IO_FLAG_MMAP`) y descarga automática del corpus desde Hugging Face en caso de no encontrarse localmente.
- Optimización de memoria (`MAX_DOCS = 12000` y liberación explícita con `gc.collect()`) para operar establemente dentro de los 512 MB de RAM del plan gratuito de Render.

---

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

---

## Instalación

1. **Entrar al directorio del proyecto**
   ```bash
   cd "Proyecto RI"
   ```

2. **Crear y activar el entorno virtual**
   ```bash
   python -m venv venv

   # En Windows
   venv\Scripts\activate

   # En macOS/Linux
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Descargar recursos de NLTK**
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

---

## Estructura del Proyecto

```text
Proyecto RI/
├── app.py                                                  # Interfaz web principal en Gradio (desplegada en Render)
├── requirements.txt                                        # Dependencias de Python para ejecución y despliegue
├── README.md                                               # Documentación del proyecto
│
├── Proyecto_1er_Bimestre_SRI_FiQA2018.ipynb                # Versión inicial del cuaderno experimental
├── Proyecto_1er_Bimestre_SRI_FiQA2018_v2.ipynb             # Cuaderno principal completo y documentado
├── Proyecto_1er_Bimestre_SRI_FiQA2018_v2sin comentar.ipynb # Versión limpia del cuaderno v2
│
├── corpus-00000-of-00001.parquet                           # Corpus completo FiQA-2018 (57,638 documentos)
├── fiqa_queries.parquet                                    # Consultas oficiales del benchmark FiQA-2018
├── fiqa_qrels_test.parquet                                 # Juicios de relevancia oficiales (ground truth test)
│
├── fiqa_faiss.index                                        # Índice vectorial FAISS precalculado (384 dims)
├── fiqa_faiss_meta.json                                    # Metadatos del índice FAISS y modelo de embeddings
├── fiqa_eval_resumen.csv                                   # Resumen global de métricas (Precision, Recall, MAP, latencia)
└── fiqa_eval_detalle.parquet                               # Resultados detallados por consulta evaluada
```

---

## Uso Principal: `app.py`

### Acceso en Producción (Render)
La aplicación se encuentra desplegada y operativa en:
- **URL pública:** `https://proyecto-1er-bimestre-sri-fiqa2018-v2.onrender.com`

*(Nota: Al usar la instancia gratuita de Render, el servidor entra en reposo tras 15 minutos de inactividad y tarda aproximadamente 40–50 segundos en despertar ante la primera petición).*

### Ejecución Local
Para iniciar la interfaz web en tu equipo local:

```bash
python app.py
```

- **Acceso local:** `http://127.0.0.1:7860` (y enlace público temporal generado por Gradio con `share=True`).

### Qué hace la aplicación
- Carga el corpus desde `corpus-00000-of-00001.parquet` (o lo descarga desde `BeIR/fiqa` en Hugging Face si no está en disco, tomando una muestra de `MAX_DOCS = 12000` para el despliegue en Render).
- Filtra las consultas y juicios de relevancia válidos desde `fiqa_queries.parquet` y `fiqa_qrels_test.parquet`.
- Construye el índice invertido propio en memoria y la matriz dispersa TF-IDF (`csr_matrix`).
- Carga el índice vectorial precalculado desde `fiqa_faiss.index` mediante mapeo en memoria (`IO_FLAG_MMAP`).
- Ejecuta búsquedas interactivas y muestra las métricas oficiales precalculadas.

### Secciones de la interfaz
- **1. Búsqueda y Comparación de Modelos:** barra de consulta en texto libre, selector de modelo principal, control `Top-N`, ejemplos rápidos preconfigurados, tabla de resultados completos del modelo seleccionado y tabla comparativa lado a lado de los 4 modelos.
- **2. Evaluación Oficial (Precision, Recall, MAP):** tabla resumen global sobre 648 consultas evaluadas del corpus completo (57,638 documentos), gráficos comparativos de calidad y latencia, y muestra del desglose por consulta.

---

## Flujo de `app.py`

1. Verifica y descarga las `stopwords` en inglés de NLTK.
2. Carga los documentos desde `corpus-00000-of-00001.parquet` (o desde Hugging Face Hub si el archivo local no existe).
3. Carga `fiqa_queries.parquet` y `fiqa_qrels_test.parquet` para identificar relevancia en tiempo real.
4. Preprocesa los textos (minúsculas, tokenización alfanumérica con `RegexpTokenizer(r"[a-z0-9]+")` y filtrado de *stopwords*) y construye `IndiceInvertido`.
5. Inicializa `ModelosClasicosRI` (Jaccard, TF-IDF + Coseno y BM25) y `RecuperadorSemanticoFAISS`.
6. Procesa la consulta del usuario, calcula los rankings de los 4 modelos y renderiza las tablas y métricas en Gradio.

---

## Modelos Disponibles en la Interfaz

### 1. Binario (Jaccard)
$$\text{Jaccard}(Q, D) = \frac{|Q \cap D|}{|Q \cup D|}$$
- Ignora frecuencias de términos y mide el solapamiento entre conjuntos únicos de tokens de la consulta y el documento.
- Útil como línea base (*baseline*) léxica rápida.
- **Implementado en:** `ModelosClasicosRI.buscar_jaccard()`

### 2. Vectorial (TF-IDF + Similitud de Coseno)
$$\text{TF}(t, d) = 1 + \log_{10}(\text{freq}(t, d)), \qquad \text{IDF}(t) = \log_{10}\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$
$$\text{Score}(Q, D) = \cos(\vec{Q}, \vec{D}) = \frac{\vec{Q} \cdot \vec{D}}{\|\vec{Q}\| \|\vec{D}\|}$$
- Modelo algebraico clásico sobre matriz dispersa `csr_matrix` en `float32`.
- Pondera la frecuencia local logarítmica con la especificidad global del término en la colección.
- **Implementado en:** `ModelosClasicosRI.buscar_tfidf_coseno()`

### 3. Probabilístico (BM25 Okapi)
$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- Modelo probabilístico con saturación de frecuencia de término ($k_1 = 1.5$) y normalización por longitud de documento ($b = 0.75$).
- **Implementado en:** `ModelosClasicosRI.buscar_bm25()`

### 4. Búsqueda Semántica (Embeddings + FAISS)
- **Modelo de Embeddings:** `all-MiniLM-L6-v2` (384 dimensiones)
- **Índice Vectorial:** FAISS (`fiqa_faiss.index`, producto interno sobre vectores normalizados L2 equivalente a similitud coseno)
- **Implementado en:** `RecuperadorSemanticoFAISS.buscar()`
- **Ventajas:** captura intención y significado financiero más allá de la coincidencia exacta de palabras clave, manejando sinónimos y paráfrasis.

---

## Evaluación

La pestaña **2. Evaluación Oficial (Precision, Recall, MAP)** presenta los resultados obtenidos en el cuaderno `Proyecto_1er_Bimestre_SRI_FiQA2018_v2.ipynb` sobre las **648 consultas** de prueba y los **57,638 documentos** completos de FiQA-2018:

| Modelo | Consultas Evaluadas | Mean Precision@10 | Mean Recall@10 | MAP@100 | Latencia Media (ms/query) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Jaccard (Binario)** | 648 | 0.0153 | 0.0625 | 0.0384 | 14.23 |
| **TF-IDF (Coseno)** | 648 | 0.0491 | 0.2232 | 0.1273 | 10.84 |
| **BM25 (Probabilístico)** | 648 | 0.0647 | 0.2890 | 0.1791 | 12.51 |
| **Embeddings (FAISS)** | 648 | **0.1045** | **0.4413** | **0.3096** | 31.70 |

---

## Detalles Técnicos

### Índice Invertido
```python
indice = {
    "termino1": {"df": 42, "postings": {"doc_id1": 3, "doc_id2": 1, ...}},
    "termino2": {"df": 15, "postings": {"doc_id1": 2, ...}},
    ...
}
```
Construido dinámicamente en memoria junto con las longitudes por documento (`longitudes_doc`) y la longitud promedio del corpus (`longitud_promedio`).

### Índice Vectorial (FAISS)
- **Archivos:** `fiqa_faiss.index` y `fiqa_faiss_meta.json`
- **Métrica:** Similitud coseno (vectores normalizados L2 con búsqueda exacta `IndexFlatIP`)
- **Carga eficiente:** lectura directa mapeada desde disco (`faiss.IO_FLAG_MMAP`)

---

## Uso Programático

```python
from app import cargar_fiqa_corpus, IndiceInvertido, ModelosClasicosRI, RecuperadorSemanticoFAISS

# 1. Cargar corpus y construir índice invertido
corpus = cargar_fiqa_corpus("corpus-00000-of-00001.parquet", max_docs=None)
indice = IndiceInvertido()
indice.construir(corpus)

# 2. Búsqueda con modelos clásicos
motor_clasico = ModelosClasicosRI(indice)
resultados_bm25 = motor_clasico.buscar_bm25("Roth IRA vs traditional 401k", top_k=5)

# 3. Búsqueda semántica con FAISS
motor_semantico = RecuperadorSemanticoFAISS(list(corpus.keys()))
resultados_faiss = motor_semantico.buscar("How can I improve my credit score?", top_k=5)
```

---

## Dependencias Clave

| Librería | Propósito |
| :--- | :--- |
| `pandas` / `pyarrow` | Carga y lectura eficiente de archivos `.parquet` y `.csv` |
| `numpy` / `scipy` | Operaciones vectoriales y matrices dispersas (`csr_matrix`) |
| `nltk` | Tokenización (`RegexpTokenizer`) y filtrado de `stopwords` |
| `scikit-learn` | Utilidades de soporte y métricas en evaluación experimental |
| `faiss-cpu` | Indexación y búsqueda de similitud en vectores densos |
| `huggingface_hub` | Descarga del corpus y generación de embeddings de consulta |
| `matplotlib` | Generación de gráficos comparativos de métricas y latencia |
| `gradio` | Interfaz web interactiva |

---

## Uso Secundario: Jupyter Notebooks

Para reproducir paso a paso la indexación completa de los **57,638 documentos**, la generación de `fiqa_faiss.index` y el cálculo completo de `Precision@10`, `Recall@10` y `MAP@100` sobre las 648 consultas de prueba, abre y ejecuta:

```bash
jupyter notebook Proyecto_1er_Bimestre_SRI_FiQA2018_v2.ipynb
```

---

## Preguntas Frecuentes

- **P: ¿Cuál es el punto de entrada principal de la aplicación web?**  
  **R:** `app.py` es la interfaz web principal construida con Gradio, desplegada en `https://proyecto-1er-bimestre-sri-fiqa2018-v2.onrender.com`.

- **P: ¿Por qué la evaluación oficial muestra 57,638 documentos mientras que la búsqueda web en Render usa una muestra?**  
  **R:** La evaluación oficial (`fiqa_eval_resumen.csv` y `fiqa_eval_detalle.parquet`) se calculó previamente sobre el corpus completo en el notebook `Proyecto_1er_Bimestre_SRI_FiQA2018_v2.ipynb`. En `app.py`, `MAX_DOCS = 12000` mantiene el consumo por debajo de los 512 MB de RAM del entorno gratuito de Render sin afectar los resultados oficiales ni la velocidad de respuesta. Si ejecutas `app.py` localmente con más RAM, puedes cambiar `MAX_DOCS = None` para buscar sobre los 57,638 documentos.

- **P: ¿Cómo se actualizan los cambios en Render?**  
  **R:** Tras hacer `git push` al repositorio, ingresa al panel del servicio en Render y selecciona **Manual Deploy → Deploy latest commit**.
