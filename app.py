# app.py — Predicción de deserción y éxito académico (Streamlit + modelo Spark ML sin Spark)
#
# Archivos necesarios en el repositorio (misma carpeta):
#   app.py | modelo_final/ (guardado desde Colab) | banner.jpg (opcional)
#   requirements.txt: streamlit, pyarrow, pandas
#   (ya NO se necesita pyspark ni packages.txt / Java)
#
# Ejecutar en local:  streamlit run app.py

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as parquet
import streamlit as st

AUTOR = "Alfredo Diaz Claro"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_final"
BANNER_PATH = BASE_DIR / "banner.jpg"

# Mismo orden que label_model.labels del notebook (orden por frecuencia).
LABELS = ["Graduate", "Dropout", "Enrolled"]
LABELS_ES = {
    "Graduate": "Graduado",
    "Dropout": "Deserción",
    "Enrolled": "Matriculado (sigue estudiando)",
}
COLORES = {"Graduate": "#1E8449", "Dropout": "#C0392B", "Enrolled": "#B9770E"}


# ----------------------------------------------------------------------------
# Lector de modelos Spark ML (PipelineModel) en Python puro
# ----------------------------------------------------------------------------
def leer_metadata(ruta: Path) -> dict:
    """Lee metadata/part-* (una línea JSON) de un modelo o etapa de Spark."""
    archivos = sorted(
        p for p in (ruta / "metadata").iterdir()
        if p.is_file() and not p.name.startswith(("_", "."))
    )
    texto = "".join(p.read_text(encoding="utf-8") for p in archivos).strip()
    return json.loads(texto.splitlines()[0])


def leer_parquet(ruta: Path) -> list[dict]:
    """Lee una carpeta parquet de Spark (ignora _SUCCESS y .crc) como lista de dicts."""
    return parquet.read_table(str(ruta)).to_pylist()


def vector(v: dict) -> list[float]:
    """Convierte un VectorUDT guardado (type 0 = disperso, 1 = denso) a lista densa."""
    if v["type"] == 1:
        return [float(a) for a in v["values"]]
    denso = [0.0] * v["size"]
    for i, a in zip(v["indices"], v["values"]):
        denso[i] = float(a)
    return denso


def matriz(m: dict) -> list[list[float]]:
    """Convierte un MatrixUDT guardado a lista de filas (numRows x numCols)."""
    filas, cols = m["numRows"], m["numCols"]
    transpuesta = bool(m["isTransposed"])
    vals = m["values"]
    if m["type"] == 1:  # densa
        if transpuesta:  # fila-mayor
            return [[float(vals[i * cols + j]) for j in range(cols)] for i in range(filas)]
        return [[float(vals[j * filas + i]) for j in range(cols)] for i in range(filas)]
    # dispersa: CSC (por columnas) o CSR (si isTransposed)
    out = [[0.0] * cols for _ in range(filas)]
    ptrs, idx = m["colPtrs"], m["rowIndices"]
    if not transpuesta:
        for j in range(cols):
            for k in range(ptrs[j], ptrs[j + 1]):
                out[idx[k]][j] = float(vals[k])
    else:
        for i in range(filas):
            for k in range(ptrs[i], ptrs[i + 1]):
                out[i][idx[k]] = float(vals[k])
    return out


def normalizar(v: list[float]) -> list[float]:
    total = sum(v)
    return [a / total for a in v] if total > 0 else [1.0 / len(v)] * len(v)


# ---- Etapas -----------------------------------------------------------------
@dataclass
class VectorAssembler:
    entradas: list[str]
    salida: str

    def aplicar(self, cols: dict) -> None:
        cols[self.salida] = [float(cols[c]) for c in self.entradas]


@dataclass
class StandardScaler:
    entrada: str
    salida: str
    media: list[float]
    desv: list[float]
    con_media: bool
    con_desv: bool

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        out = []
        for i, v in enumerate(x):
            if self.con_media:
                v -= self.media[i]
            if self.con_desv:
                v = v / self.desv[i] if self.desv[i] != 0 else 0.0
            out.append(v)
        cols[self.salida] = out


@dataclass
class MinMaxScaler:
    entrada: str
    salida: str
    min_orig: list[float]
    max_orig: list[float]
    min_sal: float
    max_sal: float

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        out = []
        for i, v in enumerate(x):
            rango = self.max_orig[i] - self.min_orig[i]
            if rango != 0:
                esc = (v - self.min_orig[i]) / rango
            else:
                esc = 0.5
            out.append(esc * (self.max_sal - self.min_sal) + self.min_sal)
        cols[self.salida] = out


@dataclass
class RegresionLogistica:
    entrada: str
    salida_prob: str
    coef: list[list[float]]   # clases x características
    intercepto: list[float]

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        margenes = [
            sum(c * v for c, v in zip(fila, x)) + b
            for fila, b in zip(self.coef, self.intercepto)
        ]
        m = max(margenes)
        exps = [math.exp(a - m) for a in margenes]
        cols[self.salida_prob] = normalizar(exps)


@dataclass
class Arbol:
    nodos: dict[int, dict] = field(default_factory=dict)

    def estadisticas_hoja(self, x: list[float]) -> list[float]:
        n = self.nodos[0]
        while n["leftChild"] != -1:
            s = n["split"]
            if s["numCategories"] == -1:  # continua: izquierda si x <= umbral
                izq = x[s["featureIndex"]] <= s["leftCategoriesOrThreshold"][0]
            else:  # categórica
                izq = x[s["featureIndex"]] in s["leftCategoriesOrThreshold"]
            n = self.nodos[n["leftChild"] if izq else n["rightChild"]]
        return n["impurityStats"]


@dataclass
class Bosque:
    entrada: str
    salida_prob: str
    arboles: list[Arbol]
    pesos: list[float]

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        acum = None
        for arbol, peso in zip(self.arboles, self.pesos):
            p = normalizar(arbol.estadisticas_hoja(x))
            if acum is None:
                acum = [0.0] * len(p)
            acum = [a + peso * b for a, b in zip(acum, p)]
        cols[self.salida_prob] = normalizar(acum)


# ---- Constructores por clase de Spark --------------------------------------
def _nodos(filas: list[dict]) -> dict[int, dict]:
    return {f["id"]: f for f in filas}


def construir_etapa(ruta: Path):
    meta = leer_metadata(ruta)
    clase = meta["class"].split(".")[-1]
    p = meta.get("paramMap", {})
    data = ruta / "data"

    if clase == "VectorAssembler":
        return VectorAssembler(p["inputCols"], p.get("outputCol", meta["uid"] + "__output"))

    if clase == "StandardScalerModel":
        d = leer_parquet(data)[0]
        return StandardScaler(
            p["inputCol"], p["outputCol"], vector(d["mean"]), vector(d["std"]),
            p.get("withMean", False), p.get("withStd", True),
        )

    if clase == "MinMaxScalerModel":
        d = leer_parquet(data)[0]
        return MinMaxScaler(
            p["inputCol"], p["outputCol"], vector(d["originalMin"]), vector(d["originalMax"]),
            p.get("min", 0.0), p.get("max", 1.0),
        )

    if clase == "LogisticRegressionModel":
        d = leer_parquet(data)[0]
        return RegresionLogistica(
            p.get("featuresCol", "features"), p.get("probabilityCol", "probability"),
            matriz(d["coefficientMatrix"]), vector(d["interceptVector"]),
        )

    if clase == "DecisionTreeClassificationModel":
        arbol = Arbol(_nodos(leer_parquet(data)))
        return Bosque(p.get("featuresCol", "features"), p.get("probabilityCol", "probability"),
                      [arbol], [1.0])

    if clase == "RandomForestClassificationModel":
        por_arbol: dict[int, list[dict]] = {}
        for fila in leer_parquet(data):
            por_arbol.setdefault(fila["treeID"], []).append(fila["nodeData"])
        ids = sorted(por_arbol)
        pesos_meta = {f["treeID"]: f["weights"] for f in leer_parquet(ruta / "treesMetadata")}
        return Bosque(
            p.get("featuresCol", "features"), p.get("probabilityCol", "probability"),
            [Arbol(_nodos(por_arbol[i])) for i in ids],
            [float(pesos_meta.get(i, 1.0)) for i in ids],
        )

    raise NotImplementedError(
        f"Etapa de Spark no soportada: {clase}. "
        "Soportadas: VectorAssembler, StandardScaler, MinMaxScaler, "
        "LogisticRegression, DecisionTree y RandomForest."
    )


@dataclass
class Modelo:
    etapas: list
    col_prob: str

    def predecir(self, fila: dict) -> tuple[int, list[float]]:
        cols = dict(fila)
        for etapa in self.etapas:
            etapa.aplicar(cols)
        probs = cols[self.col_prob]
        return max(range(len(probs)), key=probs.__getitem__), probs


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo() -> Modelo:
    dir_etapas = MODEL_PATH / "stages"
    carpetas = sorted(
        (d for d in dir_etapas.iterdir() if d.is_dir()),
        key=lambda d: int(d.name.split("_", 1)[0]),
    )
    etapas = [construir_etapa(d) for d in carpetas]
    return Modelo(etapas, etapas[-1].salida_prob)


# ----------------------------------------------------------------------------
# Configuración y estilo
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Riesgo de deserción", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
    .stApp{background-color:#F4F7FB;}
    h1,h2,h3{color:#1F3864;}
    .resultado{padding:1.1rem 1.4rem;border-radius:12px;color:white;
               font-size:1.6rem;font-weight:700;text-align:center;margin:.5rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_dir():
    st.error(
        "No se encontró la carpeta **modelo_final** junto a app.py. "
        "Descárgala de Colab (donde se ejecutó `best_model.write().overwrite().save('modelo_final')`) "
        "y colócala en la misma carpeta que este archivo."
    )
    st.stop()

try:
    modelo = cargar_modelo()
except Exception as e:  # noqa: BLE001
    st.error("No se pudo cargar `modelo_final`.")
    st.exception(e)
    st.stop()

# ----------------------------------------------------------------------------
# Catálogos de códigos (dataset UCI / Kaggle "Predict Students' Dropout")
# ----------------------------------------------------------------------------
ESTADO_CIVIL = {1: "Soltero", 2: "Casado", 3: "Viudo", 4: "Divorciado",
                5: "Unión libre", 6: "Separado legalmente"}
MODO_APLICACION = {
    1: "1.ª fase – contingente general", 2: "Ordenanza 612/93",
    5: "1.ª fase – contingente especial (Azores)", 7: "Titulares de otros cursos superiores",
    10: "Ordenanza 854-B/99", 15: "Estudiante internacional (pregrado)",
    16: "1.ª fase – contingente especial (Madeira)", 17: "2.ª fase – contingente general",
    18: "3.ª fase – contingente general", 26: "Ordenanza 533-A/99, b2 (otro plan)",
    27: "Ordenanza 533-A/99, b3 (otra institución)", 39: "Mayores de 23 años",
    42: "Transferencia", 43: "Cambio de curso",
    44: "Titulares de diploma de especialización tecnológica",
    51: "Cambio de institución/curso", 53: "Titulares de ciclo corto",
    57: "Cambio de institución/curso (internacional)",
}
CURSO = {
    33: "Tecnologías de producción de biocombustibles", 171: "Animación y diseño multimedia",
    8014: "Servicio social (nocturno)", 9003: "Agronomía", 9070: "Diseño de comunicación",
    9085: "Enfermería veterinaria", 9119: "Ingeniería informática", 9130: "Equinocultura",
    9147: "Gestión", 9238: "Servicio social", 9254: "Turismo", 9500: "Enfermería",
    9556: "Higiene oral", 9670: "Publicidad y marketing", 9773: "Periodismo y comunicación",
    9853: "Educación básica", 9991: "Gestión (nocturno)",
}
CALIF_PREVIA = {
    1: "Educación secundaria", 2: "Superior – licenciatura", 3: "Superior – grado",
    4: "Superior – maestría", 5: "Superior – doctorado", 6: "Frecuencia de educación superior",
    9: "12.º año incompleto", 10: "11.º año incompleto", 12: "Otro 11.º año",
    14: "10.º año", 15: "10.º año incompleto", 19: "Educación básica 3.er ciclo",
    38: "Educación básica 2.º ciclo", 39: "Curso de especialización tecnológica",
    40: "Superior – grado (1.er ciclo)", 42: "Curso técnico superior profesional",
    43: "Superior – maestría (2.º ciclo)",
}
SI_NO = {0: "No", 1: "Sí"}
GENERO = {0: "Femenino", 1: "Masculino"}
HORARIO = {1: "Diurno", 0: "Nocturno"}


def select(label, opciones, valor_defecto, key, **kw):
    """Selectbox sobre un diccionario código -> descripción; devuelve el código."""
    codigos = list(opciones.keys())
    return st.selectbox(
        label, codigos, index=codigos.index(valor_defecto),
        format_func=lambda c: f"{c} – {opciones[c]}" if len(opciones) > 6 else opciones[c],
        key=key, **kw,
    )


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)
st.title("🎓 Predicción de deserción y éxito académico")
st.markdown(
    "**Instrucciones:** 1) complete los datos del estudiante en las tres pestañas; "
    "2) presione **Predecir**; 3) revise la clase predicha y la probabilidad de cada clase. "
    "El modelo clasifica al estudiante en *Graduado*, *Deserción* o *Matriculado*."
)

# ----------------------------------------------------------------------------
# Formulario
# ----------------------------------------------------------------------------
with st.form("datos"):
    tab1, tab2, tab3 = st.tabs(
        ["👤 Datos personales y familiares", "📚 Rendimiento académico", "📈 Contexto económico"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            edad = st.number_input("Edad al ingreso", 16, 70, 20)
            genero = select("Género", GENERO, 0, "genero")
            estado_civil = select("Estado civil", ESTADO_CIVIL, 1, "ec")
            nacionalidad = st.number_input("Nacionalidad (código)", 1, 109, 1,
                                           help="1 = portugués; otros códigos según el dataset.")
            internacional = select("Estudiante internacional", SI_NO, 0, "intl")
            desplazado = select("Desplazado de su residencia", SI_NO, 1, "desp")
        with c2:
            curso = select("Curso", CURSO, 9500, "curso")
            modo_app = select("Modo de aplicación", MODO_APLICACION, 1, "modo")
            orden_app = st.number_input("Orden de aplicación (0 = primera opción)", 0, 9, 1)
            calif_previa = select("Calificación previa", CALIF_PREVIA, 1, "prev")
            horario = select("Horario", HORARIO, 1, "hor")
            necesidades = select("Necesidades educativas especiales", SI_NO, 0, "nee")
        with c3:
            becario = select("Becario", SI_NO, 0, "beca")
            deudor = select("Deudor", SI_NO, 0, "deudor")
            matricula_al_dia = select("Matrícula al día", SI_NO, 1, "mat")
            madre_cal = st.number_input("Calificación de la madre (código)", 1, 44, 19)
            padre_cal = st.number_input("Calificación del padre (código)", 1, 44, 19)
            madre_ocu = st.number_input("Ocupación de la madre (código)", 0, 195, 5)
            padre_ocu = st.number_input("Ocupación del padre (código)", 0, 195, 5)

    with tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**1.er semestre**")
            s1_cred = st.number_input("Unidades convalidadas (1.er sem.)", 0, 30, 0)
            s1_insc = st.number_input("Unidades inscritas (1.er sem.)", 0, 40, 6)
            s1_eval = st.number_input("Evaluaciones (1.er sem.)", 0, 60, 8)
            s1_apro = st.number_input("Unidades aprobadas (1.er sem.)", 0, 40, 5)
            s1_nota = st.number_input("Nota promedio (1.er sem.)", 0.0, 20.0, 12.0, step=0.1)
            s1_sineval = st.number_input("Unidades sin evaluaciones (1.er sem.)", 0, 20, 0)
        with col_s2:
            st.markdown("**2.º semestre**")
            s2_cred = st.number_input("Unidades convalidadas (2.º sem.)", 0, 30, 0)
            s2_insc = st.number_input("Unidades inscritas (2.º sem.)", 0, 40, 6)
            s2_eval = st.number_input("Evaluaciones (2.º sem.)", 0, 60, 8)
            s2_apro = st.number_input("Unidades aprobadas (2.º sem.)", 0, 40, 5)
            s2_nota = st.number_input("Nota promedio (2.º sem.)", 0.0, 20.0, 12.0, step=0.1)
            s2_sineval = st.number_input("Unidades sin evaluaciones (2.º sem.)", 0, 20, 0)

    with tab3:
        e1, e2, e3 = st.columns(3)
        with e1:
            desempleo = st.number_input("Tasa de desempleo (%)", 0.0, 30.0, 10.8, step=0.1)
        with e2:
            inflacion = st.number_input("Tasa de inflación (%)", -5.0, 10.0, 1.4, step=0.1)
        with e3:
            pib = st.number_input("PIB (variación %)", -10.0, 10.0, 1.74, step=0.01)

    enviado = st.form_submit_button("🔮 Predecir", use_container_width=True)

# ----------------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------------
if enviado:
    if s1_apro > s1_insc or s2_apro > s2_insc:
        st.warning("Hay más unidades aprobadas que inscritas en algún semestre; revise los datos.")

    # Variables derivadas, calculadas igual que en el notebook (A.5)
    tasa_s1 = float(s1_apro) / s1_insc if s1_insc > 0 else 0.0
    tasa_s2 = float(s2_apro) / s2_insc if s2_insc > 0 else 0.0
    delta_nota = float(s2_nota) - float(s1_nota)

    fila = {
        "marital_status": estado_civil, "application_mode": modo_app,
        "application_order": orden_app, "course": curso,
        "daytime_evening_attendance": horario, "previous_qualification": calif_previa,
        "nacionality": nacionalidad, "mother_s_qualification": madre_cal,
        "father_s_qualification": padre_cal, "mother_s_occupation": madre_ocu,
        "father_s_occupation": padre_ocu, "displaced": desplazado,
        "educational_special_needs": necesidades, "debtor": deudor,
        "tuition_fees_up_to_date": matricula_al_dia, "gender": genero,
        "scholarship_holder": becario, "age_at_enrollment": edad,
        "international": internacional,
        "curricular_units_1st_sem_credited": s1_cred,
        "curricular_units_1st_sem_enrolled": s1_insc,
        "curricular_units_1st_sem_evaluations": s1_eval,
        "curricular_units_1st_sem_approved": s1_apro,
        "curricular_units_1st_sem_grade": float(s1_nota),
        "curricular_units_1st_sem_without_evaluations": s1_sineval,
        "curricular_units_2nd_sem_credited": s2_cred,
        "curricular_units_2nd_sem_enrolled": s2_insc,
        "curricular_units_2nd_sem_evaluations": s2_eval,
        "curricular_units_2nd_sem_approved": s2_apro,
        "curricular_units_2nd_sem_grade": float(s2_nota),
        "curricular_units_2nd_sem_without_evaluations": s2_sineval,
        "unemployment_rate": float(desempleo), "inflation_rate": float(inflacion),
        "gdp": float(pib),
        "tasa_aprob_s1": tasa_s1, "tasa_aprob_s2": tasa_s2, "delta_nota": delta_nota,
    }

    try:
        indice, probs = modelo.predecir(fila)
    except Exception as e:  # noqa: BLE001
        st.error("No se pudo predecir con el modelo. Revise que `modelo_final` corresponda al notebook.")
        st.exception(e)
        st.stop()

    clase = LABELS[indice]

    st.subheader("Resultado")
    st.markdown(
        f"<div class='resultado' style='background:{COLORES[clase]}'>"
        f"Clase predicha: {LABELS_ES[clase]} ({clase})</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("**Probabilidad de cada clase**")
        for etiqueta, p in zip(LABELS, probs):
            st.write(f"{LABELS_ES[etiqueta]} ({etiqueta}): **{p:.1%}**")
            st.progress(min(max(p, 0.0), 1.0))
    with col_b:
        df_prob = pd.DataFrame({"Probabilidad": probs}, index=[LABELS_ES[e] for e in LABELS])
        st.bar_chart(df_prob)

    with st.expander("Ver variables derivadas calculadas"):
        st.write(
            {
                "tasa_aprob_s1": round(tasa_s1, 3),
                "tasa_aprob_s2": round(tasa_s2, 3),
                "delta_nota": round(delta_nota, 3),
            }
        )

# ----------------------------------------------------------------------------
# Pie: autor
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Autor: {AUTOR}")
