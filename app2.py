# app.py — Predicción de deserción (Streamlit + modelo Spark ML sin Spark)
#
# Versión 2: modelo BINARIO (Retención vs. Deserción) con variables seleccionadas.
#
# Archivos necesarios en el repositorio (misma carpeta):
#   app.py | modelo_final_binario/ (guardado desde Colab, sección E.6) | banner.jpg (opcional)
#   requirements.txt: streamlit, pyarrow, pandas
#   (no se necesita pyspark ni Java)
#
# El formulario se ADAPTA al modelo: solo pide los datos que el modelo realmente usa
# (los que quedaron seleccionados en el notebook), así que no hay que editar la app
# si se vuelve a entrenar con otras variables.
#
# Ejecutar en local:  streamlit run app.py

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as parquet
import streamlit as st

AUTOR = "Alfredo Diaz Claro"

BASE_DIR = Path(__file__).resolve().parent
# Se usa la primera carpeta que exista (la nueva tiene prioridad).
MODEL_PATH = next(
    (BASE_DIR / n for n in ("modelo_final_binario", "modelo_final") if (BASE_DIR / n).is_dir()),
    BASE_DIR / "modelo_final_binario",
)
BANNER_PATH = BASE_DIR / "banner.jpg"

# Modelo binario: mismo orden que label_bin del notebook (0 = Retención, 1 = Deserción).
LABELS_BIN = ["Retención", "Deserción"]
COLORES_BIN = {"Retención": "#1E8449", "Deserción": "#C0392B"}
# Compatibilidad con el modelo antiguo de 3 clases (orden de label_model.labels).
LABELS_3 = ["Graduate", "Dropout", "Enrolled"]
LABELS_3_ES = {"Graduate": "Graduado", "Dropout": "Deserción", "Enrolled": "Matriculado (sigue estudiando)"}
COLORES_3 = {"Graduate": "#1E8449", "Dropout": "#C0392B", "Enrolled": "#B9770E"}


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


def sigmoide(m: float) -> float:
    if m >= 0:
        return 1.0 / (1.0 + math.exp(-m))
    e = math.exp(m)
    return e / (1.0 + e)


# ---- Etapas -----------------------------------------------------------------
# Cada etapa expone: aplicar(cols), consume() -> nombres que lee, produce() -> nombres que crea.
@dataclass
class TransformadorSQL:
    """SQLTransformer: crea las variables derivadas. La sentencia de Spark
    ("SELECT *, (expr) AS nombre, ... FROM __THIS__") se ejecuta con sqlite3 (biblioteca estándar)
    sobre una tabla de una fila, así las fórmulas del notebook no se replican a mano."""
    sentencia: str
    salidas: list[str]

    def consume(self) -> list[str]:
        return re.findall(r"[A-Za-z_][A-Za-z_0-9]*", self.sentencia)

    def produce(self) -> list[str]:
        return self.salidas

    def aplicar(self, cols: dict) -> None:
        nombres = [k for k, v in cols.items() if isinstance(v, (int, float))]
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE t (" + ", ".join(f'"{k}" REAL' for k in nombres) + ")")
            con.execute(f"INSERT INTO t VALUES ({','.join('?' * len(nombres))})",
                        [float(cols[k]) for k in nombres])  # REAL => "/" es división decimal (como en Spark)
            cur = con.execute(self.sentencia.replace("__THIS__", "t"))
            valores = cur.fetchone()
            columnas = [d[0] for d in cur.description]
        finally:
            con.close()
        for nombre, valor in zip(columnas, valores):
            if nombre in self.salidas:
                cols[nombre] = float(valor) if valor is not None else 0.0


@dataclass
class VectorAssembler:
    entradas: list[str]
    salida: str

    def consume(self) -> list[str]:
        return self.entradas

    def produce(self) -> list[str]:
        return [self.salida]

    def aplicar(self, cols: dict) -> None:
        resultado: list[float] = []
        for c in self.entradas:
            v = cols[c]
            if isinstance(v, list):
                resultado.extend(v)
            else:
                resultado.append(float(v))
        cols[self.salida] = resultado


@dataclass
class StandardScaler:
    entrada: str
    salida: str
    media: list[float]
    desv: list[float]
    con_media: bool
    con_desv: bool

    def consume(self) -> list[str]:
        return [self.entrada]

    def produce(self) -> list[str]:
        return [self.salida]

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

    def consume(self) -> list[str]:
        return [self.entrada]

    def produce(self) -> list[str]:
        return [self.salida]

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
class IndexadorTexto:
    """StringIndexerModel (una o varias columnas). Si una columna de entrada no
    existe (p. ej. la etiqueta en predicción), Spark omite esa columna; aquí igual."""
    entradas: list[str]
    salidas: list[str]
    etiquetas: list[list[str]]
    invalidos: str = "error"

    def consume(self) -> list[str]:
        return self.entradas

    def produce(self) -> list[str]:
        return self.salidas

    def aplicar(self, cols: dict) -> None:
        for ent, sal, etiq in zip(self.entradas, self.salidas, self.etiquetas):
            if ent not in cols:
                continue
            v = cols[ent]
            candidatos = [str(v)]
            if isinstance(v, (int, float)) and float(v).is_integer():
                candidatos += [str(int(v)), str(float(v))]  # Spark castea números a texto
            for c in candidatos:
                if c in etiq:
                    cols[sal] = float(etiq.index(c))
                    break
            else:
                if self.invalidos == "keep":
                    cols[sal] = float(len(etiq))
                else:
                    raise ValueError(f"Valor '{v}' no visto en el entrenamiento para '{ent}'.")


@dataclass
class OneHot:
    entradas: list[str]
    salidas: list[str]
    tamanos: list[int]  # tamaño final del vector de cada columna

    def consume(self) -> list[str]:
        return self.entradas

    def produce(self) -> list[str]:
        return self.salidas

    def aplicar(self, cols: dict) -> None:
        for ent, sal, n in zip(self.entradas, self.salidas, self.tamanos):
            vec = [0.0] * n
            i = int(cols[ent])
            if 0 <= i < n:
                vec[i] = 1.0
            cols[sal] = vec


@dataclass
class RegresionLogistica:
    entrada: str
    salida_prob: str
    coef: list[list[float]]   # filas x características (binaria: 1 fila; multiclase: una por clase)
    intercepto: list[float]

    def consume(self) -> list[str]:
        return [self.entrada]

    def produce(self) -> list[str]:
        return [self.salida_prob]

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        margenes = [
            sum(c * v for c, v in zip(fila, x)) + b
            for fila, b in zip(self.coef, self.intercepto)
        ]
        if len(margenes) == 1:  # binomial: una sola fila de coeficientes -> sigmoide
            p1 = sigmoide(margenes[0])
            cols[self.salida_prob] = [1.0 - p1, p1]
            return
        m = max(margenes)  # multinomial: softmax
        exps = [math.exp(a - m) for a in margenes]
        cols[self.salida_prob] = normalizar(exps)


@dataclass
class Arbol:
    nodos: dict[int, dict] = field(default_factory=dict)

    def hoja(self, x: list[float]) -> dict:
        n = self.nodos[0]
        while n["leftChild"] != -1:
            s = n["split"]
            if s["numCategories"] == -1:  # continua: izquierda si x <= umbral
                izq = x[s["featureIndex"]] <= s["leftCategoriesOrThreshold"][0]
            else:  # categórica
                izq = x[s["featureIndex"]] in s["leftCategoriesOrThreshold"]
            n = self.nodos[n["leftChild"] if izq else n["rightChild"]]
        return n

    def estadisticas_hoja(self, x: list[float]) -> list[float]:
        return self.hoja(x)["impurityStats"]


@dataclass
class Bosque:
    entrada: str
    salida_prob: str
    arboles: list[Arbol]
    pesos: list[float]

    def consume(self) -> list[str]:
        return [self.entrada]

    def produce(self) -> list[str]:
        return [self.salida_prob]

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        acum = None
        for arbol, peso in zip(self.arboles, self.pesos):
            p = normalizar(arbol.estadisticas_hoja(x))
            if acum is None:
                acum = [0.0] * len(p)
            acum = [a + peso * b for a, b in zip(acum, p)]
        cols[self.salida_prob] = normalizar(acum)


@dataclass
class ImpulsoGradiente:
    """GBTClassificationModel (solo binario): suma ponderada de árboles de regresión;
    P(clase 1) = 1 / (1 + exp(-2 * margen)), igual que Spark."""
    entrada: str
    salida_prob: str
    arboles: list[Arbol]
    pesos: list[float]

    def consume(self) -> list[str]:
        return [self.entrada]

    def produce(self) -> list[str]:
        return [self.salida_prob]

    def aplicar(self, cols: dict) -> None:
        x = cols[self.entrada]
        margen = sum(peso * float(a.hoja(x)["prediction"]) for a, peso in zip(self.arboles, self.pesos))
        p1 = sigmoide(2.0 * margen)
        cols[self.salida_prob] = [1.0 - p1, p1]


# ---- Constructores por clase de Spark --------------------------------------
def _nodos(filas: list[dict]) -> dict[int, dict]:
    return {f["id"]: f for f in filas}


def _arboles_ensamble(ruta: Path) -> tuple[list[Arbol], list[float]]:
    """Lee un ensamble de árboles (RandomForest o GBT): data/ (treeID, nodeData) + treesMetadata/ (weights)."""
    por_arbol: dict[int, list[dict]] = {}
    for fila in leer_parquet(ruta / "data"):
        por_arbol.setdefault(fila["treeID"], []).append(fila["nodeData"])
    ids = sorted(por_arbol)
    pesos_meta = {f["treeID"]: f["weights"] for f in leer_parquet(ruta / "treesMetadata")}
    return [Arbol(_nodos(por_arbol[i])) for i in ids], [float(pesos_meta.get(i, 1.0)) for i in ids]


def construir_etapa(ruta: Path):
    meta = leer_metadata(ruta)
    clase = meta["class"].split(".")[-1]
    p = meta.get("paramMap", {})
    data = ruta / "data"

    if clase == "SQLTransformer":
        sentencia = p["statement"]
        return TransformadorSQL(sentencia, re.findall(r"\bAS\s+([A-Za-z_][A-Za-z_0-9]*)", sentencia, flags=re.I))

    if clase == "VectorAssembler":
        return VectorAssembler(p["inputCols"], p.get("outputCol", meta["uid"] + "__output"))

    if clase == "StringIndexerModel":
        d = leer_parquet(data)[0]
        if d.get("labelsArray"):
            etiquetas = [list(x) for x in d["labelsArray"]]
        else:
            etiquetas = [list(d["labels"])]
        entradas = p["inputCols"] if "inputCols" in p else [p["inputCol"]]
        salidas = p["outputCols"] if "outputCols" in p else [p["outputCol"]]
        return IndexadorTexto(entradas, salidas, etiquetas, p.get("handleInvalid", "error"))

    if clase == "OneHotEncoderModel":
        d = leer_parquet(data)[0]
        quitar_ultima = p.get("dropLast", True)
        conservar = p.get("handleInvalid", "error") == "keep"
        tamanos = []
        for s in d["categorySizes"]:
            if not quitar_ultima and conservar:
                tamanos.append(s + 1)
            elif quitar_ultima and not conservar:
                tamanos.append(s - 1)
            else:
                tamanos.append(s)
        entradas = p["inputCols"] if "inputCols" in p else [p["inputCol"]]
        salidas = p["outputCols"] if "outputCols" in p else [p["outputCol"]]
        return OneHot(entradas, salidas, tamanos)

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
        arboles, pesos = _arboles_ensamble(ruta)
        return Bosque(p.get("featuresCol", "features"), p.get("probabilityCol", "probability"), arboles, pesos)

    if clase == "GBTClassificationModel":
        arboles, pesos = _arboles_ensamble(ruta)
        return ImpulsoGradiente(p.get("featuresCol", "features"), p.get("probabilityCol", "probability"),
                                arboles, pesos)

    raise NotImplementedError(
        f"Etapa de Spark no soportada: {clase}. "
        "Soportadas: SQLTransformer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, "
        "MinMaxScaler, LogisticRegression, DecisionTree, RandomForest y GBT."
    )


@dataclass
class Modelo:
    etapas: list
    col_prob: str

    @property
    def derivadas(self) -> list[str]:
        return [n for e in self.etapas if isinstance(e, TransformadorSQL) for n in e.salidas]

    def requeridas(self, conocidas: set[str]) -> tuple[set[str], set[str]]:
        """Datos CRUDOS que el modelo necesita del usuario (recorriendo las etapas en orden).
        Devuelve (requeridas, desconocidas): desconocidas = entradas que ni se producen en el
        pipeline ni existen en el formulario."""
        producidas: set[str] = set()
        req: set[str] = set()
        desconocidas: set[str] = set()
        for e in self.etapas:
            for c in e.consume():
                if c in producidas:
                    continue
                if c in conocidas:
                    req.add(c)
                elif isinstance(e, (VectorAssembler, IndexadorTexto)):
                    desconocidas.add(c)
            producidas.update(e.produce())
        return req, desconocidas

    def predecir(self, fila: dict) -> tuple[int, list[float], dict]:
        cols = dict(fila)
        for etapa in self.etapas:
            etapa.aplicar(cols)
        probs = cols[self.col_prob]
        return max(range(len(probs)), key=probs.__getitem__), probs, cols


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo() -> Modelo:
    dir_etapas = MODEL_PATH / "stages"
    carpetas = sorted(
        (d for d in dir_etapas.iterdir() if d.is_dir()),
        key=lambda d: int(d.name.split("_", 1)[0]),
    )
    etapas = [construir_etapa(d) for d in carpetas]
    col_prob = next(e.salida_prob for e in reversed(etapas) if hasattr(e, "salida_prob"))
    return Modelo(etapas, col_prob)


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
        "No se encontró la carpeta **modelo_final_binario** junto a app.py. "
        "Descárgala de Colab (donde se ejecutó `best_bin.write().overwrite().save('modelo_final_binario')`) "
        "y colócala en la misma carpeta que este archivo."
    )
    st.stop()

try:
    modelo = cargar_modelo()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar `{MODEL_PATH.name}`.")
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
# Campos del formulario: columna del modelo -> cómo se pide al usuario.
# grupo = pestaña | col = columna dentro de la pestaña
# ----------------------------------------------------------------------------
def _sel(grupo, col, etiqueta, opciones, defecto):
    return dict(grupo=grupo, col=col, tipo="sel", etiqueta=etiqueta, opciones=opciones, defecto=defecto)


def _num(grupo, col, etiqueta, mn, mx, defecto, **kw):
    return dict(grupo=grupo, col=col, tipo="num", etiqueta=etiqueta, mn=mn, mx=mx, defecto=defecto, kw=kw)


CAMPOS = {
    # --- Personales y familiares
    "age_at_enrollment": _num("personal", 0, "Edad al ingreso", 16, 70, 20),
    "gender": _sel("personal", 0, "Género", GENERO, 0),
    "marital_status": _sel("personal", 0, "Estado civil", ESTADO_CIVIL, 1),
    "nacionality": _num("personal", 0, "Nacionalidad (código)", 1, 109, 1,
                        help="1 = portugués; otros códigos según el dataset."),
    "international": _sel("personal", 0, "Estudiante internacional", SI_NO, 0),
    "displaced": _sel("personal", 0, "Desplazado de su residencia", SI_NO, 1),
    "course": _sel("personal", 1, "Curso", CURSO, 9500),
    "application_mode": _sel("personal", 1, "Modo de aplicación", MODO_APLICACION, 1),
    "application_order": _num("personal", 1, "Orden de aplicación (0 = primera opción)", 0, 9, 1),
    "previous_qualification": _sel("personal", 1, "Calificación previa", CALIF_PREVIA, 1),
    "daytime_evening_attendance": _sel("personal", 1, "Horario", HORARIO, 1),
    "educational_special_needs": _sel("personal", 1, "Necesidades educativas especiales", SI_NO, 0),
    "scholarship_holder": _sel("personal", 2, "Becario", SI_NO, 0),
    "debtor": _sel("personal", 2, "Deudor", SI_NO, 0),
    "tuition_fees_up_to_date": _sel("personal", 2, "Matrícula al día", SI_NO, 1),
    "mother_s_qualification": _num("personal", 2, "Calificación de la madre (código)", 1, 44, 19),
    "father_s_qualification": _num("personal", 2, "Calificación del padre (código)", 1, 44, 19),
    "mother_s_occupation": _num("personal", 2, "Ocupación de la madre (código)", 0, 195, 5),
    "father_s_occupation": _num("personal", 2, "Ocupación del padre (código)", 0, 195, 5),
    # --- Rendimiento académico (col 0 = 1.er semestre, col 1 = 2.º semestre)
    "curricular_units_1st_sem_credited": _num("academico", 0, "Unidades convalidadas (1.er sem.)", 0, 30, 0),
    "curricular_units_1st_sem_enrolled": _num("academico", 0, "Unidades inscritas (1.er sem.)", 0, 40, 6),
    "curricular_units_1st_sem_evaluations": _num("academico", 0, "Evaluaciones (1.er sem.)", 0, 60, 8),
    "curricular_units_1st_sem_approved": _num("academico", 0, "Unidades aprobadas (1.er sem.)", 0, 40, 5),
    "curricular_units_1st_sem_grade": _num("academico", 0, "Nota promedio (1.er sem.)", 0.0, 20.0, 12.0, step=0.1),
    "curricular_units_1st_sem_without_evaluations": _num("academico", 0, "Unidades sin evaluaciones (1.er sem.)", 0, 20, 0),
    "curricular_units_2nd_sem_credited": _num("academico", 1, "Unidades convalidadas (2.º sem.)", 0, 30, 0),
    "curricular_units_2nd_sem_enrolled": _num("academico", 1, "Unidades inscritas (2.º sem.)", 0, 40, 6),
    "curricular_units_2nd_sem_evaluations": _num("academico", 1, "Evaluaciones (2.º sem.)", 0, 60, 8),
    "curricular_units_2nd_sem_approved": _num("academico", 1, "Unidades aprobadas (2.º sem.)", 0, 40, 5),
    "curricular_units_2nd_sem_grade": _num("academico", 1, "Nota promedio (2.º sem.)", 0.0, 20.0, 12.0, step=0.1),
    "curricular_units_2nd_sem_without_evaluations": _num("academico", 1, "Unidades sin evaluaciones (2.º sem.)", 0, 20, 0),
    # --- Contexto económico
    "unemployment_rate": _num("economico", 0, "Tasa de desempleo (%)", 0.0, 30.0, 10.8, step=0.1),
    "inflation_rate": _num("economico", 1, "Tasa de inflación (%)", -5.0, 10.0, 1.4, step=0.1),
    "gdp": _num("economico", 2, "PIB (variación %)", -10.0, 10.0, 1.74, step=0.01),
}
GRUPOS = [  # (clave, título de la pestaña, nº de columnas)
    ("personal", "👤 Datos personales y familiares", 3),
    ("academico", "📚 Rendimiento académico", 2),
    ("economico", "📈 Contexto económico", 3),
]


def dibujar(col: str, spec: dict):
    if spec["tipo"] == "sel":
        return select(spec["etiqueta"], spec["opciones"], spec["defecto"], key=col)
    return st.number_input(spec["etiqueta"], spec["mn"], spec["mx"], spec["defecto"], key=col, **spec["kw"])


requeridas, desconocidas = modelo.requeridas(set(CAMPOS))
if desconocidas:
    st.error(
        "El modelo usa variables que el formulario no conoce: "
        + ", ".join(sorted(desconocidas))
        + ". Agréguelas al diccionario `CAMPOS` de app.py."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), use_container_width=True)
st.title("🎓 Predicción de deserción estudiantil")
st.markdown(
    "**Instrucciones:** 1) complete los datos del estudiante en las pestañas; "
    "2) presione **Predecir**; 3) revise la clase predicha y la probabilidad de deserción. "
    "El modelo clasifica al estudiante en *Retención* o *Deserción*."
)
st.caption(
    f"El modelo usa {len(requeridas)} datos del estudiante: los que resultaron más útiles en el análisis "
    "de selección de variables del notebook. Por eso el formulario es más corto que el original."
)

# ----------------------------------------------------------------------------
# Formulario (solo los campos que el modelo usa)
# ----------------------------------------------------------------------------
grupos_activos = [g for g in GRUPOS if any(c in requeridas and s["grupo"] == g[0] for c, s in CAMPOS.items())]
valores: dict = {}
with st.form("datos"):
    tabs = st.tabs([titulo for _, titulo, _ in grupos_activos])
    for tab, (grupo, _, n_cols) in zip(tabs, grupos_activos):
        with tab:
            columnas = st.columns(n_cols)
            if grupo == "academico":
                for k, titulo in enumerate(["**1.er semestre**", "**2.º semestre**"]):
                    with columnas[k]:
                        st.markdown(titulo)
            for col, spec in CAMPOS.items():
                if col in requeridas and spec["grupo"] == grupo:
                    with columnas[spec["col"]]:
                        valores[col] = dibujar(col, spec)
    enviado = st.form_submit_button("🔮 Predecir", use_container_width=True)

# ----------------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------------
if enviado:
    for sem in ("1st", "2nd"):
        ins, apr = f"curricular_units_{sem}_sem_enrolled", f"curricular_units_{sem}_sem_approved"
        if ins in valores and apr in valores and valores[apr] > valores[ins]:
            st.warning("Hay más unidades aprobadas que inscritas en algún semestre; revise los datos.")
            break

    try:
        indice, probs, cols = modelo.predecir(dict(valores))  # las variables derivadas las calcula el modelo
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo predecir con el modelo. Revise que `{MODEL_PATH.name}` corresponda al notebook.")
        st.exception(e)
        st.stop()

    if len(probs) == 2:   # modelo binario (versión actual)
        etiquetas, nombres, colores = LABELS_BIN, LABELS_BIN, COLORES_BIN
    else:                 # modelo antiguo de 3 clases
        etiquetas = LABELS_3
        nombres = [LABELS_3_ES[e] for e in LABELS_3]
        colores = {LABELS_3_ES[e]: COLORES_3[e] for e in LABELS_3}
    clase = nombres[indice]

    st.subheader("Resultado")
    st.markdown(
        f"<div class='resultado' style='background:{colores[clase]}'>Predicción: {clase}</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if len(probs) == 2:
            st.metric("Probabilidad de deserción", f"{probs[1]:.1%}")
        st.markdown("**Probabilidad de cada clase**")
        for nombre, p in zip(nombres, probs):
            st.write(f"{nombre}: **{p:.1%}**")
            st.progress(min(max(p, 0.0), 1.0))
    with col_b:
        st.bar_chart(pd.DataFrame({"Probabilidad": probs}, index=nombres))

    if len(probs) == 2:
        st.caption(
            "Se predice *Deserción* cuando su probabilidad supera 50 %. El modelo se entrenó dando más peso a la "
            "clase Deserción, por lo que tiende a priorizar detectar a quienes podrían desertar, aunque eso "
            "genere algunas falsas alarmas. Es una herramienta de apoyo, no un veredicto."
        )

    derivadas = {n: round(cols[n], 3) for n in modelo.derivadas if n in cols}
    if derivadas:
        with st.expander("Ver variables derivadas calculadas por el modelo"):
            st.write(derivadas)

# ----------------------------------------------------------------------------
# Pie: autor
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Autor: {AUTOR}")