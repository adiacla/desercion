# app.py — Predicción de deserción y éxito académico (Streamlit + Spark ML)
#
# Archivos necesarios en el repositorio (misma carpeta):
#   app.py | modelo_final/ (guardado desde Colab) | banner.jpg (opcional)
#   requirements.txt: streamlit, pyspark (misma versión que en Colab), pandas
#   packages.txt: default-jre
#
# Ejecutar en local:  streamlit run app.py

import os

import pandas as pd
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

AUTOR = "Alfredo Diaz Claro"  # <-- cambia por el nombre del estudiante

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_final")
BANNER_PATH = os.path.join(BASE_DIR, "banner.jpg")

# Mismo orden que label_model.labels del notebook (orden por frecuencia).
# El modelo devuelve 0, 1, 2 y esta lista lo convierte de vuelta a texto.
LABELS = ["Graduate", "Dropout", "Enrolled"]
LABELS_ES = {
    "Graduate": "Graduado",
    "Dropout": "Deserción",
    "Enrolled": "Matriculado (sigue estudiando)",
}
COLORES = {"Graduate": "#1E8449", "Dropout": "#C0392B", "Enrolled": "#B9770E"}

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


# ----------------------------------------------------------------------------
# Carga del modelo (una sola vez)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Iniciando Spark y cargando el modelo...")
def cargar_modelo():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("app_desercion")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    return spark, PipelineModel.load(MODEL_PATH)


if not os.path.isdir(MODEL_PATH):
    st.error(
        "No se encontró la carpeta **modelo_final** junto a app.py. "
        "Descárgala de Colab (donde se ejecutó `best_model.write().overwrite().save('modelo_final')`) "
        "y colócala en la misma carpeta que este archivo."
    )
    st.stop()

spark, modelo = cargar_modelo()

# Columnas EXACTAS (nombre y tipo) con las que se entrenó el pipeline.
COLUMNAS_INT = [
    "marital_status", "application_mode", "application_order", "course",
    "daytime_evening_attendance", "previous_qualification", "nacionality",
    "mother_s_qualification", "father_s_qualification", "mother_s_occupation",
    "father_s_occupation", "displaced", "educational_special_needs", "debtor",
    "tuition_fees_up_to_date", "gender", "scholarship_holder", "age_at_enrollment",
    "international",
    "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_without_evaluations",
]
COLUMNAS_DOUBLE = [
    "curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade",
    "unemployment_rate", "inflation_rate", "gdp",
    "tasa_aprob_s1", "tasa_aprob_s2", "delta_nota",   # variables derivadas
]
ESQUEMA = StructType(
    [StructField(c, IntegerType(), True) for c in COLUMNAS_INT]
    + [StructField(c, DoubleType(), True) for c in COLUMNAS_DOUBLE]
)

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
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)
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

    # Las 3 variables derivadas se calculan igual que en el notebook (A.5)
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

    # Orden y tipos según el esquema (enteros nativos de Python, no numpy)
    valores = tuple(
        int(fila[f.name]) if isinstance(f.dataType, IntegerType) else float(fila[f.name])
        for f in ESQUEMA.fields
    )
    sdf = spark.createDataFrame([valores], schema=ESQUEMA)

    try:
        resultado = modelo.transform(sdf).select("prediction", "probability").first()
    except Exception as e:  # noqa: BLE001
        st.error(
            "No se pudo predecir con el modelo. Verifique que `modelo_final` se generó con el "
            "notebook y que la versión de pyspark es la misma que la usada en Colab."
        )
        st.exception(e)
        st.stop()

    clase = LABELS[int(resultado["prediction"])]
    probs = [float(p) for p in resultado["probability"].toArray()]

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
        df_prob = pd.DataFrame(
            {"Probabilidad": probs}, index=[LABELS_ES[e] for e in LABELS]
        )
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