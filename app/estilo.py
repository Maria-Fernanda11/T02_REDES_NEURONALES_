"""Estilo visual y piezas compartidas por las páginas de la aplicación."""

import streamlit as st

from motor import Modelo, leer_config_app

MARINO = "#0b2545"
AZUL = "#0b4f9c"
SUMA = "#2a78d6"       # par divergente: azul suma puntos, rojo resta
RESTA = "#e34948"
POBLACION = "#86b6ef"
TINTA_SUAVE = "#52514e"

CSS = f"""
<style>
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
.banda-superior {{
    background: linear-gradient(120deg, {MARINO} 0%, {AZUL} 100%);
    color: #fff; border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
}}
.banda-superior h1 {{ color: #fff; font-size: 1.9rem; margin: 0; letter-spacing: .2px; }}
.banda-superior p {{ color: #dbe7f7; margin: 6px 0 0 0; font-size: 1.02rem; }}
.marca {{ font-size: .8rem; text-transform: uppercase; letter-spacing: 2px; color: #9ec5f4; }}
.tarjeta {{
    background: #fff; border: 1px solid #e3e8ef; border-radius: 14px;
    padding: 20px 22px; box-shadow: 0 1px 3px rgba(11,37,69,.06);
}}
.puntaje-etiqueta {{ color: {TINTA_SUAVE}; font-size: .9rem; text-transform: uppercase; letter-spacing: 1px; }}
.puntaje-valor {{ color: {MARINO}; font-size: 4rem; font-weight: 700; line-height: 1.05; }}
.chip {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600;
         font-size: .95rem; color: #fff; margin-top: 6px; }}
.dato {{ color: {TINTA_SUAVE}; font-size: .85rem; margin-top: 14px; }}
.dato b {{ color: {MARINO}; font-size: 1.35rem; display: block; }}
.aviso {{ color: {TINTA_SUAVE}; font-size: .8rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 14px; }}
</style>
"""


def aplicar_estilo():
    st.markdown(CSS, unsafe_allow_html=True)


def encabezado(titulo, subtitulo):
    st.markdown(f"""<div class="banda-superior"><div class="marca">¿Me prestas? · Simulador de crédito</div>
        <h1>{titulo}</h1><p>{subtitulo}</p></div>""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(carpeta):
    return Modelo(carpeta)


def barra_lateral():
    """Selector de modelo y enlaces. Devuelve (modelo, configuración de la app)."""
    config = leer_config_app()
    with st.sidebar:
        st.markdown("### 🏦 ¿Me prestas?")
        st.caption("Estimación del riesgo de incumplimiento para créditos de consumo a 36 meses.")
        opciones = {"Regresión logística (recomendado)": config["modelo"],
                    "Red neuronal (comparación)": config["modelo_comparacion"]}
        eleccion = st.radio("Modelo", list(opciones), key="modelo_elegido",
                            help="La regresión logística tuvo igual o mejor desempeño que la red en "
                                 "datos que ninguno de los dos vio, y sus puntos se explican variable "
                                 "por variable. Ver «Cómo funciona».")
        modelo = cargar_modelo(opciones[eleccion])
        st.divider()
        st.markdown("**Enlaces**")
        for texto, clave in (("📄 Reporte técnico", "enlace_reporte"), ("🎬 Video promocional", "enlace_video"),
                             ("💻 Repositorio", "enlace_repositorio")):
            if config[clave]:
                st.link_button(texto, config[clave], width="stretch")
            else:
                st.caption(f"{texto}: disponible pronto")
        st.divider()
        st.markdown('<p class="aviso">Proyecto académico (Universidad Nacional de Colombia, 2026-02). '
                    'Basado en datos históricos de Lending Club 2007–2012. No es una decisión de crédito real.</p>',
                    unsafe_allow_html=True)
    return modelo, config
