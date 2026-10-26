"""
Evalúa en prueba los modelos guardados de la Parte 1 y genera las figuras del reporte.

    python evaluar_pd.py                         # compara models/pd_logistica y models/pd
    python evaluar_pd.py --modelos models/pd     # un solo modelo

No reentrena: carga red.npz y preprocesador.joblib de cada carpeta. Genera:
    reports/evaluacion_pd.csv          métricas en prueba: total, dti <= 30 y dti > 30
    reports/figures/pd_roc.png         curvas ROC
    reports/figures/pd_calibracion.png curva de calibración (deciles de PD)
    reports/figures/pd_ks.png          distribución acumulada del puntaje en buenos y malos

Sobre dti > 30: el máximo de dti en entrenamiento es 29.99, pero Lending Club subió su tope
durante 2012 y el 6 % de la prueba tiene dti entre 30 y 35. En esa franja la red extrapola,
por eso las métricas se reportan con y sin esos créditos.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metricas
from src.red_neuronal import RedNeuronal
from src.scorecard import Scorecard

FIGURAS = Path("reports/figures")
# Paleta categórica validada (orden fijo: el color sigue al modelo, no a su posición)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA_SUAVE, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def cargar_modelo(carpeta):
    carpeta = Path(carpeta)
    metadata = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
    return {
        "nombre": "Regresión logística" if not metadata["config"]["red"]["capas_ocultas"]
                  else "MLP " + "-".join(map(str, metadata["config"]["red"]["capas_ocultas"])),
        "red": RedNeuronal.cargar(carpeta / "red.npz"),
        "preprocesador": joblib.load(carpeta / "preprocesador.joblib"),
        "metadata": metadata,
    }


def predecir(modelo, X):
    return modelo["red"].predecir(modelo["preprocesador"].transform(X))


def curva_roc(y, p):
    orden = np.argsort(-p, kind="mergesort")
    y = np.asarray(y)[orden]
    tpr = np.r_[0, np.cumsum(y == 1) / np.sum(y == 1)]
    fpr = np.r_[0, np.cumsum(y == 0) / np.sum(y == 0)]
    return fpr, tpr


def estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=12)
    ax.set_xlabel(xlabel, color=TINTA_SUAVE)
    ax.set_ylabel(ylabel, color=TINTA_SUAVE)
    ax.grid(color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_SUAVE)
    ax.tick_params(colors=TINTA_SUAVE)


def figura_roc(y, predicciones, ruta):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, linewidth=1, linestyle="--", label="Azar (AUC = 0.5)")
    for (nombre, p), color in zip(predicciones.items(), COLORES):
        fpr, tpr = curva_roc(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{nombre} (AUC = {metricas.auc(y, p):.3f})")
    estilo(ax, "Curva ROC en prueba (jul–dic 2012)",
           "Tasa de falsos positivos (buenos rechazados)", "Tasa de verdaderos positivos (malos detectados)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_calibracion(y, predicciones, ruta):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    limite = 0
    for (nombre, p), color in zip(predicciones.items(), COLORES):
        tabla = metricas.curva_calibracion(y, p, n_grupos=10)
        limite = max(limite, tabla["p_promedio"].max(), tabla["tasa_observada"].max())
        ax.plot(tabla["p_promedio"], tabla["tasa_observada"], color=color, linewidth=2,
                marker="o", markersize=6, label=f"{nombre} (Brier = {metricas.brier(y, p):.4f})")
    limite *= 1.08
    ax.plot([0, limite], [0, limite], color=TINTA_SUAVE, linewidth=1, linestyle="--",
            label="Calibración perfecta")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    estilo(ax, "Curva de calibración en prueba (deciles de PD)",
           "PD promedio predicha", "Tasa de malos observada")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_ks(y, puntaje, nombre, ruta):
    y = np.asarray(y)
    rejilla = np.linspace(np.percentile(puntaje, 0.5), np.percentile(puntaje, 99.5), 400)
    f_malos = np.searchsorted(np.sort(puntaje[y == 1]), rejilla, side="right") / np.sum(y == 1)
    f_buenos = np.searchsorted(np.sort(puntaje[y == 0]), rejilla, side="right") / np.sum(y == 0)
    i = int(np.argmax(f_malos - f_buenos))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(rejilla, f_malos, color=COLORES[1], linewidth=2, label="Malos (y = 1)")
    ax.plot(rejilla, f_buenos, color=COLORES[0], linewidth=2, label="Buenos (y = 0)")
    ax.vlines(rejilla[i], f_buenos[i], f_malos[i], color=TINTA, linewidth=1.5)
    ax.annotate(f"KS = {f_malos[i] - f_buenos[i]:.3f}\n(puntaje {rejilla[i]:.0f})",
                (rejilla[i], (f_malos[i] + f_buenos[i]) / 2), xytext=(8, 0),
                textcoords="offset points", va="center", color=TINTA)
    estilo(ax, f"Distribución acumulada del puntaje en prueba — {nombre}",
           "Puntaje de la scorecard", "Proporción acumulada")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="+", default=["models/pd_logistica", "models/pd"])
    parser.add_argument("--datos", default="data/processed/pd_limpio.parquet")
    args = parser.parse_args(argumentos)

    modelos = [cargar_modelo(c) for c in args.modelos]
    datos = pd.read_parquet(args.datos)
    test = datos[datos["particion"] == "test"]
    y = test["y"].to_numpy()
    predicciones = {m["nombre"]: predecir(m, test) for m in modelos}

    grupos = {"total": np.ones(len(test), bool),
              "dti <= 30": (test["dti"] <= 30).to_numpy(),
              "dti > 30": (test["dti"] > 30).to_numpy()}
    filas = []
    for m in modelos:
        p = predicciones[m["nombre"]]
        for grupo, mascara in grupos.items():
            filas.append({"modelo": m["nombre"], "grupo": grupo,
                          "parametros": m["metadata"]["parametros_entrenables"],
                          **metricas.resumen(y[mascara], p[mascara]),
                          "pd_promedio": float(p[mascara].mean())})
    tabla = pd.DataFrame(filas)
    Path("reports").mkdir(exist_ok=True)
    tabla.to_csv("reports/evaluacion_pd.csv", index=False)
    print(tabla[["modelo", "grupo", "n", "tasa_malos", "pd_promedio", "auc", "gini", "ks", "brier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_roc(y, predicciones, FIGURAS / "pd_roc.png")
    figura_calibracion(y, predicciones, FIGURAS / "pd_calibracion.png")
    ultimo = modelos[-1]
    scorecard = Scorecard(**ultimo["metadata"]["config"]["scorecard"])
    figura_ks(y, scorecard.puntaje(predicciones[ultimo["nombre"]]), ultimo["nombre"],
              FIGURAS / "pd_ks.png")
    print(f"\nFiguras en {FIGURAS}/  y métricas en reports/evaluacion_pd.csv")


if __name__ == "__main__":
    main()
