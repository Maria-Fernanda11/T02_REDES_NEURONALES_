"""
Métricas de evaluación.

Parte 1 (clasificación): y = 1 es "malo" (incumple) y p es la probabilidad estimada de ser
malo. Un buen modelo asigna p más altas a los malos que a los buenos.

Parte 2 (regresión): y es la LGD observada e y_hat la predicha.
"""

import numpy as np
import pandas as pd


def _rangos_promedio(valores):
    """Rangos 1, 2, ..., n de los valores; los empates reciben el promedio de sus rangos."""
    orden = np.argsort(valores, kind="mergesort")
    ordenados = valores[orden]
    _, inicio, cuentas = np.unique(ordenados, return_index=True, return_counts=True)
    # un grupo de empates que ocupa las posiciones inicio+1, ..., inicio+cuentas
    # recibe el rango promedio  inicio + (cuentas + 1) / 2
    rangos = np.empty(len(valores))
    rangos[orden] = np.repeat(inicio + (cuentas + 1) / 2, cuentas)
    return rangos


def auc(y, p):
    """Área bajo la curva ROC.

    Es la probabilidad de que un malo escogido al azar tenga mayor p que un bueno escogido
    al azar (los empates cuentan 1/2). Se calcula con la fórmula de Mann-Whitney:
        AUC = (R1 - n1 (n1 + 1) / 2) / (n1 n0)
    donde R1 es la suma de los rangos de los malos, n1 el número de malos y n0 el de buenos.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    R1 = _rangos_promedio(p)[y == 1].sum()
    return float((R1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, p):
    """Índice de Gini (o accuracy ratio): Gini = 2 AUC - 1."""
    return 2 * auc(y, p) - 1


def ks(y, p):
    """Estadístico de Kolmogorov-Smirnov.

    Máxima distancia entre las funciones de distribución acumulada de p en malos y en buenos:
        KS = max_t | F_malos(t) - F_buenos(t) |
    Se evalúa en cada valor distinto de p, después de todos los empates con ese valor.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p, kind="mergesort")
    p_ord, y_ord = p[orden], y[orden]
    acum_malos = np.cumsum(y_ord == 1) / np.sum(y == 1)
    acum_buenos = np.cumsum(y_ord == 0) / np.sum(y == 0)
    # posición del último elemento de cada grupo de valores iguales
    _, inicio, cuentas = np.unique(p_ord, return_index=True, return_counts=True)
    ultimo = inicio + cuentas - 1
    return float(np.max(np.abs(acum_malos[ultimo] - acum_buenos[ultimo])))


def brier(y, p):
    """Puntaje de Brier: error cuadrático medio de la probabilidad, (1/n) sum (p - y)^2."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def entropia_cruzada(y, p, eps=1e-15):
    """Log-loss: -(1/n) sum [ y log p + (1 - y) log(1 - p) ]."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def curva_calibracion(y, p, n_grupos=10):
    """Tabla para la curva de calibración.

    Se ordenan los créditos por p y se parten en n_grupos de igual tamaño (deciles por
    defecto). En cada grupo se compara la p promedio predicha con la tasa de malos observada:
    si el modelo está bien calibrado, los puntos caen sobre la diagonal.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["grupo"] = pd.qcut(df["p"].rank(method="first"), n_grupos, labels=False)
    return (df.groupby("grupo")
              .agg(p_promedio=("p", "mean"), tasa_observada=("y", "mean"), n=("y", "size"))
              .reset_index())


def resumen(y, p):
    """Todas las métricas en un diccionario."""
    return {
        "auc": auc(y, p),
        "gini": gini(y, p),
        "ks": ks(y, p),
        "brier": brier(y, p),
        "entropia_cruzada": entropia_cruzada(y, p),
        "tasa_malos": float(np.mean(y)),
        "n": int(len(y)),
    }


# =============================================================================
# Regresión (Parte 2: LGD)
# =============================================================================

def mae(y, y_hat):
    """Error absoluto medio: (1/n) sum |y - y_hat|."""
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float))))


def rmse(y, y_hat):
    """Raíz del error cuadrático medio: sqrt( (1/n) sum (y - y_hat)^2 )."""
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)) ** 2)))


def ks_dos_muestras(a, b):
    """Distancia de Kolmogorov-Smirnov entre las distribuciones empíricas de a y b:
        D = max_t | F_a(t) - F_b(t) |
    Se usa para comparar la distribución de la LGD predicha con la observada.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    puntos = np.concatenate([a, b])
    F_a = np.searchsorted(a, puntos, side="right") / len(a)
    F_b = np.searchsorted(b, puntos, side="right") / len(b)
    return float(np.max(np.abs(F_a - F_b)))


def comparar_distribuciones(y, y_hat, cuantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    """Tabla con media, desviación estándar y cuantiles de lo observado y lo predicho."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    filas = {"media": (y.mean(), y_hat.mean()), "desviacion": (y.std(), y_hat.std())}
    for q in cuantiles:
        filas[f"q{int(round(q * 100)):02d}"] = (np.quantile(y, q), np.quantile(y_hat, q))
    return pd.DataFrame(filas, index=["observada", "predicha"]).T


def resumen_regresion(y, y_hat):
    """Métricas de regresión en un diccionario."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return {
        "mae": mae(y, y_hat),
        "rmse": rmse(y, y_hat),
        "media_observada": float(y.mean()),
        "media_predicha": float(y_hat.mean()),
        "ks_distribuciones": ks_dos_muestras(y, y_hat),
        "n": int(len(y)),
    }
