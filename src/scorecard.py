"""
Transformación de probabilidad de incumplimiento a puntaje (scorecard).

Se define el puntaje como una función lineal del logaritmo de las odds buenos:malos:
    odds = (1 - p) / p
    puntaje = offset + factor * ln(odds)

y se fijan dos condiciones:
    1. con odds = odds_base el puntaje vale puntaje_base:
           puntaje_base = offset + factor * ln(odds_base)
    2. cada vez que las odds se duplican el puntaje sube PDO puntos:
           puntaje_base + PDO = offset + factor * ln(2 * odds_base)

Restando (2) - (1):  PDO = factor * ln(2)   =>   factor = PDO / ln(2)
y de (1):            offset = puntaje_base - factor * ln(odds_base)

Puntaje alto = menor riesgo. Como ln(odds) = ln((1 - p) / p) = -logit(p), y la red estima
p = sigmoide(z), el puntaje es lineal en la salida antes de la sigmoide:
    puntaje = offset - factor * z
"""

import numpy as np
import pandas as pd


class Scorecard:
    def __init__(self, puntaje_base=600, odds_base=50, pdo=20):
        self.puntaje_base = puntaje_base
        self.odds_base = odds_base
        self.pdo = pdo
        self.factor = pdo / np.log(2)
        self.offset = puntaje_base - self.factor * np.log(odds_base)

    def puntaje(self, p, eps=1e-12):
        """Puntaje a partir de la probabilidad de ser malo."""
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        return self.offset + self.factor * np.log((1 - p) / p)

    def probabilidad(self, puntaje):
        """Transformación inversa: probabilidad de ser malo a partir del puntaje."""
        odds = np.exp((np.asarray(puntaje, dtype=float) - self.offset) / self.factor)
        return 1 / (1 + odds)

    def parametros(self):
        return {
            "puntaje_base": self.puntaje_base,
            "odds_base": self.odds_base,
            "pdo": self.pdo,
            "factor": float(self.factor),
            "offset": float(self.offset),
        }


def percentil(puntaje, puntajes_poblacion):
    """Porcentaje de la población con un puntaje menor o igual al dado (0 a 100)."""
    poblacion = np.sort(np.asarray(puntajes_poblacion, dtype=float))
    return 100.0 * np.searchsorted(poblacion, puntaje, side="right") / len(poblacion)


def puntos_por_variable(red, scorecard, X_transformada, referencia, grupos):
    """Cuántos puntos aporta cada variable al puntaje de cada crédito.

    Los puntos de una variable son lo que cambia el puntaje si a esa variable (todas sus
    columnas transformadas) se le pone el valor del solicitante de referencia, que es el
    promedio de train en el espacio transformado:
        puntos_j(x) = puntaje(x) - puntaje(x con el grupo j en la referencia)
    Positivo = la variable sube el puntaje (menos riesgo) frente al solicitante promedio.

    Regresión logística: logit(p) = b0 + sum_j w_j x_j, y el puntaje es lineal en el logit
    (puntaje = offset - factor * logit), así que
        puntos_j(x) = -factor * w_j (x_j - x_ref_j)
        puntaje(x)  = puntaje(x_ref) + sum_j puntos_j(x)      exactamente.
    Esa es la scorecard aditiva clásica, con variables continuas.

    Red neuronal: el logit no es una suma de efectos por variable, así que los puntos no suman
    exactamente el total. La diferencia (interaccion) mide cuánto del puntaje depende de
    combinaciones de variables.

    Devuelve (base, puntos, interaccion): base = puntaje del solicitante de referencia,
    puntos = DataFrame (créditos x variables), interaccion = total - base - suma de puntos.
    """
    X = np.atleast_2d(np.asarray(X_transformada, dtype=float))
    referencia = np.asarray(referencia, dtype=float)
    total = scorecard.puntaje(red.predecir(X))
    base = float(scorecard.puntaje(red.predecir(referencia[None, :]))[0])
    puntos = {}
    for variable, columnas in grupos.items():
        X_ref = X.copy()
        X_ref[:, columnas] = referencia[columnas]
        puntos[variable] = total - scorecard.puntaje(red.predecir(X_ref))
    puntos = pd.DataFrame(puntos)
    return base, puntos, total - base - puntos.sum(axis=1).to_numpy()
