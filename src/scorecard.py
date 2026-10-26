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
