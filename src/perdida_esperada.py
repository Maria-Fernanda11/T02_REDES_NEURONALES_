"""Pérdida esperada sobre capital original: PD * LGD_proxy * monto financiado."""
import numpy as np


def calcular_perdida_esperada(pd, lgd, ead):
    """Devuelve un vector monetario; rechaza desalineación y rangos inválidos.

    EAD debe ser una exposición determinista por crédito en la misma moneda.
    La igualdad producto no supone independencia PD/LGD: LGD es condicional
    al incumplimiento. Con EAD aleatoria se necesita modelar su relación con LGD.
    """
    pd, lgd, ead = (np.asarray(v, dtype=float) for v in (pd, lgd, ead))
    if pd.ndim != 1 or not len(pd) or pd.shape != lgd.shape or pd.shape != ead.shape:
        raise ValueError('PD, LGD y EAD deben ser vectores alineados no vacíos')
    if not all(np.isfinite(v).all() for v in (pd, lgd, ead)):
        raise ValueError('No se admiten valores ausentes o no finitos')
    if ((pd < 0) | (pd > 1)).any() or ((lgd < 0) | (lgd > 1)).any() or (ead < 0).any():
        raise ValueError('PD y LGD deben estar en [0,1]; EAD debe ser no negativa')
    return pd * lgd * ead
