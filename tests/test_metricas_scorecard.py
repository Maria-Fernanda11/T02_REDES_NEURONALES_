"""Pruebas de las métricas (contra scikit-learn) y de la scorecard."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src import metricas
from src.scorecard import Scorecard, percentil


@pytest.fixture
def y_p():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.15, size=2000), 0.001, 0.999)
    p = np.round(p, 2)  # fuerza empates, que es donde suelen fallar las implementaciones
    return y, p


def test_auc_igual_a_sklearn(y_p):
    y, p = y_p
    assert metricas.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_ks_igual_a_max_tpr_menos_fpr(y_p):
    y, p = y_p
    fpr, tpr, _ = roc_curve(y, p)
    assert metricas.ks(y, p) == pytest.approx(np.max(tpr - fpr), abs=1e-12)


def test_brier_y_entropia_igual_a_sklearn(y_p):
    y, p = y_p
    assert metricas.brier(y, p) == pytest.approx(brier_score_loss(y, p))
    assert metricas.entropia_cruzada(y, p) == pytest.approx(log_loss(y, p))


def test_curva_calibracion(y_p):
    y, p = y_p
    tabla = metricas.curva_calibracion(y, p, n_grupos=10)
    assert len(tabla) == 10
    assert tabla["n"].sum() == len(y)
    assert tabla["p_promedio"].is_monotonic_increasing


def test_scorecard_condiciones_que_la_definen():
    sc = Scorecard(puntaje_base=600, odds_base=50, pdo=20)
    p_base = 1 / (1 + 50)          # odds buenos:malos = 50  =>  p = 1/51
    p_doble = 1 / (1 + 100)        # odds duplicadas
    assert sc.puntaje(p_base) == pytest.approx(600)
    assert sc.puntaje(p_doble) == pytest.approx(620)


def test_scorecard_es_invertible_y_monotona():
    sc = Scorecard()
    p = np.linspace(0.01, 0.99, 50)
    puntajes = sc.puntaje(p)
    assert np.all(np.diff(puntajes) < 0)          # más riesgo => menos puntaje
    np.testing.assert_allclose(sc.probabilidad(puntajes), p)


def test_percentil():
    poblacion = np.arange(1, 101)                 # puntajes 1..100
    assert percentil(50, poblacion) == pytest.approx(50)
    assert percentil(0, poblacion) == 0
    assert percentil(1000, poblacion) == 100
