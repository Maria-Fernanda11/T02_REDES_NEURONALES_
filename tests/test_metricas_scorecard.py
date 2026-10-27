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


def test_puntos_por_variable_suman_el_puntaje_en_la_logistica():
    from src.red_neuronal import RedNeuronal
    from src.scorecard import puntos_por_variable

    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 5))
    grupos = {"a": [0], "b": [1, 2], "c": [3, 4]}
    sc = Scorecard(600, 50, 20)

    logistica = RedNeuronal(5, capas_ocultas=(), semilla=1)
    base, puntos, interaccion = puntos_por_variable(logistica, sc, X, X.mean(axis=0), grupos)
    total = sc.puntaje(logistica.predecir(X))
    assert np.allclose(base + puntos.sum(axis=1) + interaccion, total)
    assert np.abs(interaccion).max() < 1e-8
    # puntos_j = -factor * w_j (x_j - x_ref_j)
    w = logistica.W[0].ravel()
    assert np.allclose(puntos["a"], -sc.factor * w[0] * (X[:, 0] - X[:, 0].mean()))

    red = RedNeuronal(5, capas_ocultas=(8,), activacion="tanh", semilla=1)
    red.W = [W * 3 for W in red.W]           # pesos grandes: interacciones visibles
    _, _, interaccion = puntos_por_variable(red, sc, X, X.mean(axis=0), grupos)
    assert np.abs(interaccion).max() > 1e-3


def test_grupos_cubren_cada_columna_una_vez():
    import pandas as pd
    from src.preprocesamiento import PreprocesadorCredito

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"emp_length": rng.uniform(0, 10, 200), "dti": rng.uniform(0, 30, 200),
                       "emp_length_na": rng.integers(0, 2, 200),
                       "purpose": rng.choice(["a", "b", "c"], 200)})
    pre = PreprocesadorCredito(["emp_length", "dti"], ["purpose"], indicadoras=["emp_length_na"]).fit(df)
    grupos = pre.grupos()
    assert grupos["emp_length"] == [0, 2]                  # la indicadora va con su variable
    assert sorted(i for cols in grupos.values() for i in cols) == list(range(len(pre.get_feature_names_out())))
