"""Pruebas de la red neuronal. Ejecutar con:  pytest -q"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.red_neuronal import RedNeuronal, verificar_gradientes


def datos_binarios(n=200, d=4, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(size=(n, d))
    logit = X @ rng.normal(size=d) + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return X, y


@pytest.mark.parametrize("activacion", ["relu", "tanh", "sigmoide"])
@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_gradientes_entropia_cruzada(activacion, l2):
    X, y = datos_binarios(n=30)
    red = RedNeuronal(4, capas_ocultas=(5, 3), activacion=activacion, l2=l2, semilla=1)
    assert verificar_gradientes(red, X, y) < 1e-5


def test_gradientes_ecm_salida_sigmoide():
    # Caso de la LGD: objetivo continuo en [0, 1]
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = rng.random(30)
    red = RedNeuronal(4, capas_ocultas=(5,), activacion="tanh", salida="sigmoide",
                      perdida="ecm", l2=0.05, semilla=3)
    assert verificar_gradientes(red, X, y) < 1e-5


def test_sin_capas_ocultas_es_regresion_logistica():
    # Nuestra función objetivo: (1/n) * sum(entropía cruzada) + (lambda/2) * ||w||^2
    # La de scikit-learn:       C * sum(entropía cruzada) + (1/2) * ||w||^2
    # Son proporcionales cuando C = 1 / (n * lambda), así que el mínimo es el mismo.
    X, y = datos_binarios(n=500, d=3)
    lam = 0.01
    red = RedNeuronal(3, capas_ocultas=(), l2=lam, semilla=0)
    red.entrenar(X, y, optimizador="adam", tasa=0.05, epocas=3000, tam_lote=len(X))

    logistica = LogisticRegression(C=1 / (len(X) * lam), tol=1e-10, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(red.W[0][:, 0], logistica.coef_[0], atol=1e-3)
    np.testing.assert_allclose(red.b[0][0, 0], logistica.intercept_[0], atol=1e-3)


def test_aprende_xor():
    # XOR no es separable linealmente: necesita al menos una capa oculta
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    red = RedNeuronal(2, capas_ocultas=(8,), activacion="tanh", semilla=0)
    red.entrenar(X, y, tasa=0.05, epocas=2000, tam_lote=4)
    assert np.array_equal(red.predecir(X) > 0.5, y == 1)


def test_misma_semilla_mismo_resultado():
    X, y = datos_binarios()
    p1 = RedNeuronal(4, (8,), dropout=0.2, semilla=7).entrenar(X, y, epocas=5, tam_lote=32).predecir(X)
    p2 = RedNeuronal(4, (8,), dropout=0.2, semilla=7).entrenar(X, y, epocas=5, tam_lote=32).predecir(X)
    p3 = RedNeuronal(4, (8,), dropout=0.2, semilla=8).entrenar(X, y, epocas=5, tam_lote=32).predecir(X)
    np.testing.assert_array_equal(p1, p2)
    assert not np.allclose(p1, p3)


def test_prediccion_sin_dropout():
    # En modo predicción el dropout no se aplica: dos llamadas dan lo mismo
    X, y = datos_binarios()
    red = RedNeuronal(4, (8,), dropout=0.5, semilla=0).entrenar(X, y, epocas=3)
    np.testing.assert_array_equal(red.predecir(X), red.predecir(X))


def test_parada_temprana_recupera_mejor_epoca():
    X, y = datos_binarios(n=300)
    red = RedNeuronal(4, (64, 64), semilla=0)
    red.entrenar(X[:200], y[:200], X[200:], y[200:], tasa=0.01, epocas=500, tam_lote=32, paciencia=5)
    historia_val = red.historia["perdida_validacion"]
    assert len(historia_val) < 500
    assert red.mejor_epoca == int(np.argmin(historia_val)) + 1
    assert red.perdida(X[200:], y[200:], incluir_penalizacion=False) == pytest.approx(min(historia_val))


def test_contar_parametros():
    red = RedNeuronal(10, capas_ocultas=(5, 3))
    assert red.contar_parametros() == (10 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1)


def test_guardar_y_cargar(tmp_path):
    X, y = datos_binarios()
    red = RedNeuronal(4, (6, 3), activacion="tanh", l2=0.01, semilla=4).entrenar(X, y, epocas=3)
    ruta = tmp_path / "red.npz"
    red.guardar(ruta)
    copia = RedNeuronal.cargar(ruta)
    np.testing.assert_array_equal(red.predecir(X), copia.predecir(X))
    assert copia.config == red.config
