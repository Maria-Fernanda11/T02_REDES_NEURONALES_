"""
Perceptrón multicapa (MLP) implementado desde cero con NumPy.

Notación (la misma del reporte):
    X        matriz de entrada de tamaño (n, d): una fila por crédito, una columna por variable.
    L        número de capas con pesos (capas ocultas + capa de salida).
    W[l]     matriz de pesos de la capa l, tamaño (neuronas_entrada, neuronas_salida).
    b[l]     vector de sesgos de la capa l, tamaño (1, neuronas_salida).
    Z[l]     preactivación:  Z[l] = A[l-1] @ W[l] + b[l]
    A[l]     activación:     A[l] = g_l(Z[l]),  con A[0] = X
    A[L]     salida de la red (para la PD, una probabilidad entre 0 y 1).

Cada neurona hace exactamente lo mismo: una combinación lineal de sus entradas más un sesgo,
seguida de una función de activación. Una capa es simplemente muchas neuronas en paralelo,
por eso se escribe como un producto de matrices.
"""

import json

import numpy as np


# =============================================================================
# 1. Funciones de activación y sus derivadas
# =============================================================================
# La retropropagación necesita g'(z) para cada activación. Las derivadas se
# expresan en función de z para que cada pareja (g, g') sea autocontenida.

def relu(z):
    # g(z) = max(0, z)
    return np.maximum(0.0, z)


def relu_derivada(z):
    # g'(z) = 1 si z > 0, 0 si z <= 0 (en z = 0 se toma 0 por convención)
    return (z > 0).astype(z.dtype)


def tanh(z):
    # g(z) = (e^z - e^-z) / (e^z + e^-z), con valores en (-1, 1)
    return np.tanh(z)


def tanh_derivada(z):
    # g'(z) = 1 - tanh(z)^2
    return 1.0 - np.tanh(z) ** 2


def sigmoide(z):
    # g(z) = 1 / (1 + e^-z), con valores en (0, 1).
    # Se calcula por partes para que e^-z no se desborde cuando z es muy negativo.
    salida = np.empty_like(z)
    positivos = z >= 0
    salida[positivos] = 1.0 / (1.0 + np.exp(-z[positivos]))
    ez = np.exp(z[~positivos])
    salida[~positivos] = ez / (1.0 + ez)
    return salida


def sigmoide_derivada(z):
    # g'(z) = g(z) * (1 - g(z))
    s = sigmoide(z)
    return s * (1.0 - s)


def identidad(z):
    return z


def identidad_derivada(z):
    return np.ones_like(z)


ACTIVACIONES = {
    "relu": (relu, relu_derivada),
    "tanh": (tanh, tanh_derivada),
    "sigmoide": (sigmoide, sigmoide_derivada),
    "identidad": (identidad, identidad_derivada),
}


# =============================================================================
# 2. Optimizadores
# =============================================================================
# Un optimizador recibe la lista de parámetros [W1, ..., WL, b1, ..., bL] y la
# lista de sus gradientes en el mismo orden, y actualiza los parámetros en sitio.

class SGD:
    """Descenso de gradiente estocástico, opcionalmente con momento.

        v <- momento * v - tasa * gradiente
        theta <- theta + v

    Con momento = 0 se reduce a  theta <- theta - tasa * gradiente.
    """

    def __init__(self, tasa=0.01, momento=0.0):
        self.tasa = tasa
        self.momento = momento
        self.velocidades = None

    def paso(self, parametros, gradientes):
        if self.velocidades is None:
            self.velocidades = [np.zeros_like(p) for p in parametros]
        for p, g, v in zip(parametros, gradientes, self.velocidades):
            v *= self.momento
            v -= self.tasa * g
            p += v


class Adam:
    """Adam (Kingma y Ba, 2015).

        m <- beta1 * m + (1 - beta1) * g          promedio móvil del gradiente
        v <- beta2 * v + (1 - beta2) * g^2        promedio móvil del gradiente al cuadrado
        m_hat = m / (1 - beta1^t)                 corrección de sesgo (m y v arrancan en 0)
        v_hat = v / (1 - beta2^t)
        theta <- theta - tasa * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, tasa=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.tasa = tasa
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def paso(self, parametros, gradientes):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in parametros]
            self.v = [np.zeros_like(p) for p in parametros]
        self.t += 1
        for p, g, m, v in zip(parametros, gradientes, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.tasa * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZADORES = {"sgd": SGD, "adam": Adam}


# =============================================================================
# 3. La red neuronal
# =============================================================================

class RedNeuronal:
    """Perceptrón multicapa con una neurona de salida.

    Parámetros
    ----------
    n_entradas : número de variables de entrada (columnas de X).
    capas_ocultas : neuronas por capa oculta, por ejemplo (32, 16). () da un modelo sin
        capas ocultas: con salida sigmoide y entropía cruzada es una regresión logística.
    activacion : activación de las capas ocultas ("relu", "tanh" o "sigmoide").
    salida : activación de la capa de salida ("sigmoide" para probabilidades o
        fracciones en [0, 1], "identidad" para valores reales).
    perdida : "entropia_cruzada" (verosimilitud Bernoulli) o "ecm" (error cuadrático medio,
        verosimilitud gaussiana).
    l2 : coeficiente lambda de la penalización (lambda / 2) * suma de W^2. No penaliza sesgos.
    dropout : probabilidad de apagar cada neurona oculta durante el entrenamiento.
    semilla : fija la inicialización, el orden de los mini-lotes y las máscaras de dropout.
    """

    def __init__(self, n_entradas, capas_ocultas=(32, 16), activacion="relu",
                 salida="sigmoide", perdida="entropia_cruzada", l2=0.0, dropout=0.0,
                 semilla=0):
        if perdida == "entropia_cruzada" and salida != "sigmoide":
            raise ValueError("La entropía cruzada necesita salida sigmoide (probabilidad en (0, 1)).")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1).")

        self.config = {
            "n_entradas": int(n_entradas),
            "capas_ocultas": [int(k) for k in capas_ocultas],
            "activacion": activacion,
            "salida": salida,
            "perdida": perdida,
            "l2": float(l2),
            "dropout": float(dropout),
            "semilla": int(semilla),
        }
        self.g, self.g_derivada = ACTIVACIONES[activacion]
        self.g_salida, self.g_salida_derivada = ACTIVACIONES[salida]
        self.perdida_nombre = perdida
        self.l2 = l2
        self.dropout = dropout
        self.rng = np.random.default_rng(semilla)

        # Inicialización de pesos: varianza escalada según el número de entradas para que
        # la señal no explote ni se apague al atravesar las capas.
        #   He (He et al., 2015) para ReLU:            Var(W) = 2 / n_entrada
        #   Glorot (Glorot y Bengio, 2010) en otro caso: Var(W) = 2 / (n_entrada + n_salida)
        # Los sesgos arrancan en cero.
        tamanos = [n_entradas, *capas_ocultas, 1]
        self.W, self.b = [], []
        for l, (n_in, n_out) in enumerate(zip(tamanos[:-1], tamanos[1:])):
            es_oculta = l < len(tamanos) - 2
            if es_oculta and activacion == "relu":
                desviacion = np.sqrt(2.0 / n_in)
            else:
                desviacion = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(self.rng.normal(0.0, desviacion, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

        self.historia = None
        self.mejor_epoca = None

    @property
    def n_capas(self):
        return len(self.W)

    def contar_parametros(self):
        """Número de parámetros entrenables (pesos + sesgos)."""
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # -------------------------------------------------------------------------
    # Propagación hacia adelante
    # -------------------------------------------------------------------------
    def _propagar(self, X, entrenando):
        """Calcula la salida de la red y guarda lo que la retropropagación necesita.

        Devuelve una lista con una tupla (A_previa, Z, mascara) por capa:
            A_previa  entrada que recibió la capa (ya con dropout aplicado)
            Z         preactivación de la capa
            mascara   máscara de dropout aplicada a la salida de la capa (None si no hubo)
        y la activación final A[L].
        """
        cache = []
        A = X
        for l in range(self.n_capas):
            Z = A @ self.W[l] + self.b[l]
            es_salida = l == self.n_capas - 1
            if es_salida:
                A_nueva = self.g_salida(Z)
                mascara = None
            else:
                A_nueva = self.g(Z)
                mascara = None
                if entrenando and self.dropout > 0:
                    # Dropout invertido (Srivastava et al., 2014): cada neurona se apaga con
                    # probabilidad p y las que sobreviven se multiplican por 1 / (1 - p).
                    # Así el valor esperado de la activación no cambia y en predicción
                    # se usa la red completa sin reescalar nada.
                    mantener = self.rng.random(A_nueva.shape) >= self.dropout
                    mascara = mantener / (1.0 - self.dropout)
                    A_nueva = A_nueva * mascara
            cache.append((A, Z, mascara))
            A = A_nueva
        return cache, A

    def predecir(self, X):
        """Salida de la red para cada fila de X (para la PD: probabilidad de ser malo)."""
        _, A = self._propagar(np.asarray(X, dtype=float), entrenando=False)
        return A[:, 0]

    # -------------------------------------------------------------------------
    # Función de pérdida
    # -------------------------------------------------------------------------
    def _perdida_datos(self, Z_salida, A_salida, y):
        """Pérdida promedio sobre las observaciones, sin penalización.

        Entropía cruzada (menos log-verosimilitud Bernoulli, promediada):
            J = -(1/n) * sum[ y log(p) + (1 - y) log(1 - p) ],   p = sigmoide(z)
        Se calcula a partir de z con la identidad equivalente
            J = (1/n) * sum[ log(1 + e^z) - y z ]
        que no produce log(0) aunque p se acerque a 0 o a 1.

        ECM (menos log-verosimilitud gaussiana con varianza fija, salvo constantes):
            J = (1/n) * sum (y - y_hat)^2
        """
        if self.perdida_nombre == "entropia_cruzada":
            return float(np.mean(np.logaddexp(0.0, Z_salida) - y * Z_salida))
        return float(np.mean((A_salida - y) ** 2))

    def _penalizacion(self):
        # (lambda / 2) * suma de los cuadrados de todos los pesos (no de los sesgos)
        return 0.5 * self.l2 * sum(float(np.sum(W ** 2)) for W in self.W)

    def perdida(self, X, y, incluir_penalizacion=True):
        """Valor de la función objetivo en (X, y), con la red en modo predicción."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        cache, A = self._propagar(X, entrenando=False)
        J = self._perdida_datos(cache[-1][1], A, y)
        if incluir_penalizacion:
            J += self._penalizacion()
        return J

    # -------------------------------------------------------------------------
    # Retropropagación
    # -------------------------------------------------------------------------
    def _retropropagar(self, cache, A_salida, y):
        """Gradientes de la función objetivo respecto a cada W[l] y b[l].

        Se define delta[l] = dJ/dZ[l]. Por regla de la cadena:
            capa de salida:
                entropía cruzada + sigmoide:  delta[L] = (A[L] - y) / n
                    (la derivada de la sigmoide se cancela con la del logaritmo)
                ECM:  delta[L] = (2 / n) * (A[L] - y) * g_salida'(Z[L])
            gradientes de la capa l:
                dJ/dW[l] = A[l-1]^T @ delta[l] + lambda * W[l]
                dJ/db[l] = suma de las filas de delta[l]
            paso a la capa anterior:
                dJ/dA[l-1] = delta[l] @ W[l]^T
                (si hubo dropout en A[l-1], se multiplica por la misma máscara)
                delta[l-1] = dJ/dA[l-1] * g'(Z[l-1])      (producto elemento a elemento)
        """
        n = y.shape[0]
        _, Z_salida, _ = cache[-1]
        if self.perdida_nombre == "entropia_cruzada":
            delta = (A_salida - y) / n
        else:
            delta = (2.0 / n) * (A_salida - y) * self.g_salida_derivada(Z_salida)

        grad_W = [None] * self.n_capas
        grad_b = [None] * self.n_capas
        for l in reversed(range(self.n_capas)):
            A_previa, _, _ = cache[l]
            grad_W[l] = A_previa.T @ delta + self.l2 * self.W[l]
            grad_b[l] = delta.sum(axis=0, keepdims=True)
            if l > 0:
                dA_previa = delta @ self.W[l].T
                _, Z_previa, mascara_previa = cache[l - 1]
                if mascara_previa is not None:
                    dA_previa = dA_previa * mascara_previa
                delta = dA_previa * self.g_derivada(Z_previa)
        return grad_W, grad_b

    # -------------------------------------------------------------------------
    # Entrenamiento
    # -------------------------------------------------------------------------
    def entrenar(self, X, y, X_val=None, y_val=None, optimizador="adam", tasa=0.001,
                 epocas=100, tam_lote=256, paciencia=10, verbose=False):
        """Entrena con mini-lotes y, si hay datos de validación, parada temprana.

        En cada época se barajan las filas, se recorren en lotes de tam_lote y en cada lote
        se hace propagación -> retropropagación -> paso del optimizador.

        Parada temprana: al final de cada época se mide la pérdida (sin penalización) en
        validación. Si no mejora durante `paciencia` épocas seguidas se detiene el
        entrenamiento y se recuperan los pesos de la mejor época.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        hay_validacion = X_val is not None
        if hay_validacion:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)

        if isinstance(optimizador, str):
            optimizador = OPTIMIZADORES[optimizador](tasa=tasa)

        self.historia = {"perdida_entrenamiento": [], "perdida_validacion": []}
        mejor_perdida = np.inf
        mejores_pesos = None
        epocas_sin_mejora = 0
        n = X.shape[0]

        for epoca in range(epocas):
            orden = self.rng.permutation(n)
            for inicio in range(0, n, tam_lote):
                idx = orden[inicio:inicio + tam_lote]
                cache, A = self._propagar(X[idx], entrenando=True)
                grad_W, grad_b = self._retropropagar(cache, A, y[idx])
                optimizador.paso(self.W + self.b, grad_W + grad_b)

            perdida_ent = self.perdida(X, y, incluir_penalizacion=False)
            self.historia["perdida_entrenamiento"].append(perdida_ent)
            mensaje = f"época {epoca + 1:4d}  pérdida entrenamiento {perdida_ent:.5f}"

            if hay_validacion:
                perdida_val = self.perdida(X_val, y_val, incluir_penalizacion=False)
                self.historia["perdida_validacion"].append(perdida_val)
                mensaje += f"  validación {perdida_val:.5f}"
                if perdida_val < mejor_perdida:
                    mejor_perdida = perdida_val
                    mejores_pesos = ([W.copy() for W in self.W], [b.copy() for b in self.b])
                    self.mejor_epoca = epoca + 1
                    epocas_sin_mejora = 0
                else:
                    epocas_sin_mejora += 1

            if verbose:
                print(mensaje)
            if hay_validacion and epocas_sin_mejora >= paciencia:
                if verbose:
                    print(f"Parada temprana: mejor época {self.mejor_epoca}")
                break

        if mejores_pesos is not None:
            self.W, self.b = mejores_pesos
        else:
            self.mejor_epoca = len(self.historia["perdida_entrenamiento"])
        return self

    # -------------------------------------------------------------------------
    # Guardar y cargar
    # -------------------------------------------------------------------------
    def guardar(self, ruta):
        """Guarda la configuración y los pesos en un único archivo .npz."""
        arreglos = {f"W{l}": W for l, W in enumerate(self.W)}
        arreglos.update({f"b{l}": b for l, b in enumerate(self.b)})
        np.savez(ruta, config=json.dumps(self.config), **arreglos)

    @classmethod
    def cargar(cls, ruta):
        with np.load(ruta) as datos:
            config = json.loads(str(datos["config"]))
            red = cls(**config)
            red.W = [datos[f"W{l}"] for l in range(red.n_capas)]
            red.b = [datos[f"b{l}"] for l in range(red.n_capas)]
        return red


# =============================================================================
# 4. Verificación de gradientes
# =============================================================================

def verificar_gradientes(red, X, y, eps=1e-6):
    """Compara los gradientes de la retropropagación con diferencias finitas centradas:

        dJ/dtheta  ~  [J(theta + eps) - J(theta - eps)] / (2 eps)

    Devuelve el error relativo máximo |analítico - numérico| / max(|analítico|, |numérico|, 1e-12).
    Con float64 debería ser del orden de 1e-7 o menor. La red debe tener dropout = 0, porque
    con dropout la función objetivo cambia en cada evaluación.
    """
    if red.dropout > 0:
        raise ValueError("Verificar gradientes con dropout = 0.")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    cache, A = red._propagar(X, entrenando=False)
    grad_W, grad_b = red._retropropagar(cache, A, y)

    error_maximo = 0.0
    for parametro, gradiente in zip(red.W + red.b, grad_W + grad_b):
        for i in np.ndindex(parametro.shape):
            original = parametro[i]
            parametro[i] = original + eps
            J_mas = red.perdida(X, y)
            parametro[i] = original - eps
            J_menos = red.perdida(X, y)
            parametro[i] = original
            numerico = (J_mas - J_menos) / (2 * eps)
            analitico = gradiente[i]
            error = abs(analitico - numerico) / max(abs(analitico), abs(numerico), 1e-12)
            error_maximo = max(error_maximo, error)
    return error_maximo
