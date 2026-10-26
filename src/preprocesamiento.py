"""
Preprocesamiento que se ajusta con el conjunto de entrenamiento y se guarda junto al modelo.

La limpieza (filtrar filas, construir y, asignar la partición, corregir tipos) se hace antes,
en el script de limpieza de datos. Aquí solo va lo que depende de estadísticos de
entrenamiento (medianas, medias, desviaciones, categorías), porque eso debe calcularse
únicamente con train y aplicarse igual en validación, prueba y en la aplicación web.

    numéricas:   imputar mediana (+ indicador de faltante)  ->  estandarizar
    logarítmicas: imputar mediana (+ indicador)  ->  log(1 + x)  ->  estandarizar
    categóricas: imputar la categoría más frecuente  ->  one-hot
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def construir_preprocesador(numericas, categoricas, logaritmicas=()):
    # add_indicator=True agrega una columna 0/1 por cada variable que tuvo faltantes en
    # train. Para mths_since_last_delinq, por ejemplo, el faltante significa "nunca ha estado
    # en mora", así que el indicador conserva esa información en lugar de perderla al imputar.
    numericas = [c for c in numericas if c not in logaritmicas]
    transformadores = []
    if numericas:
        transformadores.append(("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
        ]), list(numericas)))
    if logaritmicas:
        transformadores.append(("log", Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escalar", StandardScaler()),
        ]), list(logaritmicas)))
    if categoricas:
        transformadores.append(("cat", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categoricas)))
    return ColumnTransformer(transformadores, verbose_feature_names_out=False)
