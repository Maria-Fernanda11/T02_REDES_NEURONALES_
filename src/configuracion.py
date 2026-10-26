"""
Lectura de config.ini.

Las secciones [pd.xxx] y [lgd.xxx] se convierten en un diccionario por parte. Por ejemplo,
para la parte "pd" se obtiene:
    {"semilla": 42, "datos": {...}, "red": {...}, "entrenamiento": {...},
     "scorecard": {...}, "salida": {...}}
"""

import configparser
import re

# Claves cuyo valor siempre es una lista (aunque tenga un solo elemento o ninguno)
CLAVES_LISTA = {"capas_ocultas", "numericas", "logaritmicas", "categoricas", "recortar_objetivo"}


def _convertir_escalar(texto):
    t = texto.strip()
    if t.lower() in ("si", "sí", "true"):
        return True
    if t.lower() in ("no", "false"):
        return False
    for tipo in (int, float):
        try:
            return tipo(t)
        except ValueError:
            pass
    return t


def convertir_valor(clave, texto):
    """Convierte el texto del .ini al tipo de Python que corresponde."""
    if clave in CLAVES_LISTA:
        partes = re.split(r"[,\n]", texto.strip().strip("[]"))
        return [_convertir_escalar(p) for p in partes if p.strip()]
    return _convertir_escalar(texto)


def cargar_config(ruta, parte, cambios=()):
    """Lee la configuración de una parte ("pd" o "lgd") y aplica cambios de la forma
    seccion.clave=valor, por ejemplo "red.dropout=0.3" o "semilla=7"."""
    lector = configparser.ConfigParser(inline_comment_prefixes=("#",), interpolation=None)
    if not lector.read(ruta, encoding="utf-8"):
        raise FileNotFoundError(f"No se encontró el archivo de configuración {ruta}")

    config = {k: convertir_valor(k, v) for k, v in lector["general"].items()}
    prefijo = f"{parte}."
    for seccion in lector.sections():
        if seccion.startswith(prefijo):
            config[seccion[len(prefijo):]] = {
                k: convertir_valor(k, v) for k, v in lector[seccion].items()
            }

    for cambio in cambios:
        clave, valor = cambio.split("=", 1)
        if "." in clave:
            seccion, nombre = clave.split(".", 1)
            if seccion not in config:
                raise KeyError(f"La sección [{parte}.{seccion}] no existe en {ruta}.")
            nodo = config[seccion]
        else:
            nodo, nombre = config, clave
        if nombre not in nodo:
            raise KeyError(f"La clave '{clave}' no existe en la configuración de '{parte}'.")
        nodo[nombre] = convertir_valor(nombre, valor)
    return config


def imprimir_config(config, secciones=("red", "entrenamiento", "scorecard")):
    """Muestra los hiperparámetros con que se va a entrenar."""
    print(f"semilla = {config['semilla']}")
    for seccion in secciones:
        if seccion in config:
            print(f"[{seccion}]")
            for clave, valor in config[seccion].items():
                print(f"    {clave} = {valor}")
    print()
