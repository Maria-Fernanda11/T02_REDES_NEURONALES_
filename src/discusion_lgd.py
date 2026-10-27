"""Redacción trazable: cada cifra procede de los artefactos de la corrida."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def redactar(carpeta):
    carpeta=Path(carpeta)
    m=json.loads((carpeta/'experimento.json').read_text())
    ols=pd.read_csv(carpeta/'referencia.csv').iloc[0]
    redes=pd.read_csv(carpeta/'corridas.csv')
    with np.load(carpeta/'predicciones_comparadas.npz') as d:
        y=d['y_test']; p=d[f"pred_semilla_{m['semillas'][0]}"]
        po=d['pred_lineal']
    estado='Demostración preliminar' if m['demo'] else 'Experimento completo'
    texto=f'# Discusión de resultados — {estado}\n\n'
    for metrica in ['mae','rmse']:
        media=redes[metrica].mean(); base=ols[metrica]
        mejora=100*(base-media)/base if base else float('nan')
        ganan=int((redes[metrica]<base).sum())
        texto+=(f'**{metrica.upper()}.** OLS obtuvo {base:.6f}; la red obtuvo '
                f'{media:.6f} ± {redes[metrica].std(ddof=1):.6f} entre semillas, '
                f'con mejor {redes[metrica].min():.6f} y peor {redes[metrica].max():.6f}. '
                f'La mejora relativa media frente a OLS es {mejora:.2f}% (negativa significa deterioro). '
                f'{ganan} de {len(redes)} corridas tuvieron menor error que OLS. '
                'Esta frecuencia describe variación por inicialización, no es un contraste de significancia ni incertidumbre por muestreo.\n\n')
    texto+=(f'**Distribución y residuos.** En la primera semilla predeclarada, la media observada es '
            f'{y.mean():.6f} y la predicha {p.mean():.6f}; el sesgo medio predicción menos observación es '
            f'{(p-y).mean():.6f}. La desviación observada es {y.std():.6f} y la predicha {p.std():.6f}. '
            f'Los percentiles 5/95 observados son {np.quantile(y,.05):.6f}/{np.quantile(y,.95):.6f}, '
            f'frente a {np.quantile(p,.05):.6f}/{np.quantile(p,.95):.6f} predichos. '
            'Estos valores y la Figura 1 permiten valorar compresión de colas; una media similar no demuestra igualdad de distribuciones. '
            'MSE estima una media condicional, cuya dispersión puede ser menor que la de los resultados individuales. '
            f'El residuo medio observado menos predicho es {(y-p).mean():.6f}; su desviación es {(y-p).std():.6f}. '
            f'OLS genera {int(((po<0)|(po>1)).sum())} predicciones fuera de [0,1]. '
            'La Figura 2 muestra residuos y aprendizaje; estas cifras por sí solas no prueban homocedasticidad ni independencia.\n\n')
    texto+=(f'**Complejidad y costo.** OLS tiene {int(ols.parametros)} parámetros nominales y tarda '
            f'{m["segundos_ols"]:.3f} s en ajustar. La red tiene {int(redes.parametros.iloc[0])} parámetros. '
            f'Se probaron {m["configuraciones_red"]} configuraciones neuronales frente a una OLS; '
            f'la búsqueda tardó {m["segundos_busqueda"]:.3f} s y las {len(redes)} réplicas '
            f'{m["segundos_replicas"]:.3f} s. El flujo total tardó {m["segundos_total"]:.3f} s. '
            'Los presupuestos son distintos y se declaran; OLS ofrece una interpretación lineal directa, '
            'mientras que la red puede capturar relaciones no lineales con mayor complejidad. No se atribuye '
            'una ventaja práctica general sin valorar conjuntamente errores, variabilidad y costo.\n\n')
    texto+=('**Alcance económico.** La LGD es la proxy 1 − total_rec_prncp/funded_amnt, sin '
            'recuperaciones netas ni descuento explícitos. Se conserva Charged Off/Default; la severidad '
            'para los otros estados de la clase 1 de PD no ha sido validada. Las recuperaciones pueden '
            'estar censuradas. Las particiones son aleatorias, no una prueba temporal. La aplicación '
            'al originar exige solo variables disponibles entonces y horizontes coherentes. '
            'La pérdida esperada con funded_amnt es una proxy sobre capital original, no EAD histórica exacta.\n\n')
    if m['demo']:
        texto+='**Estado:** resultados preliminares; ejecutar modo completo con al menos 30 semillas antes de presentarlos como experimento final.\n'
    else:
        texto+=f'**Semillas:** se ejecutaron {len(redes)} semillas distintas de la configuración seleccionada por validación. Esto no elimina las limitaciones de población y horizonte.\n'
    (carpeta/'discusion_resultados.md').write_text(texto,encoding='utf-8')
    return texto
