# Punto 2: red neuronal de Manuel

Se utiliza **`src/red_neuronal.py`**, la implementación de Manuel, a través de `entrenar_lgd.py`. No se usa la implementación anterior `red_neuronal/` ni sus modelos sintéticos. La red de Manuel usa NumPy y el preprocesamiento usa pandas/scikit-learn; se acepta este enfoque para esta integración.

Desde la raíz, instale `requirements.txt` en Python 3.12. Prepare el CSV y entrene:

```bash
python preparar_lgd.py --csv datos/loan/loan.csv
python entrenar_lgd.py
```

La preparación convierte plazo, años de empleo y antigüedad crediticia a valores numéricos y genera `data/processed/lgd_limpio.parquet` con sus metadatos. La imputación, logaritmos, escalado y categorías se ajustan **solo en train** mediante el preprocesador de Manuel. No cambie ese orden al predecir.

Se conserva la **LGD proxy del notebook de regresión**: `1 - total_rec_prncp/funded_amnt`. No incorpora explícitamente recuperaciones netas, no es LGD regulatoria exacta y usa monto original, no EAD histórica. La muestra se limita a `Charged Off` y `Default`; esto no redefine la etiqueta de PD del enunciado. La representatividad para los otros estados de la clase 1 requiere análisis adicional. El script rechaza denominadores inválidos y objetivos fuera de [0,1], sin recortarlos. No se pasan campos ex post a la red.

Las particiones son aleatorias 60% entrenamiento, 20% validación, 20% prueba, semilla 42. No equivalen a una evaluación temporal. El notebook lineal anterior usa 80/20: para comparar justamente, reentrene una referencia sobre **este mismo parquet y preprocesamiento**, por ejemplo:

```bash
python entrenar_lgd.py --set red.capas_ocultas= --set red.salida=identidad --set red.dropout=0 --set red.l2=0 --set salida.carpeta=models/lgd_lineal
```

Esta referencia se ajusta por optimización iterativa; compruebe convergencia, no asuma que coincide con OLS por ejecutar pocas épocas.

La red por defecto usa salida sigmoide y error cuadrático medio: predice una media de LGD acotada. Arquitectura, Adam/SGD, tasa, dropout, L2, épocas y semilla se modifican en `config.ini` o con `--set`. Se guardan pesos en `models/lgd/red.npz`, el preprocesador en `preprocesador.joblib`, las predicciones de prueba y métricas MAE/RMSE en los archivos de salida de Manuel. Estos artefactos deben cargarse juntos desde una futura aplicación Python; no son compatibles con el antiguo JSON para navegador.

```bash
python entrenar_lgd.py --demo
python -m pytest -q
```

La demostración necesita el parquet preparado; su duración debe medirse con los datos reales. No hay datos crediticios en este checkout: las pruebas usan datos sintéticos y no validan desempeño crediticio. Quedan pendientes los experimentos reales, búsqueda justificada, 30 semillas, pérdida esperada, figuras y aplicación final. Esta integración no presenta esos requisitos como cumplidos.
