# Plan de análisis descriptivo para la estimación de la LGD

## 1. Objetivo

Caracterizar los créditos incumplidos del conjunto de datos de Lending Club, evaluar la calidad y consistencia de las variables financieras disponibles y construir una aproximación a la pérdida dado el incumplimiento (**LGD proxy**), entendida como la proporción del capital originalmente financiado que no se ha recuperado a la fecha de observación.

## 2. Variables seleccionadas

| Variable | Tipo | Propósito |
|---|---|---|
| `loan_status` | Categórica | Identificar los estados de los créditos y seleccionar `Charged Off` y `Default`. |
| `funded_amnt` | Cuantitativa monetaria | Monto originalmente financiado y denominador de la LGD proxy. |
| `total_rec_prncp` | Cuantitativa monetaria | Capital recuperado acumulado hasta la fecha de observación. |
| `recoveries` | Cuantitativa monetaria | Recuperaciones reportadas después del castigo. |
| `collection_recovery_fee` | Cuantitativa monetaria | Costos de recuperación reportados. |
| `out_prncp` | Cuantitativa monetaria | Capital pendiente reportado a la fecha de observación; no equivale necesariamente a EAD. |

## 3. Etapas del análisis

### Etapa 1. Exploración y calidad de datos

- Identificar número de filas y columnas, tipos de datos y estados únicos de `loan_status`.
- Calcular cantidad y porcentaje de valores faltantes por variable.
- Revisar registros duplicados y formatos numéricos.
- Detectar montos negativos, `funded_amnt` igual a cero, `total_rec_prncp` mayor que `funded_amnt` y otras inconsistencias.
- Registrar cuántas observaciones se excluyen y por qué; revisar anomalías antes de eliminarlas.

**Entregable:** tabla de calidad de datos y registro de exclusiones.

### Etapa 2. Descripción del estado crediticio

- Calcular frecuencias absolutas y relativas de cada categoría de `loan_status`.
- Graficar las frecuencias mediante barras.
- Seleccionar `Charged Off` y `Default` para el análisis de LGD, manteniendo ambos grupos identificados.
- Comparar el número de créditos antes y después del filtrado.

**Entregable:** tabla y gráfico de estados; tamaño de la muestra analítica.

### Etapa 3. Análisis univariado de las variables monetarias

Para `funded_amnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee` y `out_prncp`:

- Calcular conteo, media, mediana, desviación estándar, mínimo, máximo, cuartiles y percentiles 5 y 95.
- Calcular proporción de valores nulos y ceros.
- Construir histogramas y diagramas de caja.
- Interpretar asimetría, dispersión y valores extremos sin asumir automáticamente que son errores.

**Entregable:** tabla estadística y visualizaciones por variable.

### Etapa 4. Comparación por estado de incumplimiento

- Comparar `Charged Off` frente a `Default` en montos financiados, capital recuperado, recuperaciones, costos y saldo pendiente.
- Presentar conteos, medianas, medias y percentiles por grupo.
- Usar diagramas de caja para identificar diferencias y dispersión.
- Advertir que los créditos `Default` pueden tener recuperaciones aún en curso.

**Entregable:** tabla comparativa por estado y gráficos.

### Etapa 5. Relaciones y controles de consistencia

- Explorar la relación entre `funded_amnt` y `total_rec_prncp`.
- Explorar la relación entre `recoveries` y `collection_recovery_fee`.
- Contrastar `funded_amnt` y `out_prncp` sin tratar este último como EAD histórica.
- Utilizar diagramas de dispersión y correlaciones descriptivas cuando corresponda.
- Revisar si existen definiciones contables superpuestas antes de combinar recuperaciones.

**Entregable:** gráficos bivariados, correlaciones y hallazgos de consistencia.

### Etapa 6. Construcción de indicadores derivados

**Capital no recuperado por crédito:**

$$
CNR_i = funded\_amnt_i - total\_rec\_prncp_i
$$

**Tasa de recuperación del capital original:**

$$
TR_i = \frac{total\_rec\_prncp_i}{funded\_amnt_i}
$$

**LGD proxy individual:**

$$
LGD_i^{proxy} = 1 - \frac{total\_rec\_prncp_i}{funded\_amnt_i}
$$

**Recuperaciones netas reportadas después del castigo:**

$$
RN_i = recoveries_i - collection\_recovery\_fee_i
$$

**LGD proxy agregada ponderada:**

$$
LGD_{agregada}^{proxy} = \frac{\sum_i(funded\_amnt_i-total\_rec\_prncp_i)}{\sum_i funded\_amnt_i}
$$

Calcular las razones únicamente cuando `funded_amnt > 0`; examinar valores fuera de rango antes de aplicar cualquier recorte. No descontar `recoveries` nuevamente del capital no recuperado sin verificar el tratamiento contable de `total_rec_prncp`.

**Entregable:** base con indicadores por crédito y resumen agregado.

### Etapa 7. Descripción de la LGD proxy

- Calcular media, mediana, desviación estándar, percentiles y rango de la LGD proxy.
- Graficar histograma y diagrama de caja.
- Comparar LGD proxy entre `Charged Off` y `Default`.
- Reportar por separado el capital no recuperado total y la LGD proxy agregada ponderada.
- Interpretar diferencias entre promedio simple y promedio ponderado.

**Entregable:** distribución de severidad de pérdida y comparación por estado.

### Etapa 8. Limitaciones e interpretación

Documentar expresamente que:

1. `funded_amnt` es el monto original, **no** la exposición histórica al incumplimiento (EAD).
2. `total_rec_prncp` acumula pagos de capital y no separa pagos anteriores y posteriores al incumplimiento.
3. La LGD proxy no incluye explícitamente todos los costos de cobranza ni el valor temporal del dinero.
4. Las recuperaciones pueden estar incompletas y variar con el tiempo.
5. `out_prncp` es el saldo reportado a la fecha de observación, no necesariamente el saldo en default.
6. La estimación debe denominarse **proporción de capital original no recuperado (LGD proxy)**, no LGD económica exacta.
7. La selección de créditos con estado conocido puede introducir sesgo si no han tenido tiempos de seguimiento comparables.

**Entregable:** discusión de limitaciones y conclusión metodológica.

## 4. Secuencia sugerida para Jupyter Notebook

1. Importación de librerías y carga de datos.
2. Selección de las seis variables.
3. Revisión de calidad, tipos, nulos y consistencia.
4. Frecuencias y gráfico de `loan_status`.
5. Filtrado de `Charged Off` y `Default`.
6. Estadísticas y visualizaciones univariadas.
7. Comparación por estado crediticio.
8. Análisis bivariado y validaciones.
9. Cálculo de indicadores y LGD proxy.
10. Distribución, resumen agregado, interpretación y limitaciones.

## 5. Resultados esperados

El informe permitirá responder: **¿cuánto capital se financió?, ¿cuánto capital se ha recuperado?, ¿cuánto permanece sin recuperar?, ¿qué porcentaje representa sobre el monto original y cómo difiere entre los estados de incumplimiento?** Los resultados serán descriptivos y no deben interpretarse como estimaciones exactas de LGD económica ni como efectos causales.
