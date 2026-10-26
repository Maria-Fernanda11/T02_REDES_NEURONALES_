# Construcción de la LGD a partir de las variables del conjunto de datos

## 1. Objetivo y definición

La **pérdida dado el incumplimiento** (*Loss Given Default*, LGD) mide qué proporción de la exposición al momento del incumplimiento no se recupera, considerando los flujos posteriores de recuperación y sus costos. La definición económica general es:

$$
LGD_i^{econ}=1-\frac{VP(\text{recuperaciones netas posteriores al default})_i}{EAD_i}
$$

Aquí, $EAD_i$ es la exposición del préstamo $i$ **en la fecha del incumplimiento** y $VP$ representa el valor presente de los flujos. El diccionario de Lending Club contiene saldos y recuperaciones acumuladas, pero no permite identificar directamente, para cada préstamo, la EAD histórica ni reconstruir todos los flujos fechados posteriores al incumplimiento. Por eso se calcula una **LGD proxy**, que debe interpretarse como **proporción de capital originalmente financiado no recuperada a la fecha de observación**, y no como LGD económica o regulatoria exacta.

## 2. Población de análisis y filtrado

Se consideran préstamos cuyo `loan_status` es `Charged Off` o `Default`. Se recomienda informar sus resultados **por separado**: `Charged Off` identifica préstamos castigados y `Default` puede contener casos cuyo proceso de recuperación aún no concluye. Los estados `Fully Paid`, `Current`, `In Grace Period` y `Late` quedan fuera de este cálculo específico porque no cumplen la definición de población incumplida adoptada. Este criterio no implica que una mora prolongada nunca pueda considerarse default bajo otra definición institucional.

Antes del filtrado se revisan los valores exactos de `loan_status` y se cuantifican los registros excluidos. No se debe asumir que los estados disponibles son idénticos en todas las versiones del conjunto.

## 3. Variables seleccionadas y justificación

| Variable | Papel | Justificación y precaución |
|---|---|---|
| `loan_status` | Filtro | Permite identificar la población con incumplimiento según la definición adoptada; no es parte de la operación aritmética. |
| `funded_amnt` | Denominador | Monto inicialmente financiado; base observable y positiva para calcular una proporción. **No equivale a EAD al default**. |
| `total_rec_prncp` | Numerador, vía resta | Capital acumulado recibido hasta la fecha de corte. Incluye cobros de capital de distintos momentos; **no identifica por sí solo recuperaciones exclusivamente posteriores al default**. |
| `recoveries` | Indicador complementario | Recuperaciones registradas después del castigo; ayuda a analizar gestión de cobro. No se resta automáticamente otra vez de `total_rec_prncp`. |
| `collection_recovery_fee` | Indicador complementario | Costos de recuperación reportados, útiles para calcular una recuperación neta nominal aproximada. |
| `out_prncp` | Comprobación descriptiva | Capital pendiente en la fecha de observación; no debe sustituir automáticamente a la EAD histórica, especialmente en préstamos castigados. |

## 4. Fórmulas de cálculo

### 4.1. LGD proxy individual

$$
\boxed{LGD_i^{proxy}=\frac{funded\_amnt_i-total\_rec\_prncp_i}{funded\_amnt_i}=1-\frac{total\_rec\_prncp_i}{funded\_amnt_i}}
$$

El resultado es una **proporción** entre 0 y 1 cuando los datos son consistentes. Multiplicar por 100 permite expresarlo en porcentaje. Un resultado de 0,60 indica que, al corte de datos, **el 60 % del capital originalmente financiado no figura como recuperado**. No significa necesariamente que se haya perdido definitivamente ese porcentaje del saldo existente al default.

### 4.2. Capital no recuperado (unidades monetarias)

$$
\boxed{CNR_i=funded\_amnt_i-total\_rec\_prncp_i}
$$

Esta cantidad está expresada en la **moneda original del conjunto** y no es una pérdida económica final auditada.

### 4.3. Recuperaciones netas posteriores al castigo (indicador complementario)

$$
\boxed{RN_i=recoveries_i-collection\_recovery\_fee_i}
$$

Se interpreta como una aproximación **nominal** a las recuperaciones netas reportadas después del castigo. No se combina mecánicamente con la fórmula principal porque los campos pueden registrar conceptos superpuestos. Tampoco se descuenta a valor presente.

### 4.4. LGD proxy agregada ponderada

$$
\boxed{LGD_{agregada}^{proxy}=\frac{\sum_i(funded\_amnt_i-total\_rec\_prncp_i)}{\sum_i funded\_amnt_i}}
$$

La ponderación por capital evita que un préstamo pequeño tenga el mismo peso que uno grande. Conviene reportar también la media, mediana, percentiles y dispersión de las LGD individuales, además del número de préstamos incluidos.

## 5. Ejemplo ilustrativo (datos hipotéticos)

| Crédito | Financiado | Capital recuperado | Capital no recuperado | LGD proxy |
|---|---:|---:|---:|---:|
| A | 10.000 | 7.000 | 3.000 | 30 % |
| B | 20.000 | 8.000 | 12.000 | 60 % |
| C | 15.000 | 3.000 | 12.000 | 80 % |
| **Total** | **45.000** | **18.000** | **27.000** | **60 % agregada** |

La LGD agregada se calcula como $27.000/45.000=60\%$. Este ejemplo **no corresponde a resultados calculados con la base real**.

## 6. Implementación reproducible en Python

El siguiente código asume que el DataFrame de préstamos se llama `df`. El archivo de diccionario describe variables, pero **no contiene por sí solo todos los registros de préstamos necesarios para producir resultados numéricos**.

```python
import pandas as pd
import numpy as np

# 1. Validar que existan las variables requeridas
cols = [
    'loan_status', 'funded_amnt', 'total_rec_prncp',
    'recoveries', 'collection_recovery_fee', 'out_prncp'
]
faltantes = [c for c in cols if c not in df.columns]
if faltantes:
    raise KeyError(f'Columnas faltantes: {faltantes}')

# 2. Revisar estados y seleccionar incumplidos
print('Estados originales:')
print(df['loan_status'].value_counts(dropna=False))

estados_default = ['Charged Off', 'Default']
df_lgd = df.loc[df['loan_status'].isin(estados_default), cols].copy()

# 3. Conversión de campos monetarios
monetarias = [c for c in cols if c != 'loan_status']
for c in monetarias:
    df_lgd[c] = pd.to_numeric(df_lgd[c], errors='coerce')

# 4. Control de calidad: contar faltantes e inconsistencias
sin_datos = df_lgd[['funded_amnt', 'total_rec_prncp']].isna().any(axis=1)
inconsistentes = (
    (df_lgd['funded_amnt'] <= 0) |
    (df_lgd['total_rec_prncp'] < 0) |
    (df_lgd['total_rec_prncp'] > df_lgd['funded_amnt'])
)
print('Incumplidos seleccionados:', len(df_lgd))
print('Con datos faltantes:', sin_datos.sum())
print('Con valores inconsistentes:', (inconsistentes & ~sin_datos).sum())

# Conservar una copia para auditar los casos excluidos
casos_revision = df_lgd.loc[sin_datos | inconsistentes].copy()
df_lgd = df_lgd.loc[~(sin_datos | inconsistentes)].copy()

# 5. LGD proxy y capital no recuperado

df_lgd['capital_no_recuperado'] = (
    df_lgd['funded_amnt'] - df_lgd['total_rec_prncp']
)
df_lgd['LGD_proxy'] = (
    df_lgd['capital_no_recuperado'] / df_lgd['funded_amnt']
)

# 6. Recuperaciones netas nominales (solo si ambos datos existen)
df_lgd['recuperaciones_netas'] = (
    df_lgd['recoveries'] - df_lgd['collection_recovery_fee']
)

# 7. Indicadores agregados
lgd_ponderada = (
    df_lgd['capital_no_recuperado'].sum() /
    df_lgd['funded_amnt'].sum()
    if df_lgd['funded_amnt'].sum() > 0 else np.nan
)

print('\nNúmero de créditos:', len(df_lgd))
print('Capital financiado:', df_lgd['funded_amnt'].sum())
print('Capital no recuperado:', df_lgd['capital_no_recuperado'].sum())
print('LGD proxy agregada (%):', round(100 * lgd_ponderada, 2))
print('\nDistribución LGD proxy:')
print(df_lgd['LGD_proxy'].describe(percentiles=[.10, .25, .5, .75, .90]))

# 8. Comparar los dos estados de incumplimiento
resumen = df_lgd.groupby('loan_status').agg(
    creditos=('LGD_proxy', 'size'),
    financiado=('funded_amnt', 'sum'),
    capital_no_recuperado=('capital_no_recuperado', 'sum'),
    lgd_media=('LGD_proxy', 'mean'),
    lgd_mediana=('LGD_proxy', 'median'),
    recuperaciones_netas=('recuperaciones_netas', 'sum')
)
resumen['lgd_ponderada'] = (
    resumen['capital_no_recuperado'] / resumen['financiado']
)
print('\nResultados por estado:')
print(resumen)
```

**Control de calidad:** los registros inconsistentes se separan para revisión en `casos_revision`; no se fuerzan silenciosamente al intervalo $[0,1]$. Si los datos de `recoveries` o `collection_recovery_fee` son nulos, la recuperación neta de ese préstamo debe tratarse como no disponible y no como cero sin justificación. El resumen de recuperaciones netas debe acompañarse de un conteo de valores completos.

## 7. Interpretación de los resultados

Se recomienda presentar: (a) número de créditos incumplidos incluidos y excluidos, (b) capital financiado total, (c) capital no recuperado total, (d) LGD proxy agregada ponderada, (e) distribución de LGD proxy individual y (f) comparación de `Charged Off` frente a `Default`.

La LGD proxy **no es una probabilidad de incumplimiento**: la PD indica la posibilidad de incumplir, mientras que la LGD describe la severidad de la pérdida condicionada al incumplimiento. Tampoco debe interpretarse el capital no recuperado como pérdida contable definitiva, pues las recuperaciones futuras y los criterios de registro pueden modificar el resultado.

## 8. Limitaciones y discusión crítica

1. **EAD no observada:** `funded_amnt` es el monto inicial, no el saldo exacto al incumplimiento. Esta es la principal razón por la que la medida se denomina proxy.
2. **Momento de los cobros:** `total_rec_prncp` agrega capital recibido antes y después del incumplimiento, impidiendo aislar recuperaciones posteriores al evento.
3. **Recuperaciones y doble contabilización:** `recoveries` y `total_rec_prncp` no deben sumarse ni restarse sin verificar su relación contable en la versión concreta de los datos.
4. **Costos:** los gastos de cobranza no están incorporados en la LGD proxy principal; `collection_recovery_fee` solo permite un análisis complementario parcial.
5. **Descuento temporal:** sin fechas y montos de cada recuperación no puede estimarse adecuadamente su valor presente.
6. **Censura de recuperaciones:** los préstamos `Default` pueden no haber completado su proceso de recuperación; incluso los `Charged Off` pueden generar cobros posteriores.
7. **Fecha de corte y comparabilidad:** préstamos originados en momentos diferentes tienen distinto tiempo para pagar o recuperar capital; las cohortes recientes pueden sesgar la comparación.
8. **Selección de estados:** limitarse a `Charged Off` y `Default` responde a una definición operacional; otras políticas pueden tratar moras prolongadas como default. La elección debe mantenerse consistente en todo el trabajo.
9. **Saldo pendiente:** `out_prncp` es un saldo a fecha de observación y puede no reflejar el capital que estaba expuesto en la fecha histórica de incumplimiento.
10. **Calidad de datos:** faltantes, importes negativos o capital recuperado superior al financiado requieren auditoría y reporte de exclusiones.
11. **Uso predictivo:** `total_rec_prncp`, `recoveries` y otros campos posteriores a la originación no deben utilizarse como predictores de default al momento de conceder un préstamo, por riesgo de fuga de información. Aquí se usan para construir un resultado descriptivo ex post.

## 9. Conclusión metodológica

Con las variables del conjunto se construye una **LGD proxy** definida como la proporción de capital inicialmente financiado que no ha sido recuperada a la fecha de observación entre los créditos clasificados como incumplidos. `loan_status` determina la población, `funded_amnt` constituye la base de comparación y `total_rec_prncp` mide el capital acumulado recibido. `recoveries`, `collection_recovery_fee` y `out_prncp` se reservan para análisis complementarios y controles, sin mezclarlos mecánicamente en una fórmula que podría duplicar flujos.

La medida permite reportar tanto una severidad relativa por préstamo como un capital no recuperado en unidades monetarias y una LGD proxy agregada ponderada. **No debe presentarse como LGD económica estricta**, pues no se dispone de EAD histórica, calendario completo de recuperaciones ni costos integrales descontados. La claridad de esta distinción hace que el cálculo sea reproducible, interpretable y metodológicamente defendible con los datos disponibles.
