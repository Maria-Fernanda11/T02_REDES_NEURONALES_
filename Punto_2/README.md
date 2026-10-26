# Metodología y Decisiones - Modelado de Riesgo de Crédito

## 1. Filtrado para los créditos que incumplieron (Clasificación - PD)

### Construcción de la variable objetivo dicotómica

Se creó una variable denominada `incumplimiento`, definida de la siguiente manera:

- **1 (Incumplimiento):** créditos clasificados como `Charged Off` o `Default`.
- **0 (No incumplimiento):** créditos clasificados como `Fully Paid`.

Se excluyeron los estados `Current`, `In Grace Period`, `Late (16-30 days)` y `Late (31-120 days)`, debido a que corresponden a créditos cuyo resultado final todavía es incierto.

Este procedimiento permite diferenciar los créditos pagados completamente de aquellos que presentaron **incumplimiento**, evitando clasificar atrasos temporales como incumplimientos definitivos.

**Nota metodológica:** Para reducir el sesgo de selección, es importante verificar que los créditos analizados hayan tenido suficiente tiempo para alcanzar un resultado definitivo.

---

## 2. Construcción de la variable LGD (Punto de Regresión)

### 2.1 Definición y Población Objetivo
La severidad de la pérdida o **Loss Given Default (LGD)** representa el porcentaje del monto expuesto que la entidad financiera no logra recuperar una vez el deudor entra en incumplimiento.

- **Población para la regresión:** El modelo de regresión se entrena **únicamente sobre el subconjunto de créditos que incumplieron** (`incumplimiento == 1`, es decir, `loan_status` en `Charged Off` o `Default`, totalizando 46,467 registros en la base de datos). 
- **Justificación:** Conceptualmente, LGD es una variable condicional al incumplimiento ($E[\text{LGD} \mid \text{Default} = 1]$). Los créditos pagados (`Fully Paid`) no entran en este modelo de severidad, pues su probabilidad de no-pérdida se modela en la etapa de clasificación (PD).

### 2.2 Variables de la Base de Datos para el Cálculo de LGD
Para la construcción de la variable objetivo de regresión se identificaron las siguientes variables clave en el dataset (`loan.csv`):

1. **`funded_amnt`** (Monto comprometido/financiado): Representa la exposición inicial total otorgada al prestatario (proxy de la Exposición al Incumplimiento o EAD base).
2. **`total_rec_prncp`** (Capital pagado a la fecha): Monto del capital amortizado efectivamente por el prestatario antes o durante el proceso de impago.
3. **`recoveries`** (Recuperaciones brutas posteriores al castigo): Monto recuperado con posterioridad a la declaración del crédito como pérdida/incobrable (*post charge off gross recovery*).
4. **`collection_recovery_fee`** (Costos/comisiones de cobranza): Gastos incurridos por agencias de cobranza o gestiones legales (*post charge off collection fee*), los cuales deben deducirse de las recuperaciones brutas.

#### Análisis sobre la variable `out_prncp` (Capital pendiente):
- **Definición:** `out_prncp` representa el capital pendiente de pago (*outstanding principal*).
- **Comportamiento en los datos:**
  - En los créditos **`Charged Off`** (45.248 créditos, 97.4% de los incumplidos), `out_prncp` es **0.00 en el 100% de los casos**. Esto ocurre por una convención contable: al castigar el crédito como incobrable (*write-off*), el saldo pendiente se elimina de los libros activos y se traslada a cuentas de orden o procesos de cobranza (`recoveries`).
  - En los créditos **`Default`** (1.219 créditos, 2.6% de los incumplidos), el crédito aún no ha sido castigado formalmente, por lo que `out_prncp` refleja el saldo activo no amortizado ($\text{funded\_amnt} - \text{total\_rec\_prncp}$).
- **Decisión:** **NO utilizar `out_prncp` directamente para el cálculo del LGD.** Si se utilizara para medir la pérdida, asignaría erróneamente un 0% de pérdida a todos los créditos castigados. La variable consistente y universal para medir el capital recuperado en todo el ciclo de vida del crédito es **`total_rec_prncp`**, complementada con `recoveries` netas.

### 2.3 Formulación Matemática de la LGD
Siguiendo las mejores prácticas bancarias y la normativa de Basilea para riesgo de crédito, se adopta el **enfoque de pérdida neta sobre capital**:

1. **Tasa de Recuperación Neta (Recovery Rate - RR):**
   $$RR = \frac{\text{total\_rec\_prncp} + (\text{recoveries} - \text{collection\_recovery\_fee})}{\text{funded\_amnt}}$$

2. **Loss Given Default (LGD):**
   $$\text{LGD} = 1 - RR = 1 - \frac{\text{total\_rec\_prncp} + \text{recoveries} - \text{collection\_recovery\_fee}}{\text{funded\_amnt}}$$
   
   Equivalentemente, mide la proporción del capital que no fue cubierto ni por amortización regular ni por cobranza neta:
   $$\text{LGD} = \frac{\text{funded\_amnt} - \text{total\_rec\_prncp} - (\text{recoveries} - \text{collection\_recovery\_fee})}{\text{funded\_amnt}}$$

### 2.4 Decisiones Metodológicas Adicionales
1. **Acotamiento (Clipping a $[0, 1]$):**
   - Teóricamente, $\text{LGD} \in [0, 1]$ (0% indica recuperación total y 100% pérdida íntegra).
   - Debido a penalizaciones, cobro de intereses moratorios o variaciones en liquidaciones, algunos registros pueden resultar matemáticamente en valores ligeramente inferiores a 0 o superiores a 1.
   - **Decisión:** Se aplica un truncamiento estricto: $\text{LGD}_{\text{final}} = \min(\max(\text{LGD}, 0), 1)$.
2. **Prevención de Fuga de Información (*Data Leakage*):**
   - Las variables utilizadas para calcular la variable objetivo (`total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `total_pymnt`, `last_pymnt_d`, etc.) son variables *ex-post* que solo se conocen después del incumplimiento.
   - **Decisión:** Estas variables se usarán exclusivamente para computar la variable target `LGD` y deberán ser eliminadas del conjunto de variables predictoras (features) de la red neuronal/modelo de regresión. Las predictoras solo deben contener información disponible al momento de la originación del crédito (ingresos, DTI, empleo, historial crediticio, tasa, plazo, grado, etc.).

### 2.5 Análisis de Significancia por Grado de Calificación (`grade`) y Decisión de Agrupamiento
Se evaluó si la severidad de pérdida ($LGD$) presenta diferencias estadísticamente significativas a lo largo de los 7 grados de riesgo crediticio asignados al originar (`A`, `B`, `C`, `D`, `E`, `F`, `G`):

- **ANOVA One-Way:** $F = 953.77$, $p\text{-valor} < 10^{-16}$. Se rechaza contundentemente la hipótesis nula de igualdad de medias entre grados.
- **Prueba Post-Hoc de Tukey HSD ($\alpha = 0.05$):**
  - Al contrastar los 21 pares de grados posibles, **20 de las 21 comparaciones son altamente significativas ($p\text{-adj} < 0.001$)**.
  - **Excepción única:** La comparación entre el **Grado F** ($\text{LGD}_{\text{media}} = 85.97\%$) y el **Grado G** ($\text{LGD}_{\text{media}} = 87.87\%$) arrojó $p\text{-adj} = 0.1424$ (Intervalo de confianza al 95%: $[-0.30\%, +4.10\%]$), por lo cual **NO se rechaza la igualdad de medias**.
- **Decisión para el modelamiento predictivo:**
  - Se identifican **6 grupos homogéneos estadísticamente independientes**: `A`, `B`, `C`, `D`, `E` y `{F, G}`.
  - Para evitar parámetros redundantes y reducir la varianza muestral (el grado `G` cuenta con solo 814 observaciones), en la etapa de preprocesamiento/ingeniería de características para la regresión se colapsarán las categorías `F` y `G` en un único nivel (`F_G` o `Alto_Riesgo`), quedando 6 grupos efectivos a codificar.

### 2.6 Especificación de la Regresión Lineal Clásica con Variables Indicadoras ($I$)
Para el estudio econométrico de referencia de la LGD, se especifica el modelo lineal clásico:

$$LGD_i = \beta_0 + \beta_1 (funded\_amnt\_k_i) + \beta_2 I_{term60, i} + \sum_{k \in \{B, C, D, E, FG\}} \beta_k I_{grade\_k, i} + \beta_3 \ln(annual\_inc_i) + \beta_4 (dti_i) + \beta_5 (revol\_util_i) + \beta_6 (inq\_6m_i) + \beta_7 (pub\_rec_i) + \beta_8 I_{verified, i} + \varepsilon_i$$

Donde las variables cualitativas se expresan mediante variables indicadoras binarias $I \in \{0, 1\}$ (tomando como base de referencia el plazo de 36 meses, el Grado A y la condición de no verificado).

- **Resultados de Significancia:**
  - $I_{term60}$: Altamente significativa ($t = 45.81$, $p < 0.001$, $\hat{\beta} \approx +0.101$).
  - Indicadoras de Grado $I_{grade\_k}$: Estrictamente crecientes y significativas ($p < 0.001$), con $\hat{\beta}$ desde $+0.033$ (Grado B) hasta $+0.152$ (Grado FG) respecto al Grado A.
  - Variables de control no significativas al 5%: el monto financiado ($funded\_amnt\_k$) y las consultas recientes ($inq\_6m$) una vez controladas las variables de grado, plazo e ingresos.
- **Métricas del Modelo y Acotamiento Económico:**
  - $R^2 \approx 0.177$ (explica el 17.7% de la varianza de la severidad).
  - $MAE \approx 0.145$ y $RMSE \approx 0.182$.
  - **Efecto del Acotamiento $\min(1, \max(0, \hat{y}))$:** Al evaluar las predicciones de OLS sobre la muestra de créditos en incumplimiento, el rango predicho se situó naturalmente entre $0.564$ y $0.989$, sin generar predicciones menores a 0 ni mayores a 1. El modelo lineal capturó que la pérdida esperada condicional al default siempre supera el 50%, por lo que la cota de truncamiento no se activa en la muestra de prueba.
  - **Comparación con GLM Fraccional (Logit):** Al incorporar una función de enlace sigmoide nativa $E[LGD \mid X] = \frac{1}{1 + e^{-X\beta}}$, se obtienen métricas comparables ($R^2 \approx 0.175$, $MAE \approx 0.140$), sirviendo como puente conceptual directo a la activación sigmoide de la futura red neuronal multicapa.
- **Diagnóstico del Error:** Ausencia de autocorrelación ($DW \approx 2.0$), con heterocedasticidad presente (Breusch-Pagan $p < 0.001$) debido a la asimetría propia de la cartera de consumo.
