"""Preparación temporal de PD auxiliar y cohorte común para pérdida esperada."""
from pathlib import Path
import numpy as np
import pandas as pd
from preparar_lgd import transformar_predictoras
from src.configuracion import cargar_config

ETIQUETAS_PD = {
    'Fully Paid':0, 'Does not meet the credit policy. Status:Fully Paid':0,
    'Charged Off':1, 'Default':1, 'Late (31-120 days)':1,
    'Does not meet the credit policy. Status:Charged Off':1,
}
EXCLUIDOS_PD = {'Current','Issued','In Grace Period','Late (16-30 days)'}


def preparar_pd_y_cohorte(bruto, salida):
    """Años <=2010 train, 2011 valid, 2012 test, como el notebook de PD.

    La cohorte común es test de PD. Puede solaparse con train de LGD, por lo que
    el producto no se presenta como evaluación independiente de pérdida esperada.
    """
    estados=bruto.loan_status.astype('string').str.strip()
    desconocidos=set(estados.dropna().unique())-set(ETIQUETAS_PD)-EXCLUIDOS_PD
    if desconocidos or estados.isna().any():
        raise ValueError(f'loan_status ausente o no reconocido: {desconocidos}')
    df=bruto.loc[estados.isin(ETIQUETAS_PD)].copy()
    df['y']=estados.loc[df.index].map(ETIQUETAS_PD).astype(int)
    df=transformar_predictoras(df)
    fecha=pd.to_datetime(df.issue_d,format='%b-%Y',errors='raise')
    df['particion']=np.select([fecha.dt.year<=2010,fecha.dt.year==2011,fecha.dt.year==2012],
                              ['train','valid','test'],default='fuera')
    cfg=cargar_config('config.ini','pd')['datos']
    columnas=list(dict.fromkeys(cfg['numericas']+cfg['categoricas']))
    df=df.loc[df.particion!='fuera'].copy()
    for nombre in ['train','valid','test']:
        parte=df.loc[df.particion==nombre]
        if parte.empty or set(parte.y.unique())!={0,1}:
            raise ValueError(f'PD auxiliar requiere ambas clases en la partición temporal {nombre}')
    destino=Path(salida);destino.mkdir(parents=True,exist_ok=True)
    ruta=destino/'pd_temporal.parquet'
    df[columnas+['y','particion']].to_parquet(ruta,index=False)
    if 'id' not in df:
        raise ValueError('Falta id original para alinear créditos')
    cohorte=df.loc[df.particion=='test',columnas+['id','funded_amnt']].copy()
    if cohorte.id.isna().any() or cohorte.id.duplicated().any():
        raise ValueError('IDs ausentes o duplicados; revisar antes de combinar modelos')
    montos=pd.to_numeric(cohorte.funded_amnt,errors='raise').to_numpy(dtype=float)
    if not np.isfinite(montos).all() or (montos<=0).any():
        raise ValueError('Exposiciones no válidas en la cohorte')
    ruta_cohorte=destino/'cohorte_perdida.parquet'
    cohorte.to_parquet(ruta_cohorte,index=False)
    return ruta,ruta_cohorte,df.groupby('particion').y.agg(['count','mean'])
