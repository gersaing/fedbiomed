# fedbiomed/modulo_dicom/balance/strategies/enn.py
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def aplicar(df: pd.DataFrame, objetivo: str, proporcion: float = 1.0, semilla: int = 42) -> pd.DataFrame:
    # Nota: `proporcion` no aplica estrictamente en ENN; se ignora.

    if objetivo not in df:
        raise ValueError("Columna requerida no presente: {objetivo}")
    # separación de características y objetivo
    x = df.drop(columns=[objetivo])
    y = df[objetivo]

    random_under = RandomUnderSampler(random_state=semilla)
    x_res, y_res = random_under.fit_resample(x, y)

    df_balanceado = pd.concat(
        [pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res, name=objetivo)],
        axis=1
    )
    df_balanceado = df_balanceado.reset_index(drop=True).sample(frac=1, random_state=semilla).reset_index(drop=True)

    return  df_balanceado