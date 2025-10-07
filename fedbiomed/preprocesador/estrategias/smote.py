# fedbiomed/modulo_dicom/balance/strategies/smote.py
import pandas as pd
from imblearn.over_sampling import SMOTE


def aplicar(df: pd.DataFrame, objetivo: str, proporcion: float = 1.0, semilla: int = 42) -> pd.DataFrame:
    """Aplica SMOTE sobre columnas numéricas, preservando `id` y la columna objetivo.
    `proporcion=1.0` intenta igualar clases (minoría ≈ mayoría).
    """
    if objetivo not in df:
        raise ValueError("Columna requerida no presente: objetivo")

    x = df.drop(columns=[objetivo])
    y = df[objetivo]

    sm = SMOTE(random_state=semilla, sampling_strategy=min(1.0, proporcion))
    X_res, Y_res = sm.fit_resample(x, y)
    df_balanceado = pd.concat(
        [pd.DataFrame(X_res, columns=x.columns), pd.Series(Y_res, name=objetivo)],
        axis=1
    )
    df_balanceado = df_balanceado.reset_index(drop=True).sample(frac=1, random_state=semilla).reset_index(drop=True)

    return  df_balanceado