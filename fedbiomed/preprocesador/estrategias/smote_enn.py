import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

def aplicar(df: pd.DataFrame, objetivo: str, col_id: str = "id", proporcion: float = 1.0, semilla: int = 42) -> pd.DataFrame:
    if objetivo not in df:
            raise ValueError("Columna requerida no presente: objetivo")

    x = df.drop(columns=[objetivo])
    y = df[objetivo]

    se = SMOTEENN(random_state=semilla, enn=EditedNearestNeighbours(sampling_strategy="majority", n_neighbors=3))
    X_res, Y_res = se.fit_resample(x, y)

    df_balanceado = pd.concat(
        [pd.DataFrame(X_res, columns=x.columns), pd.Series(Y_res, name=objetivo)],
        axis=1
    )
    df_balanceado = df_balanceado.reset_index(drop=True).sample(frac=1, random_state=semilla).reset_index(drop=True)
    return df_balanceado