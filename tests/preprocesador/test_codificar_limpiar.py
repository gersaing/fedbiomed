import pandas as pd
import pytest
from fedbiomed.preprocesador.preprocesamiento import PipelinePreprocesamiento

def test_calcular_balance_ok():
    pipe = PipelinePreprocesamiento()
    df = pd.DataFrame({"y":[0]*30 + [1]*10})
    pct = pipe.calcular_balance(df, objetivo="y")
    # minor = 10, total = 40 -> 25%
    assert abs(pct - 25.0) < 1e-6

def test_calcular_balance_col_faltante():
    pipe = PipelinePreprocesamiento()
    df = pd.DataFrame({"x":[1,2,3]})
    with pytest.raises(ValueError):
        pipe.calcular_balance(df, objetivo="y")
