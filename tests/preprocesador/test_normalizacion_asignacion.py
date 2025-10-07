from pathlib import Path
import numpy as np
import pandas as pd
from fedbiomed.preprocesador.preprocesamiento import PipelinePreprocesamiento

def test_normalizar_train_y_validacion(tmp_path: Path):
    pipe = PipelinePreprocesamiento()

    # dataset de entrenamiento
    df_train = pd.DataFrame({"a":[0,5,10], "b":[10,10,20]})
    ruta_scaler = tmp_path / "scaler.gz"
    norm = pipe.normalizar_entrenamiento(df_train, ruta_scaler)

    assert ruta_scaler.exists()
    # valores entre 0..1
    assert float(norm.to_numpy().min()) >= 0 - 1e-9
    assert float(norm.to_numpy().max()) <= 1 + 1e-9

    # validación usando el scaler guardado
    df_val = pd.DataFrame({"a":[2.5,7.5], "b":[15,20]})
    salida = tmp_path / "val_norm.csv"
    pipe.normalizar_validacion(df_val, ruta_scaler, salida)
    assert salida.exists()
    dfv = pd.read_csv(salida)
    assert list(dfv.columns) == ["a","b"]

def test_asignar_diagnostico(tmp_path: Path):
    pipe = PipelinePreprocesamiento()
    df = pd.DataFrame({"a":[1,2], "b":[3,4]})
    out = tmp_path / "asignados.csv"
    df2 = pipe.asignar_diagnostico(df, diagnostico=1, salida=str(out))
    assert out.exists()
    assert "Diagnostic" in df2.columns
    assert df2["Diagnostic"].tolist() == [1,1]
