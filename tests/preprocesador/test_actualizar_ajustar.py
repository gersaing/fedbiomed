from pathlib import Path
import pandas as pd
from fedbiomed.preprocesador.preprocesamiento import PipelinePreprocesamiento

def test_actualizar_dataset_entrenamiento_y_ajustar_validacion(tmp_path: Path):
    pipe = PipelinePreprocesamiento()

    # dataset entrenamiento inicial
    train_csv = tmp_path / "train.csv"
    df0 = pd.DataFrame({
        "f1":[0.0, 1.0],
        "f2":[10.0, 20.0],
        "Diagnostic":[0,1],
        "ImageLaterality_L":[1,0],  # simula columnas ya codificadas
        "ImageLaterality_R":[0,1],
    })
    df0.to_csv(train_csv, index=False)

    # nuevos datos (contiene duplicado y una fila nueva)
    df_new = pd.DataFrame({
        "f1":[1.0, 2.0],
        "f2":[20.0, 30.0],
        "Diagnostic":[1,0],
        "ImageLaterality_L":[0,1],
        "ImageLaterality_R":[1,0],
    })
    pipe.actualizar_dataset_entrenamiento(df_new, str(train_csv))
    df_upd = pd.read_csv(train_csv)
    # Debe contener 3 filas (2 originales + 1 nueva; el duplicado se elimina)
    assert len(df_upd) == 3

    # ajustar validación para que coincida con columnas de entrenamiento (y sin Diagnostic)
    df_val = pd.DataFrame({
        "f1":[5.0],
        "f2":[15.0],
        # faltan columnas one-hot; deben crearse como 0
    })
    df_adj = pipe.ajustar_dataset_validacion(df_val, str(train_csv))
    cols_train = pd.read_csv(train_csv).columns.tolist()

    # Esperamos las columnas de entrenamiento excepto 'Diagnostic'
    cols_expected = [c for c in cols_train if c != "Diagnostic"]
    assert list(df_adj.columns) == cols_expected

    assert "ImageLaterality_L" in df_adj.columns and "ImageLaterality_R" in df_adj.columns
    assert df_adj[["ImageLaterality_L", "ImageLaterality_R"]].iloc[0].isin([0, 1]).all()
