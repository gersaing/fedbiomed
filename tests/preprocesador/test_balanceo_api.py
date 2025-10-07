import json
from pathlib import Path
import pandas as pd
import pytest

from fedbiomed.preprocesador.preprocesamiento import PipelinePreprocesamiento

@pytest.fixture
def csv_clasico(tmp_path: Path):
    df = pd.DataFrame({
        "f1": list(range(40)),
        "f2": list(range(40))[::-1],
        "Diagnostic": [0]*30 + [1]*10,  # desbalanceado 3:1
    })
    ruta = tmp_path / "train.csv"
    df.to_csv(ruta, index=False)
    return ruta

def test_balanceo_random_under_genera_csv_y_json(csv_clasico, tmp_path):
    pipe = PipelinePreprocesamiento()
    salida = pipe.balanceo(
        ruta_csv=str(csv_clasico),
        metodo="random-under",
        objetivo="Diagnostic",
        proporcion=1.0,
        semilla=42,
        escribir_reporte=True,
    )
    # CSV de salida
    out = Path(salida)
    assert out.exists()
    dfb = pd.read_csv(out)
    vc = dfb["Diagnostic"].value_counts().to_dict()
    assert vc.get(0,0) == vc.get(1,0), "Debe quedar balanceado 1:1 con under-sampling"

    # JSON de reporte
    rep = out.with_suffix(".balance_report.json")
    assert rep.exists()
    meta = json.loads(rep.read_text(encoding="utf-8"))
    assert meta["method"] == "random-under"
    assert meta["input_csv"].endswith("train.csv")

def test_balanceo_valida_archivo_y_objetivo(tmp_path):
    pipe = PipelinePreprocesamiento()
    # archivo inexistente
    with pytest.raises(FileNotFoundError):
        pipe.balanceo(str(tmp_path / "nope.csv"), "random-under", "Diagnostic")
    # objetivo inexistente
    ruta = tmp_path / "d.csv"
    pd.DataFrame({"x":[1,2], "y":[0,1]}).to_csv(ruta, index=False)
    with pytest.raises(ValueError):
        pipe.balanceo(str(ruta), "random-under", "Diagnostic")

def test_balanceo_metodo_no_soportado(csv_clasico):
    pipe = PipelinePreprocesamiento()
    with pytest.raises(ValueError):
        pipe.balanceo(str(csv_clasico), "no-existe", "Diagnostic")
