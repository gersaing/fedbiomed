import pandas as pd
from fedbiomed.preprocesador.preprocesamiento import PipelinePreprocesamiento

def _df(minor=5, major=50):
    return pd.DataFrame({
        "f1": list(range(minor+major)),
        "Diagnostic": [0]*major + [1]*minor
    })

def test_auto_cambia_a_under_si_pocos_y_smote():
    pipe = PipelinePreprocesamiento()
    df = _df(minor=5, major=50)
    dfb, metodo = pipe.balanceo_automatico(df, objetivo="Diagnostic", metodo="smote", proporcion=1.0, semilla=123)
    # con min_count < 6 y metodo smote -> cambia a under
    assert metodo == "random-under"

def test_auto_no_cambia_si_suficientes_y_smote():
    pipe = PipelinePreprocesamiento()
    df = _df(minor=10, major=50)
    dfb, metodo = pipe.balanceo_automatico(df, objetivo="Diagnostic", metodo="smote", proporcion=1.0, semilla=123)
    assert metodo == "smote", "Cuando hay suficientes muestras y se pide smote, debe respetarse."

def test_auto_smote_enn_con_pocas_muestras_cambia_a_under():
    pipe = PipelinePreprocesamiento()
    # minor muy chico
    df = _df(minor=3, major=50)
    dfb, metodo = pipe.balanceo_automatico(df, objetivo="Diagnostic", metodo="smote-enn", proporcion=1.0, semilla=123)
    assert metodo == "random-under"

def test_auto_smote_enn_con_suficientes_muestras_se_mantiene():
    pipe = PipelinePreprocesamiento()
    # minor suficiente
    df = _df(minor=20, major=50)
    dfb, metodo = pipe.balanceo_automatico(df, objetivo="Diagnostic", metodo="smote-enn", proporcion=1.0, semilla=123)
    assert metodo == "smote-enn"
