# ejecutar_pipeline.py
import pandas as pd
from .configuracion import obtener_rutas
from .pipeline_transformacion import PipelineDicom
from ..preprocesador.preprocesamiento import PipelinePreprocesamiento
from .preprocesamiento import (
    codificar_dataset,
    normalizar_validacion,
    normalizar_entrenamiento,
    asignar_diagnostico,
    actualizar_dataset_entrenamiento,
    ajustar_dataset_validacion,
    eliminar_columnas_irrelevantes,
)
from .utilidades import obtener_ruta_salida_modulo


__intro__ = """

   __         _ _     _                          _       _   _
  / _|       | | |   (_)                        | |     | | (_)
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |   __| |  _    __   ___  _ __ ___
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` |  / _` | | |  / _| / _ \| '_  `_ \ 
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| |-| (_| | | | | (_ | (_) | | | | | |
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_|  \__,_| |_|  \__| \ __/|_| |_| |_|


"""


def intro():
    """Prints intro for the CLI"""

def main(ruta_dicoms: str, limite_dicoms: int, nombre_nodo: str, clase_diagnostico: int = None, metodo_bal: str = "smote"): 
    print("\033[91m" + __intro__ +"\033[0m")
    ruta_nodo = obtener_ruta_salida_modulo(nombre_nodo)
    rutas = obtener_rutas(ruta_nodo)
    # 2) Procesar los archivos DICOM
    print("[INFO] Iniciando procesamiento de archivos DICOM...")
    pipeline = PipelineDicom(rutas, ruta_dicoms, limite_dicoms)
    pipeline.procesar_directorio()
    pipeline.extraer_caracteristicas()
    pipeline.fusionar_caracteristicas_metadatos()

    # 3) Preprocesamiento (normalizacion, codificacion)
    pipelinePreprocesamiento = PipelinePreprocesamiento()
    df = pd.read_csv(rutas["fusionado"])
    df = pipelinePreprocesamiento.eliminar_columnas_irrelevantes(df)
    df = pipelinePreprocesamiento.codificar_dataset(df)
    #el flujo según si es entrenamiento o validación
    diagnostico = clase_diagnostico
    if diagnostico == 0 or diagnostico == 1:
        print(f"[INFO] Iniciando flujo de preprocesamiento para dataset de entrenamiento con diagnóstico ...")
        df = pipelinePreprocesamiento.normalizar_entrenamiento(df, rutas["escalador"])
        df = pipelinePreprocesamiento.asignar_diagnostico(df, int(diagnostico), rutas["nuevos"])
        pipelinePreprocesamiento.actualizar_dataset_entrenamiento(df, rutas["entrenamiento"])

    df_final = pd.read_csv(rutas["entrenamiento"])
    balance = pipelinePreprocesamiento.calcular_balance(df_final)

    if balance < 30.0:
        print(f"[INFO] Desbalanceo detectado. Clase minoritaria equivale al : {balance:.2f}% del dataset.")
        print(f"Aplicando balanceo con método: {metodo_bal} ")
        df_balanceado,metodo_usado = pipelinePreprocesamiento.balanceo_automatico(df_final, metodo=metodo_bal)
        if metodo_usado == "random-under":
            df_balanceado.to_csv(rutas["bal_under"], index=False)
            print(f"[INFO] Submuestreo aleatorio aplicado. Nuevo dataset guardado en: {rutas['bal_under']}")
        elif metodo_usado == "smote":
            df_balanceado.to_csv(rutas["bal_smote"], index=False)
            print(f"[INFO] SMOTE aplicado. Nuevo dataset guardado en: {rutas['bal_smote']}")
        elif metodo_usado == "smote-enn":
            df_balanceado.to_csv(rutas["bal_smote_enn"], index=False)
            print(f"[INFO] SMOTE-ENN aplicado. Nuevo dataset guardado en: {rutas['bal_smote_enn']}")
        
    
    """
    # 3) Preprocesamiento de datos
    df = pd.read_csv(rutas["fusionado"])
    df = eliminar_columnas_irrelevantes(df)
    df = codificar_dataset(df)

    # Elige el flujo según si es entrenamiento o validación
    diagnostico = clase_diagnostico
    if diagnostico == '0' or diagnostico == '1':
        df = normalizar_entrenamiento(df, rutas["escalador"])
        df = asignar_diagnostico(df, int(diagnostico), rutas["nuevos"])
        actualizar_dataset_entrenamiento(df, rutas["entrenamiento"])
        print(f"[INFO] Flujo de dataset entrenamiento completado.")
    else:
        df = ajustar_dataset_validacion(df, rutas["entrenamiento"])
        normalizar_validacion(df, rutas["escalador"], rutas["validacion"])
        print(f"[INFO] Flujo de dataset validación completado. ")"""
