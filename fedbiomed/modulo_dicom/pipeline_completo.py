# ejecutar_pipeline.py
import pandas as pd
from .configuracion import obtener_rutas
from .pipeline_transformacion import PipelineDicom
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

def main(ruta_dicoms: str, limite: int, nombre_nodo: str): 
    print("\033[91m" + __intro__ +"\033[0m")
    ruta_nodo = obtener_ruta_salida_modulo(nombre_nodo)
    rutas = obtener_rutas(ruta_nodo)
    # 2) Procesar los archivos DICOM
    print("[INFO] Iniciando procesamiento de archivos DICOM...")
    pipeline = PipelineDicom(rutas)
    pipeline.procesar_directorio(ruta_dicoms, limite)
    pipeline.extraer_caracteristicas()
    pipeline.fusionar_caracteristicas_metadatos()
    # 3) Preprocesamiento de datos
    df = pd.read_csv(rutas["fusionado"])
    df = eliminar_columnas_irrelevantes(df)
    df = codificar_dataset(df)

    # Elige el flujo según si es entrenamiento o validación
    diagnostico = input('[ENTRADA] Asignación de diagnóstico. Seleccione 0 (no cáncer) o 1 (cáncer), o pulse Enter para validación: ').strip()
    if diagnostico == '0' or diagnostico == '1':
        df = normalizar_entrenamiento(df, rutas["escalador"])
        df = asignar_diagnostico(df, int(diagnostico), rutas["nuevos"])
        actualizar_dataset_entrenamiento(df, rutas["entrenamiento"])
        print(f"[INFO] Flujo de dataset entrenamiento completado.")
    else:
        df = ajustar_dataset_validacion(df, rutas["entrenamiento"])
        normalizar_validacion(df, rutas["escalador"], rutas["validacion"])
        print(f"[INFO] Flujo de dataset validación completado. ")
