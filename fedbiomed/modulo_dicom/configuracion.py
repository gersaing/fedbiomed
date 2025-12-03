import os
CARPETA_IMAGENES = "imagenes"
CARPETA_METADATOS = "metadatos"
ARCHIVO_FUSIONADO = "dataset_fusionado.csv"
ARCHIVO_NUEVOS = "dataset_ultima_carga.csv"
ARCHIVO_ENTRENAMIENTO = "dataset_cancer_mama.csv"
ARCHIVO_ESCALADOR = "escalador.pkl"
ARCHIVO_VALIDACION = "dataset_cancer_mama_validacion.csv"
ARCHIVO_CARACTERISTICAS = "caracteristicas.csv"
ARCHIVO_BALANCEADO_SMOTE = "dataset_cancer_mama_smote.csv"
ARCHIVO_BALANCEADO_UNDER = "dataset_cancer_mama_under.csv"
ARCHIVO_BALANCEADO_SMOTE_ENN = "dataset_cancer_mama_smtn.csv"
RUTA_CLASIFICACION = "fedbiomed/fedbiomed/modulo_dicom/vista"
RUTA_IMAGEN_FED = "fedbiomed/fedbiomed/modulo_dicom/vista/fed-BioMed-I.png"

def obtener_rutas(ruta_nodo):
    carpeta_base = ruta_nodo
    os.makedirs(carpeta_base, exist_ok=True)
    rutas = {
        "imagenes": os.path.join(carpeta_base, CARPETA_IMAGENES),
        "metadatos": os.path.join(carpeta_base, CARPETA_METADATOS),
        "caracteristicas": os.path.join(carpeta_base, ARCHIVO_CARACTERISTICAS),
        "fusionado": os.path.join(carpeta_base,  ARCHIVO_FUSIONADO),
        "nuevos": os.path.join(carpeta_base,  ARCHIVO_NUEVOS),
        "entrenamiento": os.path.join(carpeta_base, ARCHIVO_ENTRENAMIENTO),
        "escalador": os.path.join(carpeta_base, ARCHIVO_ESCALADOR),
        "validacion": os.path.join(carpeta_base, ARCHIVO_VALIDACION),
        "bal_smote": os.path.join(carpeta_base, ARCHIVO_BALANCEADO_SMOTE),
        "bal_under": os.path.join(carpeta_base, ARCHIVO_BALANCEADO_UNDER),
        "bal_smote_enn": os.path.join(carpeta_base, ARCHIVO_BALANCEADO_SMOTE_ENN),
        "imagen_fed" : os.path.join(RUTA_IMAGEN_FED)
    }
    return rutas
