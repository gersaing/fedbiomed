import os
CARPETA_IMAGENES = "imagenes"
CARPETA_METADATOS = "metadatos"
ARCHIVO_FUSIONADO = "dataset_fusionado.csv"
ARCHIVO_NUEVOS = "dataset_ultima_carga.csv"
ARCHIVO_ENTRENAMIENTO = "dataset_cancer_mama.csv"
ARCHIVO_ESCALADOR = "escalador.pkl"
ARCHIVO_VALIDACION = "dataset_cancer_mama_validacion.csv"
ARCHIVO_CARACTERISTICAS = "caracteristicas.csv"

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
        "validacion": os.path.join(carpeta_base, ARCHIVO_VALIDACION)
    }
    return rutas
