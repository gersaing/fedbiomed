from .utilidades import obtener_lista_dicoms
from .cargador import cargar_dicom, extraer_metadatos, guardar_metadatos_txt, extraer_imagen_jpg
from .extractor_caracteristicas import extraer_caracteristicas
from .utilidades import guardar_caracteristicas_csv, fusionar_caracteristicas_metadatos
import os

class PipelineDicom:
    def __init__(self, rutas):
        self.rutas = rutas
        self.carpeta_metadatos = rutas["metadatos"]
        self.carpeta_imagenes = rutas["imagenes"]
        # Solo aquí creas las carpetas necesarias
        for carpeta in [self.carpeta_metadatos, self.carpeta_imagenes]:
            os.makedirs(carpeta, exist_ok=True)
    
    def procesar_directorio(self, ruta_directorio, limite):

        rutas_dicoms = obtener_lista_dicoms(ruta_directorio)
        ruta_csv_caracteriticas = self.rutas['caracteristicas']
        ruta_csv_fusion = self.rutas['fusionado']
        if os.path.exists(ruta_csv_caracteriticas):
            os.remove(ruta_csv_caracteriticas)

        for indice, ruta_dicom in enumerate(rutas_dicoms[:limite]):
            dicom = cargar_dicom(ruta_dicom)
            metadatos = extraer_metadatos(dicom)
            ruta_metadatos = guardar_metadatos_txt(metadatos, self.carpeta_metadatos, indice + 1)
            ruta_imagen = extraer_imagen_jpg(dicom, self.carpeta_imagenes, indice + 1)
            if ruta_imagen:
                vector = extraer_caracteristicas(ruta_imagen)
                guardar_caracteristicas_csv(vector, indice + 1, ruta_csv_caracteriticas)
                print(f"Características extraídas y guardadas para imagen {indice + 1}")

            ruta_csv_caracteriticas = self.rutas['caracteristicas']
            
            print(f"Procesado archivo {indice+1}:")
            print(f" - Metadatos en: {ruta_metadatos}")
            print(f" - Imagen en: {ruta_imagen}")
        ruta_csv_fusion = self.rutas['fusionado']
        fusionar_caracteristicas_metadatos(ruta_csv_caracteriticas,self.carpeta_metadatos,ruta_csv_fusion)
    
    