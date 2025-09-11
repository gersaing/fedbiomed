from .utilidades import obtener_lista_dicoms,limpiar_directorio,iter_rutas_imagenes,id_desde_ruta
from .cargador import cargar_dicom, extraer_metadatos, guardar_metadatos_txt, extraer_imagen_jpg
from .extractor_caracteristicas import ExtractorCaracteristicas
from .utilidades import guardar_caracteristicas_csv, fusionar_caracteristicas_metadatos
import os,shutil

class PipelineDicom:
    def __init__(self, rutas):
        self.rutas = rutas
        self.carpeta_metadatos = rutas["metadatos"]
        self.carpeta_imagenes = rutas["imagenes"]
        # Solo aquí creas las carpetas necesarias
        for carpeta in [self.carpeta_metadatos, self.carpeta_imagenes]:
            limpiar_directorio(carpeta)  # Limpia antes de crear
        for carpeta in [self.carpeta_metadatos, self.carpeta_imagenes]:
            os.makedirs(carpeta, exist_ok=True)
    
    def procesar_directorio(self, ruta_directorio, limite):

        rutas_dicoms = obtener_lista_dicoms(ruta_directorio)
        ruta_csv_caracteriticas = self.rutas['caracteristicas']
        if os.path.exists(ruta_csv_caracteriticas):
            os.remove(ruta_csv_caracteriticas)

        for indice, ruta_dicom in enumerate(rutas_dicoms[:limite]):
            dicom = cargar_dicom(ruta_dicom)
            metadatos = extraer_metadatos(dicom)
            ruta_metadatos = guardar_metadatos_txt(metadatos, self.carpeta_metadatos, indice + 1)
            ruta_imagen = extraer_imagen_jpg(dicom, self.carpeta_imagenes, indice + 1)

            print(f"Procesado archivo {indice+1}:")
            print(f" - Metadatos en: {ruta_metadatos}")
            print(f" - Imagen en: {ruta_imagen}")

    def extraer_caracteristicas(self):
        """
        Extrae las características de una imagen DICOM y las guarda en un CSV.
        """
        extractor = ExtractorCaracteristicas(dispositivo="cpu",modo_transform="compat")
        for ruta in iter_rutas_imagenes(self.carpeta_imagenes):
            vector = extractor.extraer(ruta)
            indice = id_desde_ruta(ruta)
            guardar_caracteristicas_csv(vector, indice, self.rutas['caracteristicas'])
            print(f"Características extraídas y guardadas para imagen: {ruta}")
            
    def fusionar_caracteristicas_metadatos(self):
        """
        Fusiona las características extraídas con los metadatos y guarda el resultado en un CSV.
        """
        fusionar_caracteristicas_metadatos(self.rutas['caracteristicas'], self.carpeta_metadatos, self.rutas['fusionado'])
        print(f"Características y metadatos fusionados guardados en: {self.rutas['fusionado']}")