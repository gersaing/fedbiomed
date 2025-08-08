import pydicom
import numpy as np
from PIL import Image
import os
import os, json
def cargar_dicom(ruta):
    return pydicom.dcmread(ruta)


def extraer_metadatos(archivo_dicom):
    """Extrae los metadatos relevantes de un archivo DICOM."""
    return {elem.keyword: elem.value for elem in archivo_dicom if elem.keyword and elem.keyword != "PixelData"}

def guardar_metadatos_txt(metadatos, carpeta_salida, indice):
    """
    Guarda los metadatos extraídos en un archivo txt.
    """
    import csv
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_txt = os.path.join(carpeta_salida, f"metadata_{indice}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        for clave, valor in metadatos.items():
            f.write(f"{clave}: {valor}\n")
    return ruta_txt

def extraer_imagen_jpg(archivo_dicom, carpeta_salida, indice):
    """
    Extrae la imagen de un objeto DICOM y la guarda como archivo JPG en la carpeta indicada.
    Retorna la ruta del archivo JPG guardado.
    """
    if hasattr(archivo_dicom, "pixel_array"):
        pixel_array = archivo_dicom.pixel_array

        # Comprobar el rango de los píxeles antes de la normalización (opcional para debug)
        print(f"Imagen {indice}: Valor mínimo de píxel:", np.min(pixel_array))
        print(f"Imagen {indice}: Valor máximo de píxel:", np.max(pixel_array))

        # Obtener valores de ventana si existen
        window_center = archivo_dicom.get('WindowCenter', None)
        window_width = archivo_dicom.get('WindowWidth', None)

        if window_center is not None and window_width is not None:
            window_center = float(window_center)
            window_width = float(window_width)
            lower_bound = window_center - window_width // 2
            upper_bound = window_center + window_width // 2
            pixel_array = np.clip(pixel_array, lower_bound, upper_bound)

            # Normalizar a 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)
        else:
            # Si no hay ventana, asegúrate que la imagen quede en el rango 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)

        imagen = Image.fromarray(pixel_array).convert('L')
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta_imagen = os.path.join(carpeta_salida, f"{indice}.jpg")
        imagen.save(ruta_imagen)
        print(f"Imagen guardada en: {ruta_imagen}")
        return ruta_imagen
    else:
        print(f"El archivo DICOM {indice} no contiene datos de imagen.")
        return None
