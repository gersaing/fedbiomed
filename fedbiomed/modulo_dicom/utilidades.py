import os,shutil
import csv
import re
from fedbiomed.modulo_dicom.excepciones import NodoNoEncontradoError

def obtener_lista_dicoms(ruta_directorio):
    """
    Retorna una lista con las rutas absolutas de todos los archivos DICOM (.dcm o .dicom)
    encontrados recursivamente en el directorio dado.
    """
    archivos_dicom = []
    for subdirectorio, _, archivos in os.walk(ruta_directorio):
        for archivo in archivos:
            if archivo.endswith(".dcm") or archivo.endswith(".dicom"):
                archivos_dicom.append(os.path.abspath(os.path.join(subdirectorio, archivo)))
    return archivos_dicom


def guardar_caracteristicas_csv(vector, indice, ruta_csv):
    """
    Guarda el vector de características junto con el índice en el archivo CSV indicado.
    Si el archivo no existe, escribe el encabezado.
    """
    existe = os.path.isfile(ruta_csv)
    with open(ruta_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not existe:
            encabezado = ['Id'] + [f'Feature_{i}' for i in range(len(vector))]
            writer.writerow(encabezado)
        writer.writerow([indice] + vector.tolist())

def leer_metadatos_txt(ruta_metadatos):
    metadatos = {}
    with open(ruta_metadatos, "r", encoding="utf-8") as f:
        for linea in f:
            if ':' in linea:
                clave, valor = linea.strip().split(":", 1)
                metadatos[clave.strip()] = valor.strip()
    return metadatos

import csv

def fusionar_caracteristicas_metadatos(ruta_carac_csv, carpeta_metadatos, ruta_salida_csv):
    """
    Fusiona el CSV de características con los metadatos por Id, y guarda el CSV combinado.
    """
    print("Fusionando características y metadatos...")
    # Lee características
    with open(ruta_carac_csv, newline='') as f:
        print("Leyendo CSV de características...")
        reader = list(csv.reader(f))
        encabezados_caracteristicas = reader[0]
        filas_caracteristicas = reader[1:]

    # Descubre todas las claves de metadatos
    todas_las_claves = set()
    registros_fusionados = []
    for fila in filas_caracteristicas:
        id_img = fila[0]
        ruta_metadatos = os.path.join(carpeta_metadatos, f"metadata_{id_img}.txt")
        if os.path.exists(ruta_metadatos):
            metadatos = leer_metadatos_txt(ruta_metadatos)
            todas_las_claves.update(metadatos.keys())
            registros_fusionados.append((fila, metadatos))
        else:
            # Si no hay metadatos para este id, puedes decidir omitirlo o poner vacío
            registros_fusionados.append((fila, {}))

    claves_ordenadas = sorted(todas_las_claves)

    # Escribe el CSV final
    os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
    with open(ruta_salida_csv, mode='w', newline='') as f_salida:
        escritor = csv.writer(f_salida)
        encabezado_final = encabezados_caracteristicas + claves_ordenadas
        escritor.writerow(encabezado_final)
        for fila_carac, metadatos in registros_fusionados:
            fila_meta = [metadatos.get(clave, "") for clave in claves_ordenadas]
            escritor.writerow(fila_carac + fila_meta)
    print(f"Dataset fusionado guardado en: {ruta_salida_csv}")

def limpiar_directorio(carpeta):
    if os.path.exists(carpeta):
        shutil.rmtree(carpeta)
        print(f"[INFO] Carpeta {carpeta} eliminada.")

def iter_rutas_imagenes(carpeta_imagenes):
    """
    Genera rutas .../imagen_<id>.jpg ordenadas por <id>.
    """
    prefijo="imagen_"
    ext=".jpg"
    entradas = [
        e for e in os.scandir(carpeta_imagenes)
        if e.is_file() and e.name.startswith(prefijo) and e.name.lower().endswith(ext)
    ]

    def clave_orden(e):
        nombre = e.name
        # extrae lo que hay entre prefijo y extensión
        id_str = nombre[len(prefijo):-len(ext)]
        # intenta numérico
        m = re.fullmatch(r"\d+", id_str)
        if m:
            return (0, int(id_str))        # orden numérico
        else:
            return (1, id_str.lower())     # fallback: orden lexicográfico

    entradas.sort(key=clave_orden)

    for e in entradas:
        yield e.path

def id_desde_ruta(ruta):
    prefijo="imagen_"
    ext=".jpg"
    nombre = os.path.basename(ruta)
    if nombre.startswith(prefijo) and nombre.lower().endswith(ext):
        return nombre[len(prefijo):-len(ext)]
    raise ValueError(f"Nombre no cumple {prefijo}<id>{ext}: {nombre}")

import os

def obtener_ruta_datos_nodo(nombre_nodo: str = "local") -> str:
    """
    Devuelve la ruta absoluta a data/<nombre_nodo>
    dentro del sandbox del nodo.
    """
    ruta = os.path.join(os.getcwd(), nombre_nodo)
    if not os.path.isdir(ruta):
        raise NodoNoEncontradoError(f"El nodo '{nombre_nodo}' no existe en el directorio actual.")
    return os.path.join(os.getcwd(), nombre_nodo, "data")

def obtener_ruta_salida_modulo(nombre_nodo: str = "local") -> str:
    """
    Devuelve la carpeta donde volcar los resultados:
    <nombre_nodo>/data/modulo_dicom_results
    """
    ruta_datos = obtener_ruta_datos_nodo(nombre_nodo)
    return os.path.join(ruta_datos, "resultados_modulo_dicom")

