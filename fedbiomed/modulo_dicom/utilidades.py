import os
import csv
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
    with open(ruta_salida_csv, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        encabezado_final = encabezados_caracteristicas + claves_ordenadas
        writer.writerow(encabezado_final)
        for fila_carac, metadatos in registros_fusionados:
            fila_meta = [metadatos.get(clave, "") for clave in claves_ordenadas]
            writer.writerow(fila_carac + fila_meta)
    print(f"Dataset fusionado guardado en: {ruta_salida_csv}")
