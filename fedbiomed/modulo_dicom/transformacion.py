import pydicom
import numpy as np
import re
from PIL import Image
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import csv
from torchvision.models import resnet18, ResNet18_Weights
import argparse

# Cargar el modelo preentrenado correctamente
resnet18_model = resnet18(weights=ResNet18_Weights.DEFAULT)
resnet18_model.eval()  # Establecer en modo evaluación

# Extraer características quitando la última capa FC
extractorCaracteristicas = torch.nn.Sequential(*list(resnet18_model.children())[:-1])

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ajustar tamaño para ResNet-18
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización basada en ImageNet
])

#Extrae características de una imagen usando ResNet-18
def extraccionCaracteristicas(image_path):
    image = Image.open(image_path).convert("RGB")  # Convertir a RGB para compatibilidad con el modelo
    image = transform(image).unsqueeze(0)  # Añadir batch dimension
    with torch.no_grad():
        caracteristicas = extractorCaracteristicas(image)
    return caracteristicas.squeeze().numpy()  # Convertir a numpy

def extraerDatosYMedatados(file_path, output_meta, output_img,output_csv, index):
    # Cargar el archivo DICOM
    fileDicom = pydicom.dcmread(file_path)

    # Extraer metadatos sin pixeles
    metaData = {elem.keyword: elem.value for elem in fileDicom if elem.keyword and elem.keyword != "PixelData"}

    # Guardar Metadatos en un  txt
    saveMetadata = os.path.join(output_meta, f"metadata_{index}.txt")
    with open(saveMetadata, "w") as f:
        for key, value in metaData.items():
            f.write(f"{key}: {value}\n")
    #print(f"Metadatos guardados en: {saveMetadata}")
    extraerImagenJPG(fileDicom, output_img, output_csv, index)

def inicializarCSV(output_csv, vectorCaracteristicas):
    # Crea un nuevo archivo CSV, sobrescribe el anterior e incluye los encabezados
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Id'] + [f'Feature_{i}' for i in range(len(vectorCaracteristicas))]
        writer.writerow(headers)
    print(f"Archivo CSV inicializado: {output_csv}")

def extraerImagenJPG(fileDicom, output_img, output_csv, index):
    # Extraer datos de imagen
    if hasattr(fileDicom, "pixel_array"):
        pixel_array = fileDicom.pixel_array

        # Comprobar el rango de los píxeles antes de la normalización
        print(f"Imagen {index}: Valor mínimo de píxel:", np.min(pixel_array))
        print(f"Imagen {index}: Valor máximo de píxel:", np.max(pixel_array))

        # Obtener valores de la ventana
        window_center = fileDicom.get('WindowCenter', None)
        window_width = fileDicom.get('WindowWidth', None)

        if window_center and window_width:
            # Ajustar la ventana
            window_center = float(window_center)
            window_width = float(window_width)

            # Aplicar la ventana
            lower_bound = window_center - window_width // 2
            upper_bound = window_center + window_width // 2
            pixel_array = np.clip(pixel_array, lower_bound, upper_bound)

            # Normalizar a 0-255
            pixel_array = pixel_array - np.min(pixel_array)
            pixel_array = pixel_array / np.max(pixel_array)
            pixel_array = (pixel_array * 255).astype(np.uint8)

        image = Image.fromarray(pixel_array).convert('L')  # 'L' fuerza la escala de grises

        # Guardar la imagen como PNG
        image_file = os.path.join(output_img, f"{index}.jpg")
        image.save(image_file)
         #Extraer características con ResNet-18
        vectorCaracteristicas = extraccionCaracteristicas(image_file)
        guardarCaracteristicasCsv(output_csv, vectorCaracteristicas,index)
        
    else:
        print(f"El archivo DICOM {index} no contiene datos de imagen.")

def guardarCaracteristicasCsv(output_csv, vectorCaracteristicas,index):
    if index == 1:
            inicializarCSV(output_csv, vectorCaracteristicas)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        writer.writerow([index] + vectorCaracteristicas.tolist())

def leerMetadatosPorIndice(metadata_dir, index):
    #Lee los archivos metadata.txt y lo convierte a un diccionario
    metadata_path = os.path.join(metadata_dir, f"metadata_{index}.txt")
    if not os.path.exists(metadata_path):
        print(f"No se encontró {metadata_path}. Se omitirá esta fila.")
        return None

    dic = {}
    with open(metadata_path, "r") as f:
        for line in f:
            if ':' in line:
                clave, valor = line.strip().split(":", 1)
                dic[clave.strip()] = valor.strip()
    return dic

def fusionarCaracteristicasMetadatos(caracteristicas_csv, metadata_dir, salida_csv):
    # Leer las características
    with open(caracteristicas_csv, newline='') as f:
        reader = list(csv.reader(f))
        encabezados_caracteristicas = reader[0]
        filas_caracteristicas = reader[1:]

    registros_fusionados = []
    todas_las_claves = set()

    for fila in filas_caracteristicas:
        nombre_imagen = fila[0]

        index = int(nombre_imagen)
        metadatos = leerMetadatosPorIndice(metadata_dir, index)
        if metadatos is None:
            continue

        todas_las_claves.update(metadatos.keys())
        registros_fusionados.append((fila, metadatos))

    claves_ordenadas = sorted(todas_las_claves)

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(salida_csv), exist_ok=True)

    # Escribir el CSV final
    with open(salida_csv, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        encabezado_final = encabezados_caracteristicas + claves_ordenadas
        writer.writerow(encabezado_final)

        for fila_carac, metadatos in registros_fusionados:
            fila_meta = [metadatos.get(clave, "") for clave in claves_ordenadas]
            writer.writerow(fila_carac + fila_meta)

    print(f"Dataset fusionado guardado en: {salida_csv}")

def main(dcm_path, num_dicoms):

    
    project_root = os.path.dirname(os.path.abspath(__file__))  # 
    output_metadata = os.path.join(project_root, "ResultadoDICOM", "metadata")
    output_images = os.path.join(project_root, "ResultadoDICOM", "images")
    output_csv_dir = os.path.join(project_root, "ResultadoDICOM", "caracteristicas")
    output_csv = os.path.join(output_csv_dir, "caracteristicas.csv")
    
    
    # Crear  directorios si no existen
    os.makedirs(output_metadata, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    dicom_directory = dcm_path
    limite = num_dicoms

    dicom_files = []
    for subdir, _, files in os.walk(dicom_directory):  # Recorrer subdirectorios
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.abspath(os.path.join(subdir, file)))

    # Procesar n archivos que deseemos, para este caso 5 a modo de prueba
    for index, dicom_file in enumerate(dicom_files[:limite]):
        dicom_file_path = dicom_file
        extraerDatosYMedatados(dicom_file_path, output_metadata, output_images, output_csv, index + 1)

    caracteristicas_csv = os.path.join(project_root, "ResultadoDICOM", "caracteristicas", "caracteristicas.csv")
    metadata_dir = os.path.join(project_root, "ResultadoDICOM", "metadata")
    salida_csv = os.path.join(project_root, "ResultadoDICOM", "fusionado", "dataset_fusionado.csv")

    fusionarCaracteristicasMetadatos(caracteristicas_csv, metadata_dir, salida_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de aprendizaje federado")
    parser.add_argument("--dcm_path",type=str,required=True, help="Ruta del dataset")
    parser.add_argument("--num_dicoms",type=int,required=True,help="Número de archivos dicom")

    args = parser.parse_args()
    main(args.dcm_path,args.num_dicoms)
    
