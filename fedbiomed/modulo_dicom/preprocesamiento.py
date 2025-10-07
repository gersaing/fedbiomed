import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler



def eliminar_columnas_irrelevantes(df):
    """
    Lee el CSV fusionado, elimina columnas irrelevantes.
    """
    columnas_eliminar = [
    "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "(0028,3003) LUT Explanation                     LO",
    "(0028,3006) LUT Data                            US", "BodyPartExamined", "BitsAllocated","BitsStored", "Columns",
    "HighBit", "ImagerPixelSpacing", "LossyImageCompression", "ManufacturerModelName",
    "Modality", "PatientSex", "PhotometricInterpretation", "PixelPaddingValue",
    "PixelRepresentation", "PresentationLUTShape", "RescaleIntercept", "RescaleType", 
    "Rows", "VOILUTSequence", "WindowCenterWidthExplanation","WindowWidth","Manufacturer","PresentationIntentType","Id",
    "(0008,0102) Coding Scheme Designator            SH"]
    df = df.drop(columns=columnas_eliminar, errors="ignore")
    return df

def codificar_dataset(df):
    """Realiza codificaciones específicas en el DataFrame."""
    mapeo_burned_in = {'NO': 0, 'YES': 1}

    # Codificaciones específicas
    if 'BurnedInAnnotation' in df.columns:
        df['BurnedInAnnotation'] = df['BurnedInAnnotation'].map(mapeo_burned_in)
    if 'PatientAge' in df.columns:
        df['PatientAge'] = df['PatientAge'].astype(str).str.extract(r'(\d+)').astype(float)

    #One-hot encoding
    columnas_one_hot = []
    if 'ViewPosition' in df.columns:
        columnas_one_hot.append('ViewPosition')
    if 'ImageLaterality' in df.columns:
        columnas_one_hot.append('ImageLaterality')
    if columnas_one_hot:
        df = pd.get_dummies(df, columns=columnas_one_hot, prefix=columnas_one_hot)

    # Limpieza de orientación con one-hot encoding
    if 'PatientOrientation' in df.columns:
        df['PatientOrientation'] = df['PatientOrientation'].astype(str).str.replace(r"[\[\]',]", '', regex=True).str.strip()
    # One-hot encoding

    return df

def normalizar_entrenamiento(df, ruta_escalador):
    """
    Normaliza el DataFrame de entrenamiento usando MinMaxScaler y guarda el objeto scaler.
    Retorna el DataFrame normalizado.
    """
    escalador = MinMaxScaler()
    df_entrenamiento_normalizado = pd.DataFrame(escalador.fit_transform(df), columns=df.columns).round(8)
    joblib.dump(escalador, ruta_escalador) 
    return df_entrenamiento_normalizado

def normalizar_validacion(df, ruta_escalador, salida):
    """
    Normaliza el DataFrame de validación usando el escalador previamente guardado y guarda el resultado.
    """
    escalador = joblib.load(ruta_escalador)
    df_validacion_normalizado = pd.DataFrame(escalador.transform(df), columns=df.columns).round(8)
    df_validacion_normalizado.to_csv(salida, index=False)
    print(f"[INFO] CSV de validación normalizado guardado en: {salida}")

def asignar_diagnostico(df, diagnostico, salida):
    """
    Asigna la columna 'Diagnostic' al DataFrame y lo guarda en la ruta indicada.
    """
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    df['Diagnostic'] = diagnostico
    df.to_csv(salida, index=False)
    print(f"[INFO] Asignación de diagnóstico completada y guardada en: {salida}")
    return df

def actualizar_dataset_entrenamiento(df_nuevo, csv_entrenamiento):
    """
    Actualiza el dataset de entrenamiento agregando los nuevos datos y barajando el orden.
    Evita duplicados exactos.
    """
    if os.path.exists(csv_entrenamiento):
        df_antiguo = pd.read_csv(csv_entrenamiento)
        df_actualizado = pd.concat([df_antiguo, df_nuevo], ignore_index=True)
        df_actualizado = df_actualizado.drop_duplicates()
        df_actualizado = df_actualizado.sample(frac=1, random_state=42).reset_index(drop=True)
        df_actualizado.to_csv(csv_entrenamiento, index=False)
        print("[INFO] Dataset de entrenamiento actualizado.")
    else:
        df_nuevo.to_csv(csv_entrenamiento, index=False)
        print("[INFO] Dataset de entrenamiento creado.")
        
def ajustar_dataset_validacion(df_validacion, csv_entrenamiento):
    """
    Ajusta el dataset de validación para que tenga las mismas columnas (orden y cantidad) que el entrenamiento.
    """
    df_entrenamiento = pd.read_csv(csv_entrenamiento)
    columnas_entrenamiento = df_entrenamiento.columns.tolist()

    for col in columnas_entrenamiento:
        if col not in df_validacion.columns:
            df_validacion[col] = 0

    df_validacion = df_validacion[columnas_entrenamiento]
    df_validacion = df_validacion.drop(columns=['Diagnostic'], errors='ignore')
    return df_validacion