# fedbiomed/modulo_dicom/balance/runner.py
from __future__ import annotations
import json
import pathlib as _p
from datetime import datetime
import pandas as pd

import os
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

from .estrategias import smote, smote_enn,random_under

class PipelinePreprocesamiento:
    
    def __init__(self):
        self.estrategias = {
            "smote": smote.aplicar,
            "smote-enn": smote_enn.aplicar,
            "random-under": random_under.aplicar,
        }
    def balanceo(self,ruta_csv: str, metodo: str, objetivo: str, col_id: str | None = None,
                proporcion: float = 1.0, semilla: int = 42, sufijo_salida: str = "_bal_",
                escribir_reporte: bool = True) -> _p.Path:
        ruta = _p.Path(ruta_csv)
        if not ruta.exists():
            raise FileNotFoundError(f"CSV no encontrado: {ruta}")

        df = pd.read_csv(ruta)
        if objetivo not in df.columns:
            raise ValueError(f"Columna objetivo '{objetivo}' no existe en el CSV")

        conteos_antes = df[objetivo].value_counts().to_dict()

        if metodo not in self.estrategias:
            raise ValueError(f"Método no soportado: {metodo}. Soportados: {list(self.estrategias)}")

        df_bal = self.estrategias[metodo](df=df, objetivo=objetivo, proporcion=proporcion, semilla=semilla)

        conteos_despues = df_bal[objetivo].value_counts().to_dict()

        salida = ruta.with_name(f"{ruta.stem}{sufijo_salida}{metodo}{ruta.suffix}")
        df_bal.to_csv(salida, index=False)

        if escribir_reporte:
            reporte = {
                "input_csv": str(ruta),
                "output_csv": str(salida),
                "method": metodo,
                "target": objetivo,
                "ratio": proporcion,
                "seed": semilla,
                "before": conteos_antes,
                "after": conteos_despues,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(salida.with_suffix(".balance_report.json"), "w", encoding="utf-8") as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)

        print(f"Balanceo '{metodo}' aplicado. Salida: {salida}")
        return salida

    def balanceo_automatico(self,df, objetivo: str = "Diagnostic", 
                proporcion: float = 1.0, semilla: int = 42, metodo: str = "smote") -> pd.DataFrame:
        if objetivo not in df.columns:
            raise ValueError(f"Columna objetivo '{objetivo}' no existe en el DataFrame")

        if metodo not in self.estrategias:
            raise ValueError(f"Método no soportado: {metodo}. Soportados: {list(self.estrategias)}")
        conteos = df[objetivo].value_counts()
        min_count = int(conteos.min())

        if (min_count < 6) and (metodo in {"smote", "smote-enn"}):
            print(f"[WARNING] Clases con muy pocos ejemplos ({min_count}). Cambiando método a 'random-under'.")
            metodo = "random-under"

        df_bal = self.estrategias[metodo](df=df, objetivo=objetivo, proporcion=proporcion, semilla=semilla)

        return df_bal, metodo
    @staticmethod 
    def calcular_balance(df, objetivo: str = "Diagnostic") -> float:
        """
        Calcula el porcentaje de balanceo entre la clase minoritaria y mayoritaria.
        """
        if objetivo not in df.columns:
            raise ValueError(f"La columna objetivo '{objetivo}' no existe en el DataFrame")

        conteos = df[objetivo].value_counts().to_dict()
        min_clase = min(conteos.values())
        total = sum(conteos.values())

        porcentaje = (min_clase / total) * 100 if total > 0 else 0

        return porcentaje

    @staticmethod
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
        print("[INFO] Columnas irrelevantes eliminadas.")
        return df
    @staticmethod
    def codificar_dataset_(df):
       
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
    @staticmethod
    def codificar_dataset(df):
        """Codificación temporal compatible con modelo entrenado con valores numéricos directos."""
        print("[INFO] Codificando dataset...")
        df.to_csv("prueba.csv", index=False)
        # Codificación binaria para BurnedInAnnotation
        mapeo_burned_in = {'NO': 0.0, 'YES': 1.0}
        if 'BurnedInAnnotation' in df.columns:
            df['BurnedInAnnotation'] = df['BurnedInAnnotation'].map(mapeo_burned_in)

        # Extracción numérica de edad
        if 'PatientAge' in df.columns:
            df['PatientAge'] = df['PatientAge'].astype(str).str.extract(r'(\d+)').astype(float)

        # Codificación directa para ViewPosition
        mapeo_view_position = {'CC': 0.0, 'MLO': 1.0}

        if 'ViewPosition' in df.columns:
            df['ViewPosition'] = df['ViewPosition'].map(mapeo_view_position)

        # Codificación directa para ImageLaterality
        mapeo_laterality = {'L': 0.0, 'R': 1.0}
        if 'ImageLaterality' in df.columns:
            df['ImageLaterality'] = df['ImageLaterality'].map(mapeo_laterality)

        # Limpieza de orientación (sin codificación)
        if 'PatientOrientation' in df.columns:
            df['PatientOrientation'] = df['PatientOrientation'].astype(str).str.replace(r"[\[\]',]", '', regex=True).str.strip()

        return df

    @staticmethod
    def normalizar_entrenamiento(df, ruta_escalador):
        """
        Normaliza el DataFrame de entrenamiento usando MinMaxScaler y guarda el objeto scaler.
        Retorna el DataFrame normalizado.
        """
        escalador = MinMaxScaler()
        df_entrenamiento_normalizado = pd.DataFrame(escalador.fit_transform(df), columns=df.columns).round(8)
        joblib.dump(escalador, ruta_escalador) 
        return df_entrenamiento_normalizado
    @staticmethod
    def normalizar_validacion(df, ruta_escalador, salida):
        """
        Normaliza el DataFrame de validación usando el escalador previamente guardado y guarda el resultado.
        """
        escalador = joblib.load(ruta_escalador)
        df_validacion_normalizado = pd.DataFrame(escalador.transform(df), columns=df.columns).round(8)
        df_validacion_normalizado.to_csv(salida, index=False)
        print(f"[INFO] CSV de validación normalizado guardado en: {salida}")
    @staticmethod
    def asignar_diagnostico(df, diagnostico, salida):
        """
        Asigna la columna 'Diagnostic' al DataFrame y lo guarda en la ruta indicada.
        """
        os.makedirs(os.path.dirname(salida), exist_ok=True)
        df['Diagnostic'] = diagnostico
        df.to_csv(salida, index=False)
        print(f"[INFO] Asignación de diagnóstico completada y guardada en: {salida}")
        return df
    @staticmethod
    def actualizar_dataset_entrenamiento(df_nuevo, csv_entrenamiento):
        print(f"[INFO] Actualizando dataset de entrenamiento en:")
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
    @staticmethod   
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
        print("[INFO] Dataset de validación ajustado a las columnas del entrenamiento.")
        return df_validacion