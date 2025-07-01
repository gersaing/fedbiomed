import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib 
def limpiarCsv():
    COMPLEMENTO = "ResultadoDICOM/fusionado/dataset_fusionado.csv"
    base_dir = os.path.dirname(__file__)
    ruta = os.path.join(base_dir,COMPLEMENTO )
    df = pd.read_csv(ruta)
    
    columns_to_drop = [
    "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "(0028,3003) LUT Explanation                     LO",
    "(0028,3006) LUT Data                            US", "BodyPartExamined", "BitsAllocated","BitsStored", "Columns",
    "HighBit", "ImagerPixelSpacing", "LossyImageCompression", "ManufacturerModelName",
    "Modality", "PatientSex", "PhotometricInterpretation", "PixelPaddingValue",
    "PixelRepresentation", "PresentationLUTShape", "RescaleIntercept", "RescaleType", 
    "Rows", "VOILUTSequence", "WindowCenterWidthExplanation","WindowWidth","Manufacturer","PresentationIntentType","Id"
    ]

    # Diccionario de mapeo para la codificación ordinal de Code Meaning
    ViewPosition = {
        'CC': 0,  
        'MLO': 1   
    }

    ImageLaterality_mapping ={
        'L':0,
        'R':1
    }

    # Eliminar columnas no necesarias
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Se realiza el mapeo
    if 'ViewPosition' in df.columns:
        df["ViewPosition"] = df["ViewPosition"].map(ViewPosition)
    if 'ImageLaterality' in df.columns:
        df["ImageLaterality"]= df["ImageLaterality"].map(ImageLaterality_mapping)
    # codificacion de BurnedInAnnotation 1-0
    if 'BurnedInAnnotation' in df.columns:
        df["BurnedInAnnotation"] = df["BurnedInAnnotation"].map({"NO": 0, "YES": 1})
    # limpiar edad quitar caracteres especiales
    if 'PatientAge' in df.columns:
        df["PatientAge"] = df["PatientAge"].str.extract(r'(\d+)').astype(int)
    # Se aplica One-Hot a la columna PatientOrientation
    if 'PatientOrientation' in df.columns:
        df["PatientOrientation"] = df["PatientOrientation"].astype(str).str.replace("[", "").str.replace("]", "").str.replace("'", "").str.replace(",", "").str.strip()
        df = pd.get_dummies(df, columns=["PatientOrientation","PhotometricInterpretation", "PixelIntensityRelationship","DetectorType"])  # One-Hot Encoding    
    return df


def codificacion_dinamica(df):
    print("")

def normalizar_entrenamiento(df,path_scaler='scaler.pkl'):
    scaler = MinMaxScaler()
    df_normalizado = pd.DataFrame(scaler.fit_transform(df), columns=df.columns).round(8)
    joblib.dump(scaler, path_scaler) 
    
    return df_normalizado

def normalizar_validacion(df,paht_scaller = 'scaler.pkl'):
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    output_csv_test = os.path.join(project_root, 'breast_cancer_test.csv')

    scaler = joblib.load(paht_scaller)
    df_normalizado = pd.DataFrame(scaler.transform(df),columns=df.columns).round(8)
    df_normalizado.to_csv(output_csv_test, index=False)
    print("CSV norm guardado ")

def asignarDiagnostico(df,diagnostico):
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    output_csv_dir = os.path.join(project_root, "ResultadoDICOM","nuevosDatos")
    os.makedirs(output_csv_dir, exist_ok=True)
    output_csv = os.path.join(output_csv_dir, 'breast_cancer_n.csv')
    df['Diagnostic'] = diagnostico 
    df.to_csv(output_csv, index=False)
    print('Asignación completada')
    return df

def actualizarDatos(df_nuevo):
    project_root = os.path.dirname(os.path.abspath(__file__))
    csv_antiguo = os.path.join(project_root,"breast_cancer.csv")

    if os.path.exists(csv_antiguo):
        df_antiguo = pd.read_csv(csv_antiguo)
        df_actualizado = pd.concat([df_antiguo,df_nuevo], ignore_index=True)
        df_actualizado = df_actualizado.drop_duplicates()
        df_actualizado = df_actualizado.sample(frac=1).reset_index(drop=True)
        df_actualizado.to_csv(csv_antiguo, index=False)
        print("CSV norm actualizado")
    else:
        df_nuevo.to_csv(csv_antiguo, index=False)
        print("CSV norm guardado ")

def ajustarDataset(df_validacion):

    project_root = os.path.dirname(os.path.abspath(__file__))
    csv_entrenamiento = os.path.join(project_root,"breast_cancer.csv")
    df_entrenamiento = pd.read_csv(csv_entrenamiento)

    columnas_entrenamiento = df_entrenamiento.columns.tolist()

    for col in columnas_entrenamiento:
        if col not in df_validacion.columns:
            df_validacion[col] = 0

    df_validacion = df_validacion[columnas_entrenamiento]
    df_validacion = df_validacion.drop(columns=['Diagnostic'])
    return df_validacion


def main():

    diagnostico = input('Asignacion de diagnostico Seleccione 0 (no cancer ) o 1 (cancer): ').strip()
    if diagnostico == '0' or diagnostico == '1':
        df = limpiarCsv()
        df = normalizar_entrenamiento(df)
        df = asignarDiagnostico(df,int(diagnostico))
        actualizarDatos(df)
    else:
        df = limpiarCsv()
        df = ajustarDataset(df)
        df = normalizar_validacion(df)

if __name__ == "__main__":
    main()