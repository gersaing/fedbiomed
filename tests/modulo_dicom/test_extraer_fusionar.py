from pathlib import Path
import csv
import os
import pytest
import numpy as np

from fedbiomed.modulo_dicom.pipeline_transformacion import PipelineDicom

def _crear_imagenes_dummy(carpeta: Path, ids=(1,2,3)):
    """Crea archivos imagen_<id>.jpg vacíos válidos para el pipeline (no se abren con PIL en esta prueba)."""
    from PIL import Image
    carpeta.mkdir(parents=True, exist_ok=True)
    for i in ids:
        Image = pytest.importorskip("PIL").Image
        im = Image.new("L", (32, 32), color=127)
        im.save(carpeta / f"imagen_{i}.jpg")

def test_extraer_caracteristicas_mockea_extractor(rutas_tmp, monkeypatch, tmp_path):
    """Evita dependencias pesadas de Torch/ResNet mockeando el extractor."""
    # 1) Instancia primero para que limpie directorios
    pipeline = PipelineDicom(rutas_tmp, ruta_directorio=str(tmp_path), limite=3)

    # 2) Crea las imágenes dummy dentro de rutas_tmp["imagenes"]
    _crear_imagenes_dummy(rutas_tmp["imagenes"], ids=(1, 2, 3))

    # 3) Mock del extractor
    class FakeExtractor:
        dim_salida = 4
        def __init__(self, *a, **k): pass
        def extraer(self, ruta_imagen: str):
            import os, numpy as np
            nombre = os.path.basename(ruta_imagen)
            base = int(nombre.replace("imagen_", "").replace(".jpg", ""))
            return np.array([base, base+1, base+2, base+3], dtype=np.float32)

    monkeypatch.setattr(
        "fedbiomed.modulo_dicom.pipeline_transformacion.ExtractorCaracteristicas",
        FakeExtractor
    )

    # 4) (Opcional) Asegura la ruta de salida
    pipeline.rutas.setdefault("caracteristicas", rutas_tmp["caracteristicas"])

    # 5) Ejecuta extracción
    pipeline.extraer_caracteristicas()

    # 6) Verifica salida
    assert Path(rutas_tmp["caracteristicas"]).exists(), "No se creó 'caracteristicas.csv'."


def test_fusionar_caracteristicas_metadatos(rutas_tmp, tmp_path):
    """Fusiona características con metadatos .txt y valida columnas/valores."""
    # 0) Instancia primero para que haga su limpieza de carpetas
    pipeline = PipelineDicom(rutas_tmp, ruta_directorio=str(tmp_path), limite=3)

    # 1) Crea un CSV de características mínimo con Id 1..3
    with open(rutas_tmp["caracteristicas"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Id","Feature_0","Feature_1"])
        w.writerow([1, 0.1, 0.2])
        w.writerow([2, 0.3, 0.4])
        w.writerow([3, 0.5, 0.6])

    # 2) Ahora sí, crea los metadatos después de la limpieza del __init__
    (Path(rutas_tmp["metadatos"]) / "metadata_1.txt").write_text(
        "PatientID: P001\nEdad: 55\n", encoding="utf-8"
    )
    (Path(rutas_tmp["metadatos"]) / "metadata_2.txt").write_text(
        "PatientID: P002\nEdad: 61\n", encoding="utf-8"
    )
    # El 3 lo dejamos sin metadatos para probar fallback

    # 3) Ejecuta la fusión
    pipeline.fusionar_caracteristicas_metadatos()

    # 4) Verifica salida
    out = Path(rutas_tmp["fusionado"])
    assert out.exists(), "No se creó el CSV fusionado."

    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    body = rows[1:]

    assert header[:3] == ["Id","Feature_0","Feature_1"]
    # orden puede variar
    assert header[3:] == ["Edad","PatientID"] or header[3:] == ["PatientID","Edad"]

    fila1 = next(r for r in body if r[0] == "1")
    assert set(zip(header[3:], fila1[3:])) >= {("PatientID","P001"), ("Edad","55")}
    fila3 = next(r for r in body if r[0] == "3")
    # sin metadata_3.txt debería haber campos vacíos
    assert "" in fila3[3:]

