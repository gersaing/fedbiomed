import pytest
from pathlib import Path

from fedbiomed.modulo_dicom.pipeline_transformacion import PipelineDicom

@pytest.mark.slow
def test_procesar_directorio_genera_imagenes_y_metadatos(dicoms_dir, rutas_tmp):
    """Verifica que desde DICOM se creen JPG y TXT de metadatos."""
    pipeline = PipelineDicom(rutas_tmp, ruta_directorio=str(dicoms_dir), limite=3)
    pipeline.procesar_directorio()

    # Imágenes .jpg creadas
    imgs = list(Path(rutas_tmp["imagenes"]).glob("imagen_*.jpg"))
    assert len(imgs) >= 1, "No se generaron imágenes JPG en la carpeta 'imagenes'."

    # Metadatos .txt creados
    metas = list(Path(rutas_tmp["metadatos"]).glob("metadata_*.txt"))
    assert len(metas) >= 1, "No se generaron metadatos TXT en la carpeta 'metadatos'."

    # Aún no debe existir el CSV de características (lo crea la siguiente etapa)
    assert not Path(rutas_tmp["caracteristicas"]).exists()
