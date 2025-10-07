# fedbiomed/tests/modulo_dicom/test_dicoms_corruptos.py
import pytest
from pathlib import Path
from shutil import copyfile
from fedbiomed.modulo_dicom.pipeline_transformacion import PipelineDicom

@pytest.mark.slow
def test_procesar_directorio_con_dicoms_corruptos(dicoms_corruptos, rutas_tmp, capsys, tmp_path, dicoms_dir):
    """
    Mezcla DICOMs válidos (de la fixture dicoms_dir) con corruptos y verifica el manejo.
    """
    mezcla_dir = tmp_path / "mezcla"
    mezcla_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copia 2 válidos desde la fixture dicoms_dir
    validos = sorted(Path(dicoms_dir).glob("*.dcm"))
    assert len(validos) >= 2, "La fixture dicoms_dir debería generar al menos 2 DICOM válidos."
    for i, src in enumerate(validos[:2], start=1):
        copyfile(src, mezcla_dir / f"valido_{i:02d}.dcm")

    # 2) Copia 3 corruptos
    for i, src in enumerate(list(dicoms_corruptos.values())[:3], start=1):
        copyfile(src, mezcla_dir / f"corrupt_{i:02d}.dcm")

    # 3) Ejecuta pipeline
    pipeline = PipelineDicom(rutas_tmp, ruta_directorio=str(mezcla_dir), limite=999)
    pipeline.procesar_directorio()

    # 4) Verificaciones
    imgs = list(Path(rutas_tmp["imagenes"]).glob("imagen_*.*"))
    assert 1 <= len(imgs) <= 2, f"Imágenes solo para válidos; hubo {len(imgs)}."
    metas = list(Path(rutas_tmp["metadatos"]).glob("metadata_*.txt"))
    assert len(metas) >= 2, f"Se esperaban >=2 metadatos (válidos); hubo {len(metas)}."

    out = capsys.readouterr().out.lower()
    assert ("no contiene datos de imagen" in out) or ("error" in out) or ("excepción" in out)
