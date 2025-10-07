from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytest


# --- Semillas globales para reproducibilidad ---
@pytest.fixture(autouse=True)
def _fix_seed():
    random.seed(42)
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except Exception:
        pass


# --- Rutas temporales esperadas por tu pipeline ---
@pytest.fixture
def rutas_tmp(tmp_path: Path):
    """Replica las claves reales del módulo (configuración.py)."""
    rutas = {
        "imagenes":        tmp_path / "imagenes",
        "metadatos":       tmp_path / "metadatos",
        "caracteristicas": tmp_path / "caracteristicas.csv",
        "fusionado":       tmp_path / "dataset_fusionado.csv",
    }
    rutas["imagenes"].mkdir(parents=True, exist_ok=True)
    rutas["metadatos"].mkdir(parents=True, exist_ok=True)
    return rutas


# --- Helper para crear DICOMs válidos (con file meta + TransferSyntax) ---
def crear_dicoms_minimos(carpeta: Path, n: int = 10) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        PYDICOM_IMPLEMENTATION_UID,
    )
    import datetime

    carpeta.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        fname = carpeta / f"case_{i+1:03d}.dcm"

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = f"1.2.826.0.1.3680043.8.498.{i+1}"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = FileDataset(str(fname), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        ds.PatientID = f"P{i+1:03d}"
        ds.StudyInstanceUID = f"1.2.826.0.1.3680043.2.112.{i+1}"
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = 16, 16
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        pixels = np.arange(ds.Rows * ds.Columns, dtype=np.uint16).reshape(ds.Rows, ds.Columns) + (i + 1)
        ds.PixelData = pixels.tobytes()

        ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
        ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
        ds.save_as(fname, write_like_original=False)


# --- Fixture que expone la carpeta con DICOMs sintéticos ---
@pytest.fixture
def dicoms_dir(tmp_path: Path):
    d = tmp_path / "dicoms"
    crear_dicoms_minimos(d, n=5)
    return d


# --- Helpers para generar DICOMs corruptos ---

def crear_dicoms_corruptos(carpeta: Path):
    """
    Crea varios archivos .dcm corruptos en 'carpeta' con diferentes fallos.
    Devuelve un dict con rutas por tipo.
    """
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        PYDICOM_IMPLEMENTATION_UID,
        generate_uid,
    )
    import datetime
    import os

    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {}

    # Base correcto para partir de algo válido
    def _base_valido(path: Path):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = fm.MediaStorageSOPClassUID
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.Rows, ds.Columns = 16, 16
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        pixels = np.arange(ds.Rows * ds.Columns, dtype=np.uint16).reshape(ds.Rows, ds.Columns)
        ds.PixelData = pixels.tobytes()
        ds.PatientID = "PX"
        ds.StudyInstanceUID = generate_uid()
        ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
        ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
        ds.save_as(path, enforce_file_format=True, little_endian=True, implicit_vr=False)

    # 1) Sin PixelData
    p1 = carpeta / "bad_missing_pixeldata.dcm"
    _base_valido(p1)
    delattr(pydicom.dcmread(str(p1)), "PixelData")  # tocar en disco es más simple:
    # Reescribimos sin PixelData:
    ds = pydicom.dcmread(str(p1))
    if hasattr(ds, "PixelData"):
        del ds.PixelData
    ds.save_as(p1, enforce_file_format=True, little_endian=True, implicit_vr=False)
    rutas["missing_pixeldata"] = p1

    # 2) PixelData truncado (menos bytes de los requeridos)
    p2 = carpeta / "bad_truncated_pixeldata.dcm"
    _base_valido(p2)
    ds = pydicom.dcmread(str(p2))
    ds.PixelData = ds.PixelData[:10]  # truncar
    ds.save_as(p2, enforce_file_format=True, little_endian=True, implicit_vr=False)
    rutas["truncated_pixeldata"] = p2

    # 3) Inconsistencia filas/columnas vs PixelData
    p3 = carpeta / "bad_wrong_shape.dcm"
    _base_valido(p3)
    ds = pydicom.dcmread(str(p3))
    ds.Rows, ds.Columns = 32, 32  # pero pixeldata sigue siendo de 16x16
    ds.save_as(p3, enforce_file_format=True, little_endian=True, implicit_vr=False)
    rutas["wrong_shape"] = p3

    # 4) File meta roto (TransferSyntax inválido)
    p4 = carpeta / "bad_transfer_syntax.dcm"
    _base_valido(p4)
    ds = pydicom.dcmread(str(p4))
    ds.file_meta.TransferSyntaxUID = "1.2.3.4.5.6.7.8.9.0"  # no estándar
    ds.save_as(p4, enforce_file_format=True, little_endian=True, implicit_vr=False)
    rutas["bad_ts"] = p4

    # 5) No DICOM camuflado (bytes aleatorios con extensión .dcm)
    p5 = carpeta / "not_a_dicom.dcm"
    with open(p5, "wb") as f:
        f.write(os.urandom(256))
    rutas["not_dicom"] = p5

    return rutas


@pytest.fixture
def dicoms_corruptos(tmp_path: Path):
    carpeta = tmp_path / "dicoms_corruptos"
    rutas = crear_dicoms_corruptos(carpeta)
    return rutas
