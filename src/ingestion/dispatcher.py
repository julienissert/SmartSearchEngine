# ingestion/dispatcher.py

import os

from ingestion.csv_ingestion import load_csv
from ingestion.pdf_ingestion import load_pdf
from ingestion.image_ingestion import load_image
from ingestion.h5_ingestion import load_h5
from ingestion.txt_ingestion import load_txt


def dispatch_loader(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv"]:
        return load_csv(path)

    if ext in [".pdf"]:
        return load_pdf(path)

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return load_image(path)

    if ext in [".h5", ".hdf5"]:
        return load_h5(path)

    if ext in [".txt"]:
        return load_txt(path)

    raise ValueError(f"Aucun loader disponible pour l'extension {ext}")
