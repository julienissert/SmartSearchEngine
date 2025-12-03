#ingestion/folder_scanner.py
import os

VALID_EXT = [".csv", ".pdf", ".png", ".jpg", ".jpeg", ".h5"]

def scan_folder(folder):
    files = []

    for root, _, filenames in os.walk(folder):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_EXT:
                files.append(os.path.join(root, name))

    return files
