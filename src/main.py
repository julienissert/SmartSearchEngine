import os
import json
import config
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader 
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata, get_all_metadata
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label, analyze_dataset_structure

METADATA_FILE = "metadata_db.json"

def process_document(doc, valid_labels):
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    
    # 1. Détection du Domaine (Food vs Medical)
    domain, domain_scores = detect_domain(text=str(content), pil_image=image)
    
    # 2. Détection du Label
    # Priorité absolue au label suggéré par le loader (ex: colonne "Item" d'un CSV)
    if doc.get("suggested_label"):
        label = doc["suggested_label"]
    else:
        # Sinon, détection standard via dossier, texte ou vision
        label = detect_label(
            filepath=source_path, 
            text=str(content), 
            image=image, 
            known_labels=valid_labels
        )
    
    # 3. Vectorisation (Embedding 512 dims)
    vector = embed_image(image) if image else embed_text(str(content))

    # 4. Stockage FAISS + Métadonnées
    if vector is not None:
        add_to_index(domain, vector)

    store_metadata({
        "source": source_path,
        "type": doc["type"],
        "domain": domain,
        "label": label,
        "domain_scores": domain_scores,
        "snippet": str(content)[:200] if content else ""
    })
    
    print(f"Traité: {os.path.basename(source_path)} -> Label: {label}")

def save_metadata_to_disk():
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(get_all_metadata(), f, indent=4, ensure_ascii=False)
    print(f"Sauvegarde terminée dans {METADATA_FILE}")

def main():
    if not os.path.exists(config.DATASET_DIR):
        print(f"Erreur: Dossier {config.DATASET_DIR} introuvable.")
        return

    # Phase 1 : Analyse intelligente de la structure du dataset
    valid_labels = analyze_dataset_structure(config.DATASET_DIR)

    print("Début de l'ingestion des fichiers...")
    files = scan_folder(config.DATASET_DIR)

    # Phase 2 : Traitement fichier par fichier
    for f in files:
        try:
            docs = dispatch_loader(f)
            for d in docs:
                process_document(d, valid_labels)
        except Exception as e:
            print(f"Erreur sur {f}: {e}")

    save_metadata_to_disk()

if __name__ == "__main__":
    main()