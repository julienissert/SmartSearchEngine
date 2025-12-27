# src/utils/visualizer.py
import sys
import os
import hashlib
from pathlib import Path

# Configuration des chemins pour les imports
# On s'assure que le dossier 'src' est dans le chemin de recherche
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

import indexing.faiss_index as faiss_idx
import indexing.metadata_index as meta_idx
import config

def get_pastel_color(name):
    """Génère une couleur pastel unique et stable à partir d'une chaîne de caractères."""
    hash_digest = hashlib.md5(name.encode()).digest()
    # Plage 150-255 pour garantir des tons clairs/pastels sur fond noir
    r = (hash_digest[0] % 105) + 150
    g = (hash_digest[1] % 105) + 150
    b = (hash_digest[2] % 105) + 150
    return f"rgb({r}, {g}, {b})"

def generate_immersive_3d(output_html="space_explorer.html"):
    """
    Génère une visualisation 3D immersive sans axes ni grilles.
    Sauvegarde le résultat dans computed-data/visualizations/.
    """
    
    # --- 1. GESTION DU RÉPERTOIRE DE SORTIE ---
    viz_dir = config.COMPUTED_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / output_html

    # --- 2. CHARGEMENT DES DONNÉES ---
    meta_idx.load_metadata_from_disk()
    faiss_idx.load_all_indexes()
    
    current_indexes = faiss_idx.indexes
    current_metadata = meta_idx.metadata_stores
    data_points = []
    
    # Mapping dynamique des couleurs
    domain_colors = {dom: get_pastel_color(dom) for dom in config.TARGET_DOMAINS}

    print(f"Extraction des vecteurs pour l'immersion : {config.TARGET_DOMAINS}...")
    
    for domain in config.TARGET_DOMAINS:
        if domain not in current_indexes or domain not in current_metadata:
            continue
            
        base_index = current_indexes[domain].index if hasattr(current_indexes[domain], 'index') else current_indexes[domain]
        ntotal = base_index.ntotal
        if ntotal == 0: continue
        
        vectors = base_index.reconstruct_n(0, ntotal)
        meta_list = current_metadata[domain]
        
        for i in range(ntotal):
            entry = meta_list[i] if i < len(meta_list) else {}
            data_points.append({
                "vector": vectors[i],
                "domain": domain.capitalize(), 
                "label": entry.get("label", "N/A"),
                "source": os.path.basename(entry.get("source", "Inconnu")),
                "score": entry.get("domain_score", 0.0)
            })

    if not data_points:
        print("Aucune donnée disponible.")
        return

    # --- 3. RÉDUCTION DE DIMENSION (PCA) ---
    X = np.array([p["vector"] for p in data_points])
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    # --- 4. CONSTRUCTION DU GRAPHIQUE ---
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        color=[p["domain"] for p in data_points],
        hover_name=[f"Fichier: {p['source']}" for p in data_points],
        hover_data={
            "Label": [p["label"] for p in data_points],
            "Audit IA": [f"{p['score']*100:.2f}%" for p in data_points],
            "Domaine": [p["domain"] for p in data_points]
        },
        color_discrete_map={d.capitalize(): c for d, c in domain_colors.items()}
    )

    # --- 5. DESIGN IMMERSIF ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="black", 
        plot_bgcolor="black",
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False), 
            bgcolor="black"
        ),
        legend=dict(
            title_text="",
            font=dict(size=13, color="#E5E5E7", family="Arial, sans-serif"),
            bgcolor="rgba(0,0,0,0.4)", 
            x=0.05,
            y=0.9,
            itemsizing='constant'
        ),
        margin=dict(l=0, r=0, b=0, t=0) 
    )

    fig.update_traces(
        marker=dict(size=3, opacity=0.85, line=dict(width=0)),
        selector=dict(mode='markers')
    )

    # --- 6. SAUVEGARDE ET OUVERTURE ---
    fig.write_html(str(output_path), config={'displayModeBar': False})
    
    print(f"Espace généré dans : {output_path}")
    import webbrowser
    webbrowser.open(f"file:///{output_path.absolute()}")

if __name__ == "__main__":
    generate_immersive_3d()