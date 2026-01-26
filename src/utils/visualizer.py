# src/utils/visualizer.py
import sys
import os
import hashlib
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

from src.indexing import vector_store
from src import config

def get_pastel_color(name):
    """Génère une couleur pastel stable à partir d'une chaîne."""
    hash_digest = hashlib.md5(name.encode()).digest()
    r = (hash_digest[0] % 105) + 150
    g = (hash_digest[1] % 105) + 150
    b = (hash_digest[2] % 105) + 150
    return f"rgb({r}, {g}, {b})"

def generate_immersive_3d(output_html="space_explorer.html"):
    """
    Génère une visualisation 3D immersive depuis le store LanceDB.
    """
    viz_dir = config.COMPUTED_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / output_html

    # --- 1. CHARGEMENT DES DONNÉES DEPUIS LANCED B ---
    print("Connexion au store pour extraction visuelle...")
    table = vector_store.init_tables()
    
    # On récupère les données sous forme de DataFrame
    # Note : On limite à 5000 points pour garder un navigateur fluide (30Go = trop de points sinon)
    df = table.to_pandas()
    
    if df.empty:
        print(" Aucune donnée dans LanceDB pour la visualisation.")
        return

    if len(df) > 5000:
        print(f" Dataset important ({len(df)} points). Échantillonnage de 5000 points pour la fluidité.")
        df = df.sample(5000)

    # --- 2. PRÉPARATION DES VECTEURS POUR PCA ---
    X = np.array(df['vector'].tolist())
    
    print(f"Réduction dimensionnelle (PCA) sur {len(df)} vecteurs...")
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)

    # Ajout des coordonnées PCA au DataFrame pour Plotly
    df['x'] = components[:, 0]
    df['y'] = components[:, 1]
    df['z'] = components[:, 2]
    
    # Préparation des labels d'affichage
    df['source_name'] = df['source'].apply(lambda x: os.path.basename(x))
    df['domain_display'] = df['domain'].apply(lambda x: x.capitalize())

    # --- 3. MAPPING DES COULEURS ---
    unique_domains = df['domain'].unique()
    domain_colors = {dom.capitalize(): get_pastel_color(dom) for dom in unique_domains}

    # --- 4. CONSTRUCTION DU GRAPHIQUE ---
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='domain_display',
        hover_name='source_name',
        hover_data={
            'x': False, 'y': False, 'z': False,
            'label': True,
            'domain_score': ':.2f',
            'type': True
        },
        color_discrete_map=domain_colors
    )

    # --- 5. DESIGN IMMERSIF (STYLE ÉLITE) ---
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
            title_text="DOMAINES",
            font=dict(size=12, color="#E5E5E7"),
            bgcolor="rgba(0,0,0,0.5)", 
            x=0.05, y=0.9
        ),
        margin=dict(l=0, r=0, b=0, t=0) 
    )

    fig.update_traces(
        marker=dict(size=3, opacity=0.7, line=dict(width=0)),
        selector=dict(mode='markers')
    )

    # --- 6. SAUVEGARDE ET OUVERTURE ---
    fig.write_html(str(output_path), config={'displayModeBar': False})
    
    print(f"✅ Immersion 3D générée : {output_path}")
    import webbrowser
    webbrowser.open(f"file:///{output_path.absolute()}")

if __name__ == "__main__":
    generate_immersive_3d()