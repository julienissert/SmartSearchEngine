import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import lancedb
from src import config

# --- CONFIGURATION INTERFACE ---
st.set_page_config(layout="wide", page_title="SmartSearch Engine")

st.title("SmartSearch Explorer")

# --- CONNEXION BASE DE DONNÉES ---
@st.cache_resource
def get_db_connection():
    return lancedb.connect(config.LANCEDB_URI)

db = get_db_connection()

# --- BARRE LATÉRALE (SIDEBAR) : FILTRES & STATS ---
with st.sidebar:
    st.header("Configuration")
    
    table = db.open_table(config.TABLE_NAME)
    has_contracts = "folder_contracts" in db.table_names()
    
    st.metric("Total Documents", f"{len(table):,}")
    if has_contracts:
        contract_table = db.open_table("folder_contracts")
        st.metric("Contrats Dossiers", f"{len(contract_table)}")
    
    st.divider()
    
    st.subheader("Recherche & Filtres")
    filter_query = st.text_input(
        "Filtre SQL (WHERE)", 
        placeholder="ex: domain = 'food' AND domain_score > 0.8"
    )
    
    limit = st.number_input("Nombre de lignes à afficher", value=100, step=50)
    
    st.divider()
    st.caption(f"Base : {config.LANCEDB_URI}")

# --- ZONE CENTRALE : LES ONGLETS ---
tab1, tab2 = st.tabs(["Catalogue Multimodal", "Contrats de Confiance"])

# --- ONGLET 1 : CATALOGUE ---
with tab1:
    query = table.search()
    if filter_query:
        try:
            query = query.where(filter_query)
        except Exception as e:
            st.warning(f"Requête SQL invalide : {e}")

    df = query.limit(limit).to_pandas()
    
    if not df.empty:
        st.dataframe(
            df.drop(columns=['vector']) if 'vector' in df.columns else df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "domain_score": st.column_config.NumberColumn("Score Confiance", format="%.4f"),
                "extra": st.column_config.JsonColumn("Détails IA (JSON)"),
                "source": st.column_config.TextColumn("Chemin du fichier")
            }
        )
    else:
        st.info("Aucun document trouvé dans le catalogue.")

with tab2:
    if has_contracts:
        df_contracts = contract_table.to_pandas()
        if not df_contracts.empty:
            st.dataframe(
                df_contracts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence": st.column_config.NumberColumn("Confiance Initiale", format="%.4f"),
                    "is_verified": st.column_config.CheckboxColumn("Vérifié")
                }
            )
        else:
            st.info("Aucun contrat généré pour le moment.")
    else:
        st.warning("La table des contrats n'est pas encore créée.")