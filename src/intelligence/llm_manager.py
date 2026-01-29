# src/intelligence/llm_manager.py
import requests
import json
import time
import os
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("LLMManager")

class LLMManager:    
    def __init__(self):
        self.url = config.OLLAMA_GENERATE_URL
        self.config = config.LLM_CONFIG
        self.model = self.config["model"]

    def is_healthy(self) -> bool:
        """
        Vérifie si le serveur Ollama répond.
        Essentiel pour valider l'environnement avant l'ingestion.
        """
        try:
            # On teste l'URL de base (ex: http://localhost:11434)
            base_url = self.url.replace("/api/generate", "")
            response = requests.get(base_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _generate(self, prompt: str, system_prompt: str = ""):
        """
        Moteur d'inférence privé avec mécanisme de résilience.
        Force le format JSON, respecte les limites de VRAM et retente en cas d'échec.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.config["temperature"],
                "num_ctx": self.config["num_ctx"]
            }
        }
        
        default_return = {"domain": "unknown", "label": "unknown", "type": "unknown"}
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Appel à l'API Ollama avec le timeout dynamique de la config
                response = requests.post(
                    self.url, 
                    json=payload, 
                    timeout=self.config["timeout"]
                )
                response.raise_for_status()
                
                raw_res = response.json().get("response", "")
                
                # Validation du JSON avant de le retourner
                if not raw_res:
                    raise ValueError("Réponse LLM vide")
                
                start = raw_res.find('{')
                end = raw_res.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = raw_res[start:end]
                    return json.loads(json_str)   
                 
                return json.loads(raw_res)

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Tentative {attempt + 1}/{max_retries} échouée pour {self.model}. "
                        f"Nouvel essai dans 2s... (Erreur: {e})"
                    )
                    time.sleep(2) # Micro-pause pour libérer le bus GPU/RAM
                else:
                    logger.error(
                        f"Échec définitif de l'inférence LLM après {max_retries} tentatives. "
                        f"Modèle: {self.model} | Erreur: {e}"
                    )
        
        return default_return

    # --- 1. L'ARBITRE DE DOMAINE (Domain Decision) ---
    def arbitrate_domain(self, text_sample: str | None, clip_scores: dict, filepath: str | None = None):
            """
            Tranche entre les domaines si CLIP hésite.
            """
            system = "Tu es un expert en classification documentaire. Réponds uniquement en JSON."
            
            # On sécurise les variables pour éviter les erreurs de manipulation None
            safe_text = (text_sample or "")[:1000]
            filename = os.path.basename(filepath) if filepath else "Inconnu"
            
            prompt = (
                f"FICHIER : {filename}\n"
                f"TEXTE : '{safe_text}'\n"
                f"SCORES CLIP : {clip_scores}\n"
                "Analyse le texte et détermine si le domaine proposé par CLIP est correct. "
                "Réponds avec ce format : {\"final_domain\": \"string\", \"confidence\": float, \"justification\": \"string\"}"
            )
            return self._generate(prompt, system)

    # --- 2. L'EXPERT LABELS (CSV & OCR) ---
    def identify_csv_mapping(self, csv_sample: str):
        """
        Analyse la structure d'un CSV pour mapper les colonnes critiques.
        Gère intelligemment l'absence de colonnes d'images.
        """
        system = "Tu es un expert en analyse de données structurées. Réponds uniquement en JSON."
        prompt = (
            f"Voici un extrait de CSV : '{csv_sample}'\n"
            "1. Identifie la colonne qui contient le nom principal ou la catégorie (le label).\n"
            "2. Cherche si une colonne contient des chemins ou noms de fichiers images (ex: .jpg, .png, .webp).\n"
            "Si aucune colonne d'image n'est trouvée, mets 'image_path_column' à null.\n"
            "Réponds avec ce format exact : "
            "{\"label_column\": \"string\", \"image_path_column\": \"string\" | null, \"reason\": \"string\"}"
        )
        return self._generate(prompt, system)

    # --- 3. L'IDENTIFICATEUR D'IMAGE (Nom générique vs Contenu) ---
    def refine_image_label(self, ocr_text: str | None, current_label: str):
        """
        Précise un label d'image (ex: '1039.jpg' -> 'Facture Pharmacie').
        Utilise safe_ocr pour éviter les plantages si aucun texte n'est extrait.
        """
        # Sécurisation : si None, devient une chaîne vide, puis on limite à 500 caractères
        safe_ocr = (ocr_text or "")[:500]
        
        system = "Tu es un expert en reconnaissance de documents. Réponds uniquement en JSON."
        
        prompt = (
            f"Texte extrait de l'image (OCR) : '{safe_ocr}'\n" # <-- ON UTILISE SAFE ICI
            f"Label actuel (CLIP) : '{current_label}'\n"
            "Génère un label court et précis (3-5 mots) décrivant ce document. "
            "Réponds au format : {{"
            "\"refined_label\": \"string\", \"is_document\": bool"
            "}}"
        )
        return self._generate(prompt, system)

    # --- 4. L'ENRICHISSEUR (Version Élite Corrigée) ---
    def extract_extra_metadata(self, text: str | None, domain: str | None = None):
        """
        Analyse et extrait des métadonnées sans aucune instruction en dur.
        Force l'IA à utiliser son expertise interne pour le domaine spécifié.
        """
        if not text:
            return {"status": "no_content", "confidence": 0.0}

        # On injecte le domaine dynamiquement depuis la config
        role_domain = domain if domain in config.TARGET_DOMAINS else "Général"
        
        system = f"Tu es un expert mondial en analyse de documents pour le domaine '{role_domain}'. Réponds uniquement en JSON."
        
        # NOTE : On utilise {{ }} pour que les accolades soient ignorées par la f-string
        prompt = (
            f"TEXTE À ANALYSER : '{text[:2000]}'\n\n"
            "MISSION D'EXTRACTION :\n"
            f"1. Utilise ta connaissance experte du domaine '{role_domain}' pour identifier les 5 informations les plus critiques présentes dans ce texte.\n"
            "2. Extrais la date du document au format ISO 'YYYY-MM-DD' (si absente: null).\n"
            "3. Rédige un résumé technique d'une phrase et liste 5 mots-clés sémantiques.\n\n"
            "FORMAT DE RÉPONSE (JSON STRICT) :\n"
            "{{"
            "\"date\": \"string\", \"entities\": [], \"keywords\": [], "
            "\"summary\": \"string\", \"confidence\": float"
            "}}"
        )

        return self._generate(prompt, system)

    # --- 5. LE GÉNÉRATEUR RAG (Version Élite) ---
    def synthesize_answer(self, query: str, context: list):
        """
        Rédige une réponse basée exclusivement sur le contexte fourni.
        Gère les sources multimodales et évite les hallucinations.
        """
        if not context:
            return {
                "answer": "Désolé, je n'ai trouvé aucun document pertinent pour répondre à votre question.",
                "sources_used": [],
                "found": False
            }

        # Formatage propre du contexte pour l'IA
        # On transforme la liste d'extraits en une chaîne structurée
        formatted_context = ""
        for i, doc in enumerate(context, 1):
            source_name = os.path.basename(doc.get('source', 'Inconnu'))
            snippet = doc.get('snippet', doc.get('content', ''))[:1000]
            formatted_context += f"--- DOCUMENT [{i}] (Source: {source_name}) ---\n{snippet}\n\n"

        system = (
            "Tu es un assistant de recherche expert. Ta mission est de répondre à la question "
            "en utilisant UNIQUEMENT les documents fournis. Si l'information est absente, "
            "dis explicitement que tu ne sais pas. Réponds en JSON, dans la langue de la question."
        )
        
        prompt = (
            f"CONTEXTE DE RECHERCHE :\n{formatted_context}\n"
            f"QUESTION DE L'UTILISATEUR : '{query}'\n\n"
            "CONSIGNES :\n"
            "1. Cite les sources en utilisant leur numéro entre crochets, ex: [1].\n"
            "2. Sois précis et technique.\n"
            "3. Si tu ne trouves pas la réponse, l'attribut 'answer' doit l'indiquer.\n\n"
            "FORMAT DE RÉPONSE (JSON) :\n"
            "{{"
            "\"answer\": \"string\", \"sources_used\": [\"string\"], "
            "\"confidence\": float, \"found\": bool"
            "}}"
        )
        
        return self._generate(prompt, system)
    
    # --- 6. L'ANALYSTE D'OCR  ---
    def analyze_scan_intent(self, ocr_text: str):
        """
        Transforme le bruit de l'OCR en filtres de recherche précis.
        """
        if not ocr_text or len(ocr_text) < 5:
            return {"domain": "unknown", "label": "unknown", "type": "image"}

        system = "Tu es un expert en analyse de documents et produits. Réponds uniquement en JSON."
        prompt = (
            f"TEXTE EXTRAIT DU SCAN : '{ocr_text}'\n\n"
            "Analyse ce texte et identifie :\n"
            "1. Le domaine le plus probable : ['food', 'medical']\n"
            "2. Le nom précis de l'objet ou du médicament (le label).\n"
            "3. Le type de document s'il y a des indices (ex: 'facture', 'ordonnance', 'produit').\n\n"
            "Réponds avec ce format : {\"domain\": \"string\", \"label\": \"string\", \"type\": \"string\"}"
        )
        return self._generate(prompt, system)
    
# Instance unique (Singleton)
llm = LLMManager()