# src/intelligence/llm_manager.py
import requests
import json
import time
import os
import sys
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("LLMManager")

class LLMManager:    
    def __init__(self):
        self.url = config.OLLAMA_GENERATE_URL
        # On déduit l'URL de base (ex: http://ollama:11434) pour les commandes de gestion
        self.base_url = self.url.replace("/api/generate", "")
        self.config = config.LLM_CONFIG
        self.model = self.config["model"]

        # --- AUTO-INSTALLATION AU DÉMARRAGE ---
        self._ensure_service_and_model()

    def _ensure_service_and_model(self):
        """
        Bloque le démarrage tant que Ollama n'est pas prêt et télécharge le modèle si nécessaire.
        """
        logger.info(f"⏳ Initialisation du LLM Manager (Cible : {self.base_url})...")
        
        # 1. Attente du service (Healthcheck)
        max_retries = 30
        service_ready = False
        for i in range(max_retries):
            try:
                requests.get(self.base_url, timeout=2)
                service_ready = True
                logger.info("✅ Service Ollama connecté.")
                break
            except requests.exceptions.RequestException:
                if i % 5 == 0: logger.warning("En attente de Ollama...")
                time.sleep(2)
        
        if not service_ready:
            logger.error("❌ Impossible de joindre Ollama. Vérifie le conteneur docker.")
            sys.exit(1)

        # 2. Vérification et Téléchargement du modèle
        if not self._check_model_exists():
            self._pull_model()

    def _check_model_exists(self) -> bool:
        """Vérifie si le modèle est présent localement via l'API Tags"""
        try:
            res = requests.get(f"{self.base_url}/api/tags")
            if res.status_code == 200:
                models = res.json().get('models', [])
                # On regarde si le nom du modèle est contenu dans un des tags
                return any(self.model in m['name'] for m in models)
        except Exception as e:
            logger.warning(f"Erreur check modèle : {e}")
        return False

    def _pull_model(self):
        """Ordonne à Ollama de télécharger le modèle"""
        logger.info(f"⬇️ Modèle '{self.model}' introuvable. Téléchargement en cours (ceci peut prendre du temps)...")
        try:
            # stream=False bloque la requête jusqu'à la fin du téléchargement
            res = requests.post(
                f"{self.base_url}/api/pull", 
                json={"name": self.model, "stream": False},
                timeout=None 
            )
            if res.status_code == 200:
                logger.info(f"✅ Modèle '{self.model}' installé avec succès !")
            else:
                logger.error(f"❌ Échec téléchargement : {res.text}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Erreur critique pull modèle : {e}")
            sys.exit(1)

    # --- LE REST DE TON CODE RESTE IDENTIQUE ---
    
    def is_healthy(self) -> bool:
        """Vérifie si le serveur Ollama répond."""
        try:
            response = requests.get(self.base_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _generate(self, prompt: str, system_prompt: str = ""):
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
                response = requests.post(
                    self.url, 
                    json=payload, 
                    timeout=self.config["timeout"]
                )
                response.raise_for_status()
                
                raw_res = response.json().get("response", "")
                
                if not raw_res: raise ValueError("Réponse LLM vide")
                
                start = raw_res.find('{')
                end = raw_res.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = raw_res[start:end]
                    return json.loads(json_str)   
                 
                return json.loads(raw_res)

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Échec LLM : {e}")
        
        return default_return

    # [Tes autres méthodes: arbitrate_domain, identify_csv_mapping, etc. restent inchangées]
    def arbitrate_domain(self, text_sample: str | None, clip_scores: dict, filepath: str | None = None):
        system = "Tu es un expert en classification documentaire. Réponds uniquement en JSON."
        safe_text = (text_sample or "")[:1000]
        filename = os.path.basename(filepath) if filepath else "Inconnu"
        prompt = (
            f"FICHIER : {filename}\nTEXTE : '{safe_text}'\nSCORES CLIP : {clip_scores}\n"
            "Analyse le texte et détermine si le domaine proposé par CLIP est correct. "
            "Réponds avec ce format : {\"final_domain\": \"string\", \"confidence\": float, \"justification\": \"string\"}"
        )
        return self._generate(prompt, system)

    def identify_csv_mapping(self, csv_sample: str):
        system = "Tu es un expert en analyse de données structurées. Réponds uniquement en JSON."
        prompt = (
            f"Voici un extrait de CSV : '{csv_sample}'\n"
            "1. Identifie la colonne qui contient le nom principal (label).\n"
            "2. Cherche une colonne avec des chemins d'images (ex: .jpg).\n"
            "Format: {\"label_column\": \"string\", \"image_path_column\": \"string\" | null, \"reason\": \"string\"}"
        )
        return self._generate(prompt, system)

    def refine_image_label(self, ocr_text: str | None, current_label: str):
        safe_ocr = (ocr_text or "")[:500]
        system = "Tu es un expert en reconnaissance de documents. Réponds uniquement en JSON."
        prompt = (
            f"OCR : '{safe_ocr}'\nLabel CLIP : '{current_label}'\n"
            "Génère un label court (3-5 mots). Format: {\"refined_label\": \"string\", \"is_document\": bool}"
        )
        return self._generate(prompt, system)

    def extract_extra_metadata(self, text: str | None, domain: str | None = None):
        if not text: return {"status": "no_content", "confidence": 0.0}
        role_domain = domain if domain in config.TARGET_DOMAINS else "Général"
        system = f"Expert domaine '{role_domain}'. Réponds en JSON."
        prompt = (
            f"TEXTE : '{text[:2000]}'\n"
            "Extrais 5 infos critiques, date, résumé, mots-clés.\n"
            "Format: {{\"date\": \"string\", \"entities\": [], \"keywords\": [], \"summary\": \"string\", \"confidence\": float}}"
        )
        return self._generate(prompt, system)

    def synthesize_answer(self, query: str, context: list):
        if not context: return {"answer": "Non trouvé.", "sources_used": [], "found": False}
        formatted_context = ""
        for i, doc in enumerate(context, 1):
            snippet = doc.get('snippet', doc.get('content', ''))[:1000]
            formatted_context += f"--- DOC [{i}] ---\n{snippet}\n\n"
        system = "Assistant expert. Réponds uniquement avec le contexte fourni en JSON."
        prompt = (
            f"CONTEXTE:\n{formatted_context}\nQUESTION: '{query}'\n"
            "Format: {{\"answer\": \"string\", \"sources_used\": [\"string\"], \"confidence\": float, \"found\": bool}}"
        )
        return self._generate(prompt, system)
    
    def analyze_scan_intent(self, ocr_text: str):
        if not ocr_text or len(ocr_text) < 5: return {"domain": "unknown", "label": "unknown", "type": "image"}
        system = "Expert analyse documents. Réponds en JSON."
        prompt = (
            f"TEXTE : '{ocr_text}'\n"
            "Format: {\"domain\": \"string\", \"label\": \"string\", \"type\": \"string\"}"
        )
        return self._generate(prompt, system)
    
    def identify_mapping_plan(self, sample_text: str, extension: str):
        """
        Phase de 'Solo Test' : Analyse un échantillon pour fixer la règle d'extraction
        """
        system = "Tu es un expert en analyse de structure de données. Réponds uniquement en JSON."
        
        # Le prompt s'adapte selon l'extension pour être plus précis
        if extension in ['.csv', '.tsv']:
            prompt = (
                f"Voici un échantillon d'un fichier {extension} :\n'{sample_text}'\n"
                "Identifie la colonne qui contient le label (le nom, le titre ou l'entité principale).\n"
                "Format de réponse : {\"type\": \"column\", \"key\": \"nom_de_la_colonne\"}"
            )
        else: # Cas du .txt (les 10 lignes que tu envoies)
            prompt = (
                f"Voici les 10 premières lignes d'un fichier .txt :\n'{sample_text}'\n"
                "Analyse la structure pour extraire le label le plus pertinent (Titre, Sujet, etc.).\n"
                "Est-ce la première ligne ? Un motif spécifique comme 'Titre: ...' ?\n"
                "Format : {\"type\": \"txt_strategy\", \"key\": \"first_line\" | \"pattern\", \"pattern\": \"string_si_besoin\" | null}"
            )
            
        return self._generate(prompt, system)

# Instance unique
llm = LLMManager()