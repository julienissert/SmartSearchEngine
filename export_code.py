import os

# Les dossiers à ignorer pour ne pas polluer (venv, git, cache...)
IGNORE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'node_modules', 'data', 'logs'}
# Les extensions de fichiers qu'on veut lire
EXTENSIONS = {'.py', '.json', '.md', '.yml', '.yaml', '.txt', '.env'}

def gather_code(start_path="."):
    output = []
    
    # 1. D'abord, on affiche l'arborescence
    output.append("=== ARBORESCENCE DU PROJET ===")
    for root, dirs, files in os.walk(start_path):
        # On filtre les dossiers ignorés
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        output.append(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output.append(f"{subindent}{f}")
    
    output.append("\n" + "="*50 + "\n")

    # 2. Ensuite, on affiche le contenu des fichiers
    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                file_path = os.path.join(root, file)
                # On évite notre propre script d'export
                if file == "export_code.py":
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    output.append(f"START FILE: {file_path}")
                    output.append("-" * 20)
                    output.append(content)
                    output.append("-" * 20)
                    output.append(f"END FILE: {file_path}\n")
                except Exception as e:
                    print(f"Erreur de lecture sur {file_path}: {e}")

    return "\n".join(output)

if __name__ == "__main__":
    full_context = gather_code(".")
    
    # Écriture dans un fichier
    with open("project_context.txt", "w", encoding="utf-8") as f:
        f.write(full_context)
    
    print("✅ Tout le code a été sauvegardé dans 'project_context.txt'")
    print("👉 Tu peux maintenant glisser-déposer ce fichier ou copier son contenu dans le chat.")