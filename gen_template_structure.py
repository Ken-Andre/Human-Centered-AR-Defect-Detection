import os

folders = [
    "app/templates",
    "app/static/css",
    "app/static/js",
    "app/static/img",
    # "backend",
    "tests"
]

files = {
    "app/templates/index.html": "<!-- index.html generated -->",
    "app/static/css/style.css": "/* style.css generated */",
    "app/static/js/main.js": "// main.js generated",
    # "backend/server.py": "# server.py generated\n",
    # "backend/db.py": "# db.py placeholder\n",
    # "backend/sudo_detection.py": "# sudo_detection.py placeholder\n",
    "README.md": "# Human-Centered-AR-Defect-Detection\n\nProject structure initialized.",
    "requirements.txt": "# Flask\n# flask_socketio\n# requests\n"
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Ajoute un .gitkeep pour que le dossier soit versionné
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        f.write("")

for file_path, content in files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Projet généré !")
print("Mets à jour les fichiers index.html, main.js, style.css avec le code fourni.")
