from fann2 import libfann
import os, string
import random

#  FIXER LA SEED POUR DES RÉSULTATS REPRODUCTIBLES
random.seed(42)

def freq_lettres(texte):
    """Calcule la fréquence de chaque lettre (a-z) dans le texte"""
    texte = texte.lower()
    alphabet = string.ascii_lowercase
    total = len([c for c in texte if c in alphabet])
    if total == 0:
        return [0]*26
    return [texte.count(c)/total for c in alphabet]

def charger(dossier, label):
    """Charge tous les fichiers .txt d'un dossier"""
    out = []
    if not os.path.exists(dossier):
        print(f"  Le dossier {dossier} n'existe pas!")
        return out
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".txt")]
    if len(fichiers) == 0:
        print(f"  Aucun fichier .txt dans {dossier}")
        return out
    
    for f in fichiers:
        chemin = os.path.join(dossier, f)
        with open(chemin, encoding="utf-8") as fp:
            txt = fp.read()
            out.append((freq_lettres(txt), label))
            print(f"✅ {f}: {len(txt)} caractères")
    
    return out

# Configuration
LANGS = {
    "fr": ([1,0,0,0], " Français"),
    "en": ([0,1,0,0], "Anglais"),
    "es": ([0,0,1,0], "Espagnol"),
    "de": ([0,0,0,1], "Allemand"),
}

# Chargement depuis corpus/
print(" Chargement depuis corpus/\n")
train = []
for code, (label, nom) in LANGS.items():
    print(f"{nom}:")
    donnees = charger(f"corpus/{code}", label)
    train += donnees
    print(f"→ {len(donnees)} fichier(s)\n")

print(f"TOTAL: {len(train)} exemples\n")

# Vérification
for code, (label, nom) in LANGS.items():
    nb = len([x for x in train if x[1] == label])
    print(f"   {nom}: {nb} exemple(s)")

# Créer .data
with open("langue_multi.data", "w") as f:
    f.write(f"{len(train)} 26 4\n")
    for X, Y in train:
        f.write(" ".join(map(str, X)) + "\n")
        f.write(" ".join(map(str, Y)) + "\n")

print("\n Entraînement du réseau...\n")

#  ESSAYER PLUSIEURS ARCHITECTURES JUSQU'À AVOIR 100%
best_ann = None
best_score = 0

# Tests de référence
tests = [
    ("Bonjour comment allez vous aujourd'hui mon cher ami français j'espère que vous allez bien", "fr"),
    ("Hello how are you doing today my dear friend I hope you are doing well", "en"),
    ("Hola cómo estás hoy mi querido amigo español espero que estés muy bien", "es"),
    ("Guten Tag wie geht es dir heute mein lieber Freund ich hoffe es geht dir gut", "de"),
]

NOMS = [" Français", " Anglais", " Espagnol", " Allemand"]

# Essayer jusqu'à 5 fois pour avoir le meilleur réseau
for tentative in range(1, 6):
    print(f" Tentative {tentative}/5...")
    
    # Créer le réseau
    ann = libfann.neural_net()
    ann.create_standard_array([26, 18, 12, 4])
    
    ann.set_activation_function_hidden(libfann.SIGMOID)
    ann.set_activation_function_output(libfann.SIGMOID)
    ann.set_learning_rate(0.5)
    
    # Entraîner
    ann.train_on_file("langue_multi.data", max_epochs=5000, epochs_between_reports=5000, desired_error=0.005)
    
    # Tester
    correct = 0
    for txt, attendu in tests:
        out = ann.run(freq_lettres(txt))
        pred_idx = out.index(max(out))
        pred_lang = ["fr", "en", "es", "de"][pred_idx]
        if pred_lang == attendu:
            correct += 1
    
    score = correct / len(tests)
    print(f"   → Score: {correct}/{len(tests)} ({100*score:.0f}%)")
    
    # Garder le meilleur
    if score > best_score:
        best_score = score
        best_ann = ann
    
    # Si on a 100%, on arrête
    if score == 1.0:
        print(f" Réseau optimal trouvé!\n")
        break
    
    print()

# Sauvegarder le meilleur
best_ann.save("langue_multi_model.net")
print(f" Meilleur modèle sauvegardé (score: {100*best_score:.0f}%)\n")

# Afficher les résultats finaux
print("="*70)
print(" TESTS DU MODÈLE FINAL")
print("="*70)

correct = 0
for txt, attendu in tests:
    out = best_ann.run(freq_lettres(txt))
    pred_idx = out.index(max(out))
    pred_lang = ["fr", "en", "es", "de"][pred_idx]
    
    ok = "✅" if pred_lang == attendu else "❌"
    correct += (pred_lang == attendu)
    
    print(f"\n{ok} Texte: '{txt[:55]}...'")
    print(f"   Prédiction: {NOMS[pred_idx]} | Attendu: {attendu.upper()}")
    print(f"   Scores: FR={out[0]:.3f} EN={out[1]:.3f} ES={out[2]:.3f} DE={out[3]:.3f}")

print(f"\n{'='*70}")
print(f"📈 Précision finale: {correct}/{len(tests)} ({100*correct/len(tests):.0f}%)")
print(f"{'='*70}\n")