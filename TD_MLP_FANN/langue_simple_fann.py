from fann2 import libfann
import os
import string

def freq_lettres(texte):
    texte = texte.lower()
    alphabet = string.ascii_lowercase
    total = len([c for c in texte if c in alphabet])
    if total == 0:
        return [0]*26
    return [texte.count(c)/total for c in alphabet]

def charger_corpus(dossier, label):
    data = []
    for f in os.listdir(dossier):
        if f.endswith(".txt"):
            with open(os.path.join(dossier, f), "r", encoding="utf-8") as fp:
                txt = fp.read()
                data.append((freq_lettres(txt), label))
    return data

# Charger données
train = []
train += charger_corpus("corpus/fr", [1, 0])  # Français
train += charger_corpus("corpus/en", [0, 1])  # Anglais

# Créer fichier data
datafile = "langue2.data"
with open(datafile, "w") as f:
    f.write(f"{len(train)} 26 2\n")
    for X, Y in train:
        f.write(" ".join(map(str, X)) + "\n")
        f.write(" ".join(map(str, Y)) + "\n")

# MLP
ann = libfann.neural_net()
ann.create_standard_array([26, 20, 2])

ann.set_activation_function_hidden(libfann.SIGMOID)
ann.set_activation_function_output(libfann.SIGMOID)

ann.train_on_file(datafile, max_epochs=3000, epochs_between_reports=500, desired_error=0.01)

print(" ENTRAINEMENT TERMINÉ.\n")

#  AJOUT: Sauvegarder le modèle
ann.save("langue2_model.net")
print(" Modèle sauvegardé dans langue2_model.net\n")

#  AJOUT: Tests multiples
tests = [
    "Je suis français et j'aime les croissants.",
    "Hello, how are you today?",
    "The weather is beautiful in London.",
    "Bonjour, comment allez-vous?"
]

print(" Tests du modèle:\n")
for txt in tests:
    out = ann.run(freq_lettres(txt))
    langue = " Français" if out[0] > out[1] else " Anglais"
    confiance = max(out) * 100
    print(f"Texte: '{txt[:50]}...'")
    print(f"Prédiction: {langue} (confiance: {confiance:.1f}%)")
    print(f"Scores [FR/EN]: [{out[0]:.3f}, {out[1]:.3f}]\n")