from fann2 import libfann
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 1️Charger la base Iris

iris = load_iris()
X = iris.data         # (4 caractéristiques)
y = iris.target.reshape(-1, 1) 

#  encoding pour les 3 classes
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(y) 

# Diviser en train/test (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 2️Créer fichier .data pour FANN

with open("iris.data", "w") as f:
    f.write(f"{len(X_train)} 4 3\n")  # nb samples, nb entrées, nb sorties
    for xi, yi in zip(X_train, Y_train):
        f.write(" ".join(map(str, xi)) + "\n")
        f.write(" ".join(map(str, yi)) + "\n")


# 3️ Créer le réseau MLP

ann = libfann.neural_net()
ann.create_standard_array([4, 6, 3])  # 4 entrées, 6 neurones cachés, 3 sorties

# Fonctions d’activation
ann.set_activation_function_hidden(libfann.SIGMOID)
ann.set_activation_function_output(libfann.SIGMOID)


# 4️Entraîner le réseau

ann.train_on_file("iris.data", max_epochs=5000, epochs_between_reports=500, desired_error=0.01)
print("\nEntraînement Iris terminé.")


# 5️Tester sur le set test

print("\nTest du réseau sur le set test :")
for xi, yi in zip(X_test, Y_test):
    pred = ann.run(xi)
    pred_class = np.argmax(pred)
    true_class = np.argmax(yi)
    print(f"Entrée: {xi}")
    print(f"Sortie attendue: {yi} | Classe: {true_class}")
    print(f"Prédiction: {pred} | Classe: {pred_class}\n")
