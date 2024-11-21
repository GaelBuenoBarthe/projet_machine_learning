"""
************************************************************************************************************************

Entraînement et Évaluation de Modèles de Machine Learning : Régression, Arbres, SVM et Réseaux de Neurones

************************************************************************************************************************
"""

# Importation des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

"""
========================================================================================================================

                                                1. Préparation des Données

========================================================================================================================
"""

# Charger les données nettoyées et prétraitées
df = pd.read_csv('data/data_cleaned_preprocessed_reduced.csv')

# Séparer les features (X) et la cible (y)
X = df.drop(columns='target')  # X : Features explicatives
y = df['target']  # y : Variable cible

# Diviser les données en ensembles d'entraînement et de test (80% pour l'entraînement, 20% pour le test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nJeu d'entraînement : {X_train.shape[0]} échantillons")
print(f"Jeu de test : {X_test.shape[0]} échantillons")

"""
========================================================================================================================

                                                2. Entraînement des Modèles

========================================================================================================================
"""

# Dictionnaire des modèles à entraîner
models = {
    "Regression Logistique": LogisticRegression(max_iter=1000, random_state=42),
    "Arbres de Decision": DecisionTreeClassifier(random_state=42),
    "Forets Aleatoires": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Support Vector Machines)": SVC(probability=True, random_state=42),
    "Reseau de Neurones": MLPClassifier(max_iter=1000, random_state=42)
}

# Initialiser un dossier pour sauvegarder les modèles entraînés
output_dir = "sections/classification/classification_visualization/model_training"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Entraîner et sauvegarder chaque modèle
for model_name, model in models.items():
    print(f"\nEntraînement du modèle : {model_name} en cours...")

    try:
        # Entraîner le modèle sur les données d'entraînement
        model.fit(X_train, y_train)
        print(f"{model_name} entraîné avec succès !")

        # Sauvegarder le modèle entraîné
        model_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Modèle {model_name} sauvegardé sous : {model_path}")

    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle {model_name} : {e}")

"""
========================================================================================================================

                                                3. Évaluation des Modèles

========================================================================================================================
"""

# Évaluation des modèles entraînés
for model_name, model in models.items():
    print(f"\nÉvaluation du modèle : {model_name}...")

    try:
        # Faire des prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calculer la précision du modèle
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Précision du modèle {model_name} : {accuracy * 100:.2f}%")

        # Afficher le rapport de classification
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Sauvegarder le rapport de classification dans un fichier
        classification_report_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}_classification_report.txt"
        with open(classification_report_path, 'w') as f:
            f.write(classification_report(y_test, y_pred))
        print(f"Rapport de classification sauvegardé sous : {classification_report_path}")

        # Affichage de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_,cbar=False)
        plt.title(f"Matrice de confusion - {model_name}")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")

        # Sauvegarder la matrice de confusion sous forme de fichier
        confusion_matrix_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        print(f"Matrice de confusion sauvegardée sous : {confusion_matrix_path}")

        # Fermeture de la figure pour éviter des problèmes avec plt.show()
        plt.close()

    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle {model_name} : {e}")

"""
========================================================================================================================

                                                4. Conclusion et Résultats

========================================================================================================================
"""

# Affichage du résumé du modèle
print("\nRésumé des résultats de l'entraînement :")
print("---------------------------------------------------")
for model_name, model in models.items():
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModèle {model_name} - Précision : {accuracy * 100:.2f}%")

        # Sauvegarder les résultats du modèle pour une analyse future
        model_results_path = f"{output_dir}/{model_name.replace(' ', '_').lower()}_model_results.txt"
        with open(model_results_path, 'w') as f:
            f.write(f"Précision du modèle {model_name} : {accuracy * 100:.2f}%\n")
            f.write("Rapport de classification :\n")
            f.write(classification_report(y_test, y_pred))
        print(f"Résultats du modèle sauvegardés sous : {model_results_path}")

    except Exception as e:
        print(f"Erreur lors de l'affichage des résultats du modèle {model_name} : {e}")

# Fin du script
print("\nEntraînement et évaluation des modèles terminés.")
