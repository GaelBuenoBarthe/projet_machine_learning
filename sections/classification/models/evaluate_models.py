"""
************************************************************************************************************************

                             Évaluation des Modèles Entraînés : Régression, Arbres, SVM et Réseaux de Neurones

************************************************************************************************************************
"""

# Importation des bibliothèques nécessaires
import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt

"""
========================================================================================================================

                                                1. Chargement des Modèles et des Données

========================================================================================================================
"""

# Dossier de sortie pour les résultats d'évaluation
evaluation_output_dir = "sections/classification/classification_visualization/evaluation"
if not os.path.exists(evaluation_output_dir):
    os.makedirs(evaluation_output_dir)

# Dictionnaire des modèles à charger
models = {
    "Régression Logistique": joblib.load("sections/classification/classification_visualization/model_training/regression_logistique_model.pkl"),
    "Arbres de Décision": joblib.load("sections/classification/classification_visualization/model_training/arbres_de_decision_model.pkl"),
    "Forêts Aléatoires": joblib.load("sections/classification/classification_visualization/model_training/forets_aleatoires_model.pkl"),
    "SVM (Support Vector Machines)": joblib.load("sections/classification/classification_visualization/model_training/svm_(support_vector_machines)_model.pkl"),
    "Réseau de Neurones": joblib.load("sections/classification/classification_visualization/model_training/reseau_de_neurones_model.pkl")
}

# Charger les données nettoyées et prétraitées
df = pd.read_csv('data/data_cleaned_preprocessed_reduced.csv')

# Séparer les features (X) et la cible (y)
X = df.drop(columns='target')  # X : Features explicatives
y = df['target']  # y : Variable cible

# Diviser les données en ensembles d'entraînement et de test (80% pour l'entraînement, 20% pour le test)
# Pour l'évaluation, nous n'avons pas besoin de diviser les données ici
print(f"Jeu de données : {X.shape[0]} échantillons")

"""
========================================================================================================================

                                                2. Évaluation des Modèles

========================================================================================================================
"""

# Dictionnaire pour stocker les résultats des modèles
evaluation_results = []

# Préparer une figure pour les matrices de confusion consolidées
plt.figure(figsize=(15, 15))  # Taille de la figure pour afficher toutes les matrices côte à côte

# Évaluer chaque modèle
for i, (model_name, model) in enumerate(models.items()):
    print(f"\nÉvaluation du modèle : {model_name} en cours...")

    try:
        # Faire des prédictions sur l'ensemble des données
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)  # Pour AUC-ROC et Log-Loss

        # Calculer la précision du modèle
        accuracy = accuracy_score(y, y_pred)
        print(f"Précision du modèle {model_name} : {accuracy * 100:.2f}%")

        # Calculer les autres métriques : F1-Score, Précision, Rappel, AUC-ROC, Log-Loss
        f1 = f1_score(y, y_pred, average=None)  # F1-Score pour chaque classe
        precision = precision_score(y, y_pred, average=None)  # Précision pour chaque classe
        recall = recall_score(y, y_pred, average=None)  # Rappel pour chaque classe
        auc_roc = roc_auc_score(y, y_pred_proba, multi_class='ovr')  # AUC-ROC pour toutes les classes
        log_loss_value = log_loss(y, y_pred_proba)  # Log-Loss pour le modèle

        # Afficher les résultats pour chaque classe
        for class_label in range(len(f1)):
            print(f"Classe {class_label}: F1-Score: {f1[class_label]:.2f}, Précision: {precision[class_label]:.2f}, Rappel: {recall[class_label]:.2f}")

        # Sauvegarder la précision dans un fichier
        accuracy_file_path = f"{evaluation_output_dir}/{model_name.replace(' ', '_').lower()}_accuracy.txt"
        with open(accuracy_file_path, 'w') as f:
            f.write(f"Précision du modèle {model_name} : {accuracy * 100:.2f}%\n")
        print(f"Précision sauvegardée sous : {accuracy_file_path}")

        # Ajouter les résultats au tableau
        evaluation_results.append({
            'Modèle': model_name,
            'Précision': accuracy,
            'AUC-ROC': auc_roc,
            'Log-Loss': log_loss_value,
            'F1-Score (Classes)': ', '.join([f"{x:.2f}" for x in f1]),
            'Précision (Classes)': ', '.join([f"{x:.2f}" for x in precision]),
            'Rappel (Classes)': ', '.join([f"{x:.2f}" for x in recall])
        })

        # Affichage et sauvegarde de la matrice de confusion
        cm = confusion_matrix(y, y_pred)
        plt.subplot(3, 2, i+1)  # Placer chaque matrice dans une sous-figure
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, cbar=False)
        plt.title(f"Matrice de confusion - {model_name}")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")

        # Ajuster la mise en page pour éviter tout chevauchement
        plt.tight_layout()

    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle {model_name} : {e}")

# Sauvegarder la figure consolidée des matrices de confusion
confusion_matrix_output_path = os.path.join(evaluation_output_dir, 'confusion_matrices.png')
plt.savefig(confusion_matrix_output_path)
plt.close()  # Fermer la figure après l'enregistrement
print(f"Matrices de confusion sauvegardées sous : {confusion_matrix_output_path}")

# Créer un DataFrame avec les résultats des évaluations
evaluation_df = pd.DataFrame(evaluation_results)

# Sauvegarder les résultats dans un fichier CSV
evaluation_results_file_path = os.path.join(evaluation_output_dir, 'evaluation_results.csv')
evaluation_df.to_csv(evaluation_results_file_path, index=False)
print(f"Résultats d'évaluation sauvegardés sous : {evaluation_results_file_path}")
