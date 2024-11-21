"""
************************************************************************************************************************

                             Prédiction avec les Modèles Entraînés : Régression, Arbres, SVM et Réseaux de Neurones

************************************************************************************************************************
"""

# Importation des bibliothèques nécessaires
import pandas as pd
import os
import joblib

"""
========================================================================================================================

                                                1. Chargement des Modèles et des Données

========================================================================================================================
"""

# Dossier de sortie pour les résultats de prédictions
predictions_output_dir = "sections/classification/classification_visualization/predictions"
if not os.path.exists(predictions_output_dir):
    os.makedirs(predictions_output_dir)

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
# Pour la prédiction, nous n'avons pas besoin de diviser les données ici
print(f"Jeu de données : {X.shape[0]} échantillons")

"""
========================================================================================================================

                                                2. Prédictions des Modèles

========================================================================================================================
"""

# Faire des prédictions pour chaque modèle
for model_name, model in models.items():
    print(f"\nPrédiction avec le modèle : {model_name} en cours...")

    try:
        # Prédictions des probabilités sur l'ensemble des données (pas uniquement le test)
        y_pred_proba = model.predict_proba(X)  # Probabilités des classes

        # Créer un DataFrame pour les probabilités et les classes prédites
        df_predictions = pd.DataFrame(y_pred_proba, columns=[f"Class_{i}" for i in range(y_pred_proba.shape[1])])
        df_predictions['Predicted_Class'] = df_predictions.idxmax(axis=1)  # Classe prédite (celle avec la probabilité la plus élevée)
        df_predictions['Actual_Class'] = y  # Classe réelle

        # Sauvegarder les prédictions dans un fichier CSV
        predictions_file_path = f"{predictions_output_dir}/{model_name.replace(' ', '_').lower()}_predictions.csv"
        df_predictions.to_csv(predictions_file_path, index=False)
        print(f"Les prédictions ont été sauvegardées sous : {predictions_file_path}")

    except Exception as e:
        print(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")

"""
========================================================================================================================

                                                3. Résumé et Conclusion

========================================================================================================================
"""

# Fin du script
print("\nPrédictions terminées pour tous les modèles.")
