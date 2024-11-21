import streamlit as st
import pandas as pd
import os
import joblib

# Dossier de sortie pour les résultats de prédictions
predictions_output_dir = "sections/classification/classification_visualization/predictions"
if not os.path.exists(predictions_output_dir):
    os.makedirs(predictions_output_dir)

# Dictionnaire des modèles à charger
models = {
    "Régression Logistique": joblib.load(
        "sections/classification/classification_visualization/model_training/regression_logistique_model.pkl"),
    "Arbres de Décision": joblib.load(
        "sections/classification/classification_visualization/model_training/arbres_de_decision_model.pkl"),
    "Forêts Aléatoires": joblib.load(
        "sections/classification/classification_visualization/model_training/forets_aleatoires_model.pkl"),
    "SVM (Support Vector Machines)": joblib.load(
        "sections/classification/classification_visualization/model_training/svm_(support_vector_machines)_model.pkl"),
    "Réseau de Neurones": joblib.load(
        "sections/classification/classification_visualization/model_training/reseau_de_neurones_model.pkl")
}

# Charger les données nettoyées et prétraitées
df = pd.read_csv('data/data_cleaned_preprocessed_reduced.csv')

# Séparer les features (X) et la cible (y)
X = df.drop(columns='target')  # X : Features explicatives
y = df['target']  # y : Variable cible

# Titre de l'application Streamlit
st.title("Prédiction avec les Modèles Entraînés")
st.write("Sélectionnez un modèle pour effectuer des prédictions sur les données.")

# Sélectionner un modèle
model_name = st.selectbox("Choisissez un modèle", list(models.keys()))

# Bouton pour lancer les prédictions
if st.button("Lancer les prédictions"):
    try:
        st.write(f"Prédiction avec le modèle : {model_name} en cours...")

        # Charger le modèle sélectionné
        model = models[model_name]

        # Prédictions des probabilités sur l'ensemble des données
        y_pred_proba = model.predict_proba(X)  # Probabilités des classes

        # Créer un DataFrame pour les probabilités et les classes prédites
        df_predictions = pd.DataFrame(y_pred_proba, columns=[f"Class_{i}" for i in range(y_pred_proba.shape[1])])
        df_predictions['Predicted_Class'] = df_predictions.idxmax(axis=1)  # Classe prédite
        df_predictions['Actual_Class'] = y  # Classe réelle

        # Sauvegarder les prédictions dans un fichier CSV
        predictions_file_path = f"{predictions_output_dir}/{model_name.replace(' ', '_').lower()}_predictions.csv"
        df_predictions.to_csv(predictions_file_path, index=False)

        st.write(f"Les prédictions ont été sauvegardées sous : {predictions_file_path}")

        # Affichage des prédictions
        st.write("Aperçu des prédictions :")
        st.dataframe(df_predictions.head())

    except Exception as e:
        st.error(f"Erreur lors de la prédiction avec le modèle {model_name} : {e}")

# Message de fin
st.write("\nPrédictions terminées pour tous les modèles.")
