import streamlit as st
import pandas as pd
from sections.classification.data_preprocessing.data_cleaning import nettoyage_donnees
from sections.classification.data_preprocessing.data_preprocessing import appliquer_smote, encoder_categoriel, mise_a_echelle
from sections.classification.data_preprocessing.feature_selection import feature_selection  # Assurez-vous d'importer votre fonction
from sections.classification.models.train_models import train_model, evaluate_model  # Importation de nos fonctions

# Fonction principale
def classification_page():
    st.header("Bienvenue dans l'application de Classification")
    st.caption("Cette application permet de classifier et de prétraiter vos données.")

    # Utilisation de radio buttons pour naviguer
    page = st.radio("Sélectionner une étape",
                    ["Accueil", "Nettoyage des données", "Prétraitement des données", "Appliquer SMOTE", "Sélection des features", "Entraîner un modèle"])

    if page == "Accueil":
        st.write("Bienvenue dans l'outil de classification !")
        st.write("Sélectionnez une étape dans la barre latérale pour commencer.")

    elif page == "Nettoyage des données":
        if 'df' not in st.session_state:
            st.session_state.df = None  # Initialiser la variable si elle n'existe pas

        if st.session_state.df is None:
            if st.button("Démarrer le nettoyage des données"):
                st.write("Nettoyage des données en cours...")
                df = nettoyage_donnees()  # Vous pouvez récupérer vos données ici
                st.session_state.df = df
                st.write("Premières lignes du DataFrame nettoyé :")
                st.write(df.head())

    elif page == "Prétraitement des données":
        if 'df' in st.session_state and st.session_state.df is not None:
            if st.button("Démarrer le prétraitement des données"):
                st.write("Prétraitement des données en cours...")
                df = encoder_categoriel(st.session_state.df)
                df, X, target_columns = mise_a_echelle(df)
                st.write("Premières lignes des données après prétraitement :")
                st.write(df.head())

                # Affichage des colonnes cibles et de leur fréquence
                for col in target_columns:
                    st.write(f"{col} : {df[col].sum()} ({df[col].sum() / len(df):.2%})")

    elif page == "Appliquer SMOTE":
        if 'df' in st.session_state and st.session_state.df is not None:
            X = st.session_state.df.drop(columns=["target"])  # Assuming "target" is the column to predict
            target_columns = ['target']  # Exemple, ajustez selon vos données
            apply_smote = st.radio("Souhaitez-vous appliquer SMOTE ?", ["Non", "Oui"])

            if apply_smote == "Oui":
                try:
                    y = st.session_state.df[target_columns[0]]  # Prenez la première colonne cible
                    X_resampled, y_resampled = appliquer_smote(X, y)
                    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                    df_resampled['target'] = y_resampled
                    st.write("Données après rééchantillonnage avec SMOTE :")
                    st.write(df_resampled.head())

                    st.write("Fréquences des classes après SMOTE :")
                    for class_label, frequency in df_resampled['target'].value_counts(normalize=True).items():
                        st.write(f"Classe '{class_label}' : {frequency:.2f}%")

                    st.session_state.df_resampled = df_resampled

                except Exception as e:
                    st.error(f"Erreur lors de l'application de SMOTE : {str(e)}")

    elif page == "Sélection des features":
        if 'df_resampled' in st.session_state and st.session_state.df_resampled is not None:
            st.write("Vérification des données disponibles :")
            st.write(st.session_state.df_resampled.head())  # Affiche les premières lignes du DataFrame rééchantillonné

            if st.button("Démarrer la sélection des features"):
                try:
                    if 'target' not in st.session_state.df_resampled.columns:
                        st.error("La colonne 'target' est absente des données rééchantillonnées.")
                    else:
                        df_reduced, k_best_features = feature_selection(
                            st.session_state.df_resampled,
                            output_dir="sections/classification/classification_visualization/feature_selection"
                        )
                        st.write("Données après sélection des fonctionnalités :")
                        st.write(df_reduced.head())
                        st.write("Les meilleures fonctionnalités sélectionnées :")
                        st.write(k_best_features)
                except Exception as e:
                    st.error(f"Erreur lors de la sélection des fonctionnalités : {str(e)}")

    elif page == "Entraîner un modèle":
        if 'df_resampled' in st.session_state:
            # Sélectionner un modèle
            model_options = ["Régression Logistique", "Arbres de Décision", "Forêts Aléatoires",
                             "SVM (Support Vector Machines)", "Réseau de Neurones"]
            selected_model = st.selectbox("Choisissez le modèle à entraîner", model_options)

            if st.button(f"Entraîner le modèle {selected_model}"):
                # Appeler la fonction pour entraîner le modèle
                model, X_test, y_test, model_name, result, fig = train_model(st.session_state.df_resampled,
                                                                             selected_model)

                if result is not None and fig is not None:
                    st.write(f"Précision : {result['accuracy'] * 100:.2f}%")
                    st.write("Rapport de classification :")
                    st.text(result["classification_report"])

                    st.write("Matrice de confusion :")
                    st.pyplot(fig)
                else:
                    st.error("L'évaluation du modèle a échoué.")
