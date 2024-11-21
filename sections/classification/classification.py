import streamlit as st
import pandas as pd
from sections.classification.data_preprocessing.data_cleaning import nettoyage_donnees
from sections.classification.data_preprocessing.data_preprocessing import appliquer_smote, encoder_categoriel, mise_a_echelle
from sections.classification.data_preprocessing.feature_selection import feature_selection  # Assurez-vous d'importer votre fonction

# Fonction principale
def classification_page():
    st.header("Bienvenue dans l'application de Classification")
    st.caption("Cette application permet de classifier et de prétraiter vos données.")

    # Utilisation de radio buttons pour naviguer
    page = st.radio("Sélectionner une étape",
                    ["Accueil", "Nettoyage des données", "Prétraitement des données", "Appliquer SMOTE", "Sélection des features"])

    if page == "Accueil":
        # Affichage de la page d'accueil
        st.write("Bienvenue dans l'outil de classification !")
        st.write("Sélectionnez une étape dans la barre latérale pour commencer.")

    elif page == "Nettoyage des données":
        # Nettoyage des données
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
        # Prétraitement des données
        if 'df' in st.session_state and st.session_state.df is not None:
            if st.button("Démarrer le prétraitement des données"):
                st.write("Prétraitement des données en cours...")
                df = encoder_categoriel(st.session_state.df)
                df, X, target_columns = mise_a_echelle(df)
                st.write("Premières lignes des données après prétraitement :")
                st.write(df.head())

                # Affichage des fréquences des classes avant SMOTE
                for col in target_columns:
                    st.write(f"{col} : {df[col].sum()} ({df[col].sum() / len(df):.2%})")

    elif page == "Appliquer SMOTE":
        # Application de SMOTE
        if 'df' in st.session_state and st.session_state.df is not None:
            X = st.session_state.df.drop(columns=["target"])  # Assuming "target" is the column to predict
            target_columns = ['target']  # Example, adjust based on your data
            apply_smote = st.radio("Souhaitez-vous appliquer SMOTE ?", ["Non", "Oui"])

            if apply_smote == "Oui":
                try:
                    y = st.session_state.df[target_columns[0]]  # Prenez la première colonne cible
                    X_resampled, y_resampled = appliquer_smote(X, y)
                    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                    df_resampled['target'] = y_resampled
                    st.write("Données après rééchantillonnage avec SMOTE :")
                    st.write(df_resampled.head())

                    # Affichage des fréquences des classes après SMOTE
                    st.write("Fréquences des classes après SMOTE :")
                    for class_label, frequency in df_resampled['target'].value_counts(normalize=True).items():
                        st.write(f"Classe '{class_label}' : {frequency:.2f}%")

                    # Stocker le DataFrame rééchantillonné dans la session pour la prochaine étape
                    st.session_state.df_resampled = df_resampled

                except Exception as e:
                    st.error(f"Erreur lors de l'application de SMOTE : {str(e)}")

    elif page == "Sélection des features":
        # Sélection des fonctionnalités après SMOTE
        if 'df_resampled' in st.session_state and st.session_state.df_resampled is not None:
            st.write("Vérification des données disponibles :")
            st.write(st.session_state.df_resampled.head())  # Affiche les premières lignes du DataFrame rééchantillonné

            if st.button("Démarrer la sélection des features"):
                try:
                    # Appel à la fonction de sélection des features
                    df_reduced, k_best_features = feature_selection(st.session_state.df_resampled)

                    # Affichage des résultats après sélection des fonctionnalités
                    st.write("Données après sélection des fonctionnalités :")
                    st.write(df_reduced.head())  # Affiche les premières lignes du DataFrame après la sélection

                    # Affichage des meilleures fonctionnalités sélectionnées
                    st.write("Les meilleures fonctionnalités sélectionnées :")
                    st.write(k_best_features)  # Affiche les meilleures fonctionnalités

                except Exception as e:
                    # Gestion des erreurs et affichage d'un message d'erreur
                    st.error(f"Erreur lors de la sélection des fonctionnalités : {e}")
        else:
            # Si les données rééchantillonnées ne sont pas disponibles
            st.error("Les données rééchantillonnées (df_resampled) ne sont pas disponibles. Veuillez revenir à l'étape SMOTE.")

# Exécution de l'application
if __name__ == "__main__":
    classification_page()
