import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

"""
************************************************************
                     Prétraitement des données
************************************************************
"""

"""
============================================================
                   7. Sauvegarde du DataFrame prétraité
============================================================
"""
def sauvegarder_donnees(df, fichier_sortie):
    """
    Fonction pour sauvegarder le DataFrame prétraité dans un fichier CSV.
    """
    df.to_csv(fichier_sortie, index=False)
    st.write(f"Jeu de données prétraité sauvegardé sous le nom '{fichier_sortie}'")

"""
============================================================
                   1. Importation des données
============================================================
"""
def importer_donnees(fichier):
    """
    Fonction pour importer les données à partir d'un fichier CSV.
    """
    df = pd.read_csv(fichier)
    st.write("Colonnes disponibles dans le DataFrame :", df.columns)
    return df

df = importer_donnees('data/data_cleaned.csv')

"""
============================================================
                   2. Encodage des variables catégorielles
============================================================
"""
def encoder_categoriel(df):
    """
    Fonction pour encoder les variables catégorielles.
    """
    if 'target' in df.columns:
        encoder = OneHotEncoder(sparse_output=False)

        # Encodage de la colonne 'target'
        encoded_data = encoder.fit_transform(df[['target']])

        # Convertir les données encodées en DataFrame
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['target']))

        # Ajouter les colonnes encodées et supprimer la colonne originale 'target'
        df = pd.concat([df, df_encoded], axis=1).drop(columns=['target'])
    else:
        st.error("Erreur : La colonne 'target' n'existe pas dans le DataFrame initial.")
    return df

df = encoder_categoriel(df)

"""
============================================================
                   3. Mise à l'échelle des données
============================================================
"""
def mise_a_echelle(df):
    """
    Fonction pour normaliser les données numériques.
    """
    target_columns = [col for col in df.columns if 'target' in col]
    X = df.drop(columns=target_columns).select_dtypes(include=['float64', 'int64'])

    # Vérifier si des colonnes numériques sont présentes avant d'appliquer la mise à l'échelle
    if X.empty:
        st.error("Erreur : Aucune colonne numérique n'est présente pour la mise à l'échelle.")
        return df, None, target_columns

    # Appliquer la mise à l'échelle des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remplacer les valeurs des colonnes indépendantes dans le DataFrame
    df[X.columns] = X_scaled
    return df, X, target_columns

df, X, target_columns = mise_a_echelle(df)

"""
============================================================
                   4. Analyse des classes
============================================================
"""
def analyser_classes(df, target_columns):
    """
    Fonction pour analyser la distribution des classes et regénérer la colonne 'target'.
    """
    # Vérification de la présence de la colonne 'target'
    if 'target' not in df.columns:
        st.error("Erreur : La colonne 'target' n'existe pas dans le DataFrame après traitement.")
        return df, None

    st.write("Fréquences des classes :")
    for col in target_columns:
        st.write(f"{col} : {df[col].sum()} ({df[col].sum() / len(df):.2%})")

    # Créer une seule colonne 'target' à partir des colonnes encodées
    y = df[target_columns].idxmax(axis=1)

    # Si les classes sont encodées sous forme de chaîne, ajuster le nom
    if y.str.contains('target_').any():
        df['target'] = y.str.replace('target_', '')
    else:
        df['target'] = y

    # Supprimer les colonnes encodées
    df = df.drop(columns=target_columns)
    return df, y

df, y = analyser_classes(df, target_columns)

"""
============================================================
                   5. Gestion des déséquilibres avec SMOTE
============================================================
"""
def appliquer_smote(X, y):
    """
    Applique SMOTE pour rééchantillonner les classes.
    """
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Affichage des fréquences et pourcentages avant l'application de SMOTE
if 'target' in df.columns:
    st.write("\nFréquence des classes avant SMOTE :")
    st.write(df['target'].value_counts())
    st.write("\nPourcentage des classes avant SMOTE :")
    st.write(df['target'].value_counts(normalize=True) * 100)
else:
    st.error("Erreur : La colonne 'target' n'est pas présente dans le DataFrame.")

# Demander si SMOTE doit être appliqué avec Streamlit
choix_smote = st.radio("Souhaitez-vous appliquer SMOTE ?", ("Non", "Oui"))

if choix_smote == "Oui":
    X_resampled, y_resampled = appliquer_smote(X, y)

    # Recréer le DataFrame avec les données rééchantillonnées
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['target'] = y_resampled

    # Calculer et afficher les nouvelles fréquences des classes après rééchantillonnage
    class_frequencies = df_resampled['target'].value_counts(normalize=True) * 100
    st.write("\nFréquences des classes après rééchantillonnage :")
    for class_label, frequency in class_frequencies.items():
        st.write(f"Classe '{class_label}' : {frequency:.2f}%")

    # Sauvegarder les données rééchantillonnées
    sauvegarder_donnees(df_resampled, "data/data_cleaned_preprocessed_smote.csv")
else:
    # Sauvegarder les données sans SMOTE
    sauvegarder_donnees(df, "data/data_cleaned_preprocessed.csv")
