import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Visualisation des données
def visualize_data(file_path):
    # Charger les données depuis le fichier CSV
    df = pd.read_csv(file_path)

    # Visualiser un aperçu des données
    st.subheader("Aperçu des Données")
    st.write(df)

    # Affichage des informations générales sur les données
    st.write(f"Nombre de lignes : {df.shape[0]}")
    st.write(f"Nombre de colonnes : {df.shape[1]}")
    st.write(df.describe())

# Visualisation des données après prétraitement
def visualize_preprocessed_data(file_path):
    df = pd.read_csv(file_path)

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    st.subheader("Aperçu des Données Prétraitées")
    st.write(df.head())

    st.subheader("Visualisation des Correlations après Prétraitement")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

