import streamlit as st

# Configuration de la page en premier
st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import (
    data_visualization_page, preprocessed_data_visualization_page,
    data_management_page, model_training_page, model_comparaison_page, neural_network_page
)

# Sidebar pour choisir le type de playground
type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Accueil", "Regression", "Classification", "NailsDetection"],
    index=0
)

# Dictionnaire de sections pour la régression
regression_sections = {
    "Visualisation des Données": data_visualization_page,
    "Visualisation des Données après Prétraitement": preprocessed_data_visualization_page,
    "Gestion des Données": data_management_page,
    "Comparaison des Modèles": model_comparaison_page,
    "Entraînement du Modèle": model_training_page,
    "Réseau de neurones": neural_network_page
}

if type_data == "Accueil":
    st.write("Bienvenue sur le Playground ML. Veuillez sélectionner une option dans la barre latérale.")
elif type_data == "Regression":
    regression_section = st.sidebar.radio(
        "Choisissez une section de régression",
        list(regression_sections.keys())
    )
    regression_sections[regression_section]()  # Appel de la fonction correspondante
elif type_data == "Classification":
    classification_page()
elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")