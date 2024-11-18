import streamlit as st
from sections.classification.classification import classification_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import (
    data_visualization_page, preprocessed_data_visualization_page,
    data_management_page, model_training_page, model_comparaison_page, neural_network_page
)

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    regression_section = st.sidebar.radio(
        "Choisissez une section de régression",
        ["Visualisation des Données", "Visualisation des Données après Prétraitement",
         "Gestion des Données", "Comparaison des Modèles", "Entraînement du Modèle", "Réseau de neurones"]
    )
    if regression_section == "Visualisation des Données":
        data_visualization_page()
    elif regression_section == "Visualisation des Données après Prétraitement":
        preprocessed_data_visualization_page()
    elif regression_section == "Gestion des Données":
        data_management_page()
    elif regression_section == "Comparaison des Modèles":
        model_comparaison_page()
    elif regression_section == "Entraînement du Modèle":
        model_training_page()
    elif regression_section == "Réseau de neurones":
        neural_network_page()
elif type_data == "Classification":
    classification_page()
elif type_data == "NailsDetection":
    nail_page()
else:
    st.write("Choisissez une option")