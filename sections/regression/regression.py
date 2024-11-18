import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sections.regression.data_management.data_management import load_and_preprocess_data
from sections.regression.data_visualization.data_visualization import visualize_preprocessed_data, visualize_data
from sections.regression.model.model import train_and_evaluate_models, get_model
from sections.regression.model.neural_network import build_and_train_nn



def regression_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans le Playground de Régression linéaire de données sur le diabète")


def data_visualization_page():
    st.header("Visualisation des Données")
    file_path = 'data/diabete.csv'
    visualize_data(file_path)


def preprocessed_data_visualization_page():
    st.header("Visualisation des Données après Prétraitement")
    file_path = 'data/diabete.csv'
    visualize_preprocessed_data(file_path)


def data_management_page():
    st.header("Gestion des Données")
    file_path = 'data/diabete.csv'

    test_size = st.slider("Choisissez la proportion de test", 0.1, 0.5, 0.2)
    st.write(f"Ratio de test sélectionné: {test_size}")
    st.session_state['test_size'] = test_size

    try:
        st.write("Chargement et traitement de la data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, test_size)
        st.write("Les données ont été chargées et prétraitées.")
        st.write("Forme de X_train :", X_train.shape)
        st.write("Forme de X_test :", X_test.shape)
        st.write("Forme de y_train :", y_train.shape)
        st.write("Forme de y_test :", y_test.shape)
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        st.write("Détails de l'erreur:", str(e))


def model_comparaison_page():
    # Chargement des données
    file_path = 'data/diabete.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Choix des modèles
    model_choices = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'Gradient Boosting']

    # Créer une liste pour stocker les résultats
    model_results = []

    # Affichage des réglages pour chaque modèle
    for model_choice in model_choices:
        with st.expander(f"Réglages pour {model_choice}"):
            params = {}

            if model_choice == 'Random Forest' or model_choice == 'Gradient Boosting':
                params['n_estimators'] = st.slider(f'Nombre d\'estimations pour {model_choice}', min_value=10,
                                                   max_value=500, value=100, step=10)
            if model_choice == 'Random Forest':
                params['max_depth'] = st.slider(f'Profondeur maximale pour {model_choice}', min_value=1, max_value=20,
                                                value=5)
            if model_choice == 'Gradient Boosting':
                params['learning_rate'] = st.slider(f'Taux d\'apprentissage pour {model_choice}', min_value=0.01,
                                                    max_value=1.0, value=0.1, step=0.01)
            if model_choice in ['Ridge Regression', 'Lasso Regression']:
                params['alpha'] = st.slider(f'Paramètre alpha pour {model_choice}', min_value=0.01, max_value=10.0,
                                            value=1.0, step=0.01)

            # Bouton pour lancer l'entraînement du modèle
            if st.button(f"Lancer l'entraînement pour {model_choice}"):
                result = train_and_evaluate_models(X_train, X_test, y_train, y_test, model_choice, **params)
                model_results.append(result)
                st.dataframe(result)

    # Résumé des performances des modèles
    if model_results:
        all_results = pd.concat(model_results, ignore_index=True)
        st.write("Résumé des performances des modèles")
        st.dataframe(all_results)

        # Enregistrer le meilleur modèle en session pour l'entraînement dans la page suivante
        best_model_row = all_results.loc[all_results['R²'].idxmax()]  # Sélectionner le modèle avec le meilleur R²
        st.session_state.best_model = best_model_row['Model']
        st.session_state.best_model_params = best_model_row['Best Params']

        st.write(f"Le meilleur modèle sélectionné est : {st.session_state.best_model}")
        st.write(f"Paramètres du meilleur modèle : {st.session_state.best_model_params}")
    else:
        st.write("Aucun modèle n'a été testé.")


# Page d'entraînement du modèle
def model_training_page():
    # Vérifier si un modèle a été sélectionné
    if 'best_model' not in st.session_state:
        st.warning("Aucun modèle sélectionné. Veuillez sélectionner un modèle dans la page de comparaison.")
        return

    # Accéder au modèle et aux paramètres depuis la session
    model = get_model(st.session_state.best_model, **st.session_state.best_model_params)

    # Charger les données
    file_path = 'data/diabete.csv'
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    best_model = st.session_state.best_model
    best_model_params = st.session_state.best_model_params

    # Entraîner le meilleur modèle avec les paramètres sélectionnés
    st.write(f"Entraînement du modèle : {best_model} avec les paramètres : {best_model_params}")

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"R² : {r2}")
    st.write(f"MSE : {mse}")
    st.write(f"MAE : {mae}")

    # Scatter Plot pour Prédictions vs Vérités Terrain
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Prédictions")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ligne idéale")
    ax.set_title('Prédictions vs Vérités Terrain')
    ax.set_xlabel('Vérités Terrain (y_test)')
    ax.set_ylabel('Prédictions (y_pred)')
    ax.legend()
    st.pyplot(fig)

    # Scatter Plot pour Résidus
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, residuals, color='purple', alpha=0.5)
    ax.axhline(0, color='red', linestyle='--', label="0 (Pas de Résidu)")
    ax.set_title('Résidus vs Vérités Terrain')
    ax.set_xlabel('Vérités Terrain (y_test)')
    ax.set_ylabel('Résidus (y_test - y_pred)')
    ax.legend()
    st.pyplot(fig)

# Fonction de la page principale du réseau neuronal
def neural_network_page():
    st.title("Entraînement du Réseau de Neurones")

    # Charger et préparer les données
    file_path = 'data/diabete.csv'  # Path vers votre dataset
    test_size = st.slider("Choisissez la proportion de test", 0.1, 0.5, 0.2)
    st.write(f"Ratio de test sélectionné: {test_size}")

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, test_size)
        st.write("Les données ont été prétraitées avec succès.")
        st.write(f"Forme de X_train : {X_train.shape}")
        st.write(f"Forme de X_test : {X_test.shape}")
        st.write(f"Forme de y_train : {y_train.shape}")
        st.write(f"Forme de y_test : {y_test.shape}")
    except Exception as e:
        st.error(f"Erreur lors du prétraitement des données: {e}")
        return

    # Paramètres du modèle
    hidden_layers = st.slider("Nombre de couches cachées", 1, 5, 2)
    neurons_per_layer = st.slider("Nombre de neurones par couche", 10, 100, 50)
    epochs = st.slider("Nombre d'époques", 10, 200, 50)
    batch_size = st.slider("Taille du batch", 16, 128, 32)
    learning_rate = st.slider("Taux d'apprentissage", 0.0001, 0.01, 0.001)

    # Construction et entraînement du modèle
    model, history = build_and_train_nn(X_train, y_train, hidden_layers, neurons_per_layer, epochs, batch_size, learning_rate)

    # Prédictions du modèle
    predictions = model.predict(X_test)
    if predictions.ndim > 1:
        predictions = predictions.ravel()

    # Calcul des résidus
    residuals = y_test - predictions

    # Calcul des métriques
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Affichage des résultats
    st.subheader("Résultats du Modèle")
    st.write(f"R² (Test) : {r2}")
    st.write(f"MSE (Test) : {mse}")

    # Visualisation des courbes d'entraînement
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Perte d\'Entraînement')
    ax.plot(history.history['val_loss'], label='Perte de Validation')
    ax.set_title('Courbes de Perte')
    ax.set_xlabel('Époques')
    ax.set_ylabel('Perte')
    ax.legend()
    st.pyplot(fig)

    # Visualisation des résidus
    residuals = y_test - predictions
    st.subheader("Distribution des Résidus")
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50, color='blue')
    plt.title('Distribution des Résidus')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    st.pyplot()

    # Visualisation des prédictions vs. valeurs réelles
    st.subheader("Prédictions vs Réelles")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Prédictions vs. Réelles')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Prédictions')
    st.pyplot()