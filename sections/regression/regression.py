import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sections.regression.data_management.data_management import load_and_preprocess_data
from sections.regression.data_visualization.data_visualization import visualize_preprocessed_data, visualize_data
from sections.regression.model.model import train_and_evaluate_models, get_model
from sections.regression.model.neural_network import draw_nn



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
    st.title("Réseau de Neurones Interactif")

    # Générer des données artificielles
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Paramètres dynamiques
    layers = st.slider("Nombre de couches", min_value=2, max_value=10, value=3)
    neurons_per_layer = [st.slider(f"Neurones dans la couche {i+1}", 2, 20, 5) for i in range(layers)]
    learning_rate = st.slider("Taux d'apprentissage (learning rate)", 0.001, 0.1, 0.01, step=0.001)
    max_iter = st.slider("Nombre d'itérations", 10, 500, 100, step=10)

    # Afficher le réseau
    fig = draw_nn(range(layers), neurons_per_layer)
    st.plotly_chart(fig)

    # Entraînement du modèle
    st.write("Entraînement du modèle en cours...")
    progress_bar = st.progress(0)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(neurons_per_layer),
        learning_rate_init=learning_rate,
        max_iter=1,  # Itérations par étape
        warm_start=True,  # Conserver les paramètres pour continuer l'entraînement
        random_state=42
    )

    train_losses = []
    test_r2_scores = []

    for i in range(max_iter):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Suivi des métriques
        train_loss = mean_squared_error(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        train_losses.append(train_loss)
        test_r2_scores.append(test_r2)

        progress_bar.progress((i + 1) / max_iter)

    st.success("Entraînement terminé !")

    # Résultats finaux
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"R² : {r2:.4f}")
    st.write(f"MSE : {mse:.4f}")
    st.write(f"MAE : {mae:.4f}")

    # Courbe d'apprentissage
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Perte (loss) - Entraînement")
    ax.plot(test_r2_scores, label="R² - Test")
    ax.set_title("Courbe d'apprentissage")
    ax.set_xlabel("Itérations")
    ax.set_ylabel("Metrics")
    ax.legend()
    st.pyplot(fig)

    # Scatter plot : prédictions vs réalité
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    ax.set_title("Prédictions vs Réalité")
    ax.set_xlabel("Vérités terrain (y_test)")
    ax.set_ylabel("Prédictions (y_pred)")
    st.pyplot(fig)

    # Histogramme des résidus
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title("Distribution des résidus")
    ax.set_xlabel("Résidus")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)