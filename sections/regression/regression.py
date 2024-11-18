import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sections.regression.data_management.data_management import load_and_preprocess_data
from sections.regression.data_visualization.data_visualization import visualize_preprocessed_data, visualize_data
from sections.regression.model.model import train_and_evaluate_models
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
    st.header("Comparaison des Modèles")
    file_path = 'data/diabete.csv'

    test_size = st.session_state.get('test_size', 0.3)

    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, test_size)
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    results_df['Model'] = results_df['Model'].apply(lambda x: str(x))

    st.subheader("Résultats des modèles")
    st.dataframe(results_df)

    selected_model = st.selectbox("Sélectionnez un modèle pour l'entraînement", results_df['Model'].tolist())

    selected_metrics = results_df[results_df['Model'] == selected_model].iloc[0]
    st.subheader(f"Métriques pour le modèle sélectionné : {selected_model}")
    st.write(f"R² (Test) : {selected_metrics['R² (Test)']}")
    st.write(f"MSE (Test) : {selected_metrics['MSE (Test)']}")

    # Vérification de la disponibilité de 'MSE (CV)' et 'Mean R² (CV)'
    if 'MSE (CV)' in selected_metrics:
        st.write(f"MSE (CV) : {selected_metrics['MSE (CV)']}")
    else:
        st.warning("MSE (CV) non calculé pour ce modèle.")

    if 'Mean R² (CV)' in selected_metrics:
        st.write(f"Mean R² (CV) : {selected_metrics['Mean R² (CV)']}")
    else:
        st.warning("Mean R² (CV) non calculé pour ce modèle.")

# Page d'entraînement du modèle
def model_training_page():
    st.title("Entraînement du Modèle Sélectionné")

    # Demander à l'utilisateur de sélectionner un modèle
    model_options = ["Lasso Regression", "Ridge Regression", "Linear Regression"]
    selected_model = st.selectbox("Sélectionnez un modèle", model_options)

    # Si le modèle sélectionné est Lasso Regression, demander la valeur de Alpha
    if selected_model == "Lasso Regression":
        selected_alpha = st.slider("Sélectionnez la valeur d'Alpha", min_value=0.001, max_value=10.0, value=0.1)
        selected_metrics = {'Alpha': selected_alpha}
    else:
        selected_metrics = {}

    # Vérification de la clé 'Alpha' dans selected_metrics
    if 'Alpha' in selected_metrics:
        alpha_value = selected_metrics['Alpha']
    else:
        st.error("La clé 'Alpha' n'a pas été trouvée dans selected_metrics.")
        alpha_value = None  # Vous pouvez aussi définir une valeur par défaut ou gérer l'erreur autrement

    # Si alpha_value est valide, entraîner le modèle
    if alpha_value is not None:
        # Charger les données
        file_path = 'data/diabete.csv'  # Chemin vers votre fichier CSV
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

        # Sélectionner le modèle en fonction de l'utilisateur
        if selected_model == "Lasso Regression":
            model = Lasso(alpha=alpha_value)
        elif selected_model == "Ridge Regression":
            model = Ridge(alpha=alpha_value)
        else:
            model = LinearRegression()

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions et calcul des métriques
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Affichage des résultats
        st.subheader(f"Métriques pour le modèle {selected_model}")
        st.write(f"R² (Test): {r2}")
        st.write(f"MSE (Test): {mse}")
        st.write(f"MAE (Test): {mae}")
    else:
        st.error("Le modèle n'a pas pu être entraîné, car 'Alpha' est manquant.")

    # Afficher le graphique de comparaison
    st.subheader("Visualisation des Prédictions")
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Prédictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Égalité')
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.title(f"{selected_model} - Valeurs Réelles vs Prédictions")
    plt.legend()
    st.pyplot(plt)


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