from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

import pandas as pd
import numpy as np

# Chargement et prétraitement des données
def load_and_preprocess_data(file_path, test_size=0.2):
    df = pd.read_csv(file_path)

    # Séparer les features et la target
    X = df.drop(columns=['target'])
    y = df['target']

    # Séparation des données en jeux d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Entraînement et évaluation des modèles
def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_choice, **params):
    # Sélection du modèle en fonction du choix
    if model_choice == 'Linear Regression':
        model = LinearRegression()
    elif model_choice == 'Ridge Regression':
        model = Ridge(alpha=params.get('alpha', 1.0))  # Exemple d'hyperparamètre pour Ridge
    elif model_choice == 'Lasso Regression':
        model = Lasso(alpha=params.get('alpha', 1.0))  # Exemple d'hyperparamètre pour Lasso
    elif model_choice == 'Random Forest':
        model = RandomForestRegressor(n_estimators=params.get('n_estimators', 100),
                                      max_depth=params.get('max_depth', None))
    elif model_choice == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100),
                                          learning_rate=params.get('learning_rate', 0.1),
                                          max_depth=params.get('max_depth', 3))

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions et évaluation des performances
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Résultats sous forme de dictionnaire
    result = {
        'Model': model_choice,
        'MSE': mse,
        'R²': r2,
        'Best Params': params
    }

    # Retourner les résultats sous forme de DataFrame
    return pd.DataFrame([result])

# retourne le modèle en fonction du choix
def get_model(model_name, **params):
    if model_name == 'Linear Regression':
        return LinearRegression()
    elif model_name == 'Ridge Regression':
        return Ridge(alpha=params.get('alpha', 1.0))
    elif model_name == 'Lasso Regression':
        return Lasso(alpha=params.get('alpha', 1.0))
    elif model_name == 'Random Forest':
        return RandomForestRegressor(n_estimators=params.get('n_estimators', 100),
                                     max_depth=params.get('max_depth', None))
    elif model_name == 'Gradient Boosting':
        return GradientBoostingRegressor(n_estimators=params.get('n_estimators', 100),
                                         learning_rate=params.get('learning_rate', 0.1),
                                         max_depth=params.get('max_depth', 3))
    else:
        raise ValueError(f"Modèle {model_name} non reconnu.")