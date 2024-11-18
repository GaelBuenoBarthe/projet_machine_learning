from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Modèles
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge()),
        "Lasso Regression": make_pipeline(StandardScaler(), Lasso()),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet()),
        "Polynomial Regression (degree=2)": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    # Recherche des meilleurs hyperparamètres
    # Ridge Regression
    ridge_params = {'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    ridge_grid = GridSearchCV(models["Ridge Regression"], ridge_params, cv=5, scoring='r2')
    ridge_grid.fit(X_train, y_train)
    models["Ridge Regression"] = ridge_grid.best_estimator_

    # Lasso Regression
    lasso_params = {'lasso__alpha': [0.001, 0.01, 0.1, 1.0]}
    lasso_grid = GridSearchCV(models["Lasso Regression"], lasso_params, cv=5, scoring='r2')
    lasso_grid.fit(X_train, y_train)
    models["Lasso Regression"] = lasso_grid.best_estimator_

    # ElasticNet
    elastic_params = {'elasticnet__alpha': [0.001, 0.01, 0.1, 1.0], 'elasticnet__l1_ratio': [0.2, 0.5, 0.8]}
    elastic_grid = GridSearchCV(models["ElasticNet"], elastic_params, cv=5, scoring='r2')
    elastic_grid.fit(X_train, y_train)
    models["ElasticNet"] = elastic_grid.best_estimator_

    # Random Forest
    rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=5, scoring='r2')
    rf_grid.fit(X_train, y_train)
    models["Random Forest"] = rf_grid.best_estimator_

    # Gradient Boosting
    gb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 4, 5]}
    gb_grid = GridSearchCV(GradientBoostingRegressor(), gb_params, cv=5, scoring='r2')
    gb_grid.fit(X_train, y_train)
    models["Gradient Boosting"] = gb_grid.best_estimator_

    # Résultats des modèles
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul des métriques
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=["r2", "neg_mean_squared_error"])
        mean_cv_r2 = np.mean(cv_results["test_r2"])
        mean_cv_mse = -np.mean(cv_results["test_neg_mean_squared_error"])

        results.append({
            "Model": name,
            "R² (Test)": r2,
            "MSE (Test)": mse,
            "MAE (Test)": mae,
            "Mean R² (CV)": mean_cv_r2,
            "Mean MSE (CV)": mean_cv_mse,
            "Best Params": model.get_params() if hasattr(model, "get_params") else None,
            "Model Object": model
        })

    # Tri des résultats par R² (Test)
    results_df = pd.DataFrame(results).sort_values(by="R² (Test)", ascending=False)
    return results_df

# Chargement des données
file_path = 'data/diabete.csv'  # Chemin vers votre fichier CSV
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Entraînement des modèles et affichage des résultats
results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
print(results_df)
