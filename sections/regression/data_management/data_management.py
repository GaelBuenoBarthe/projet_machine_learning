import pandas as pd
from sklearn.model_selection import train_test_split

# Fonction pour charger et préparer les données
def load_and_preprocess_data(file_path, test_size=0.2):
    df = pd.read_csv(file_path)

    # Séparer les features et la target
    X = df.drop(columns=['target'])
    y = df['target']

    # Séparation des données en jeux d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test