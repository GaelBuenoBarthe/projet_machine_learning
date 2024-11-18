import pandas as pd

def preprocess_data(file_path):
    # Charger les données
    df = pd.read_csv(file_path)

    # Suppression de la colonne 'Unnamed: 0'
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Remplacer les valeurs manquantes
    df.fillna(df.mean(), inplace=True)

    # Supprimer les doublons
    df.drop_duplicates(inplace=True)

    return df