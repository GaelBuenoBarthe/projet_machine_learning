"""
************************************************************
                     Nettoyage des données
************************************************************
"""
# Importation des bibliothèques nécessaires
import pandas as pd

"""
============================================================
                   1. Importation des données
============================================================
"""


def nettoyage_donnees():
    """
    Fonction de nettoyage des données : suppression des colonnes inutiles, gestion des valeurs manquantes,
    correction des fautes d'orthographe, conversion en données catégorielles et suppression des doublons.
    Retourne le DataFrame nettoyé.
    """
    # Lire le fichier CSV avec un délimiteur spécifique ','
    df = pd.read_csv('data/vin.csv', sep=',')

    # Utiliser l'argument `index_col` pour traiter la première colonne comme un index (si nécessaire)
    # df = pd.read_csv('data/vin.csv', sep=',', index_col=0)

    """
    ============================================================
                   2. Exploration des données
    ============================================================
    """

    # Afficher les premières lignes pour vérifier l'importation des données
    print("Premières lignes du DataFrame :")
    print(df.head())

    # Afficher la liste des colonnes sous forme de tableau
    columns_df = pd.DataFrame(df.columns, columns=['Nom des Colonnes'])
    print("\nListe des colonnes :")
    print(columns_df)

    # Afficher les dimensions du DataFrame (nombre de lignes et de colonnes)
    print(f"\nLe DataFrame contient {df.shape[0]} lignes et {df.shape[1]} colonnes.")

    # Afficher la description statistique des données
    print("\nDescription statistique des données :")
    print(df.describe())

    # Afficher les informations sur le DataFrame (types de données et valeurs manquantes)
    print("\nInformations sur le DataFrame (types de données et valeurs manquantes) :")
    df.info()

    """
    ============================================================
                   3. Typologie des données
    ============================================================
    """

    # Description des variables quantitatives et qualitatives
    print("""
    Variables quantitatives (numériques) :

        - Alcohol : Teneur en alcool du vin (en %).
        - Malic Acid : Quantité d’acide malique (g/L), affecte l'acidité du vin.
        - Ash : Quantité de cendres, indicatif de la minéralité.
        - Alcalinity of Ash : Alcalinité des cendres, impacte la stabilité du vin.
        - Magnesium : Teneur en magnésium, important pour les propriétés du vin.
        - Total Phenols : Quantité totale de phénols, influence la couleur et la structure.
        - Flavonoids : Contenu en flavonoïdes, affecte la couleur et l'amertume.
        - Nonflavanoid Phenols : Phénols non flavonoïdes, influencent les propriétés sensorielles.
        - Proanthocyanins : Antioxydants responsables de la couleur et de l’astringence.
        - Color Intensity : Intensité de la couleur du vin, mesurée par absorption lumineuse.
        - Hue : Teinte du vin, déterminée par la longueur d’onde de lumière absorbée.
        - Od280/od315 of Diluted Wines : Rapport d'absorbance des composés phénoliques.
        - Proline : Teneur en proline, acide aminé, utilisé pour déterminer les conditions de culture.

    Variable qualitative (catégorielle) :

        - Target : Classe du vin (ex : "Vin amer", "Vin sucré" ou "Vin équilibré").
    """)

    """
    ============================================================
                   4. Nettoyage des données
    ============================================================
    """

    # ---------------------------------------------------------
    # 1. Suppression des colonnes inutiles
    # ---------------------------------------------------------
    # Supprimer la colonne 'Unnamed: 0' qui peut être présente dans les fichiers CSV
    df = df.drop(columns=["Unnamed: 0"])

    # Afficher les premières lignes pour vérifier la suppression de la colonne
    print("\nPremières lignes après suppression de la colonne 'Unnamed: 0' :")
    print(df.head())

    # ---------------------------------------------------------
    # 2. Gestion des valeurs manquantes (NA)
    # ---------------------------------------------------------
    # Afficher le nombre de NaN dans chaque colonne
    print("\nNombre de valeurs manquantes (NaN) dans chaque colonne :")
    print(df.isna().sum())

    # ---------------------------------------------------------
    # 3. Vérification des types de données
    # ---------------------------------------------------------
    # Vérifier les types de données de chaque colonne
    print("\nTypes de données de chaque colonne :")
    print(df.dtypes)

    # ---------------------------------------------------------
    # 4. Vérification et correction des fautes d'orthographe
    # ---------------------------------------------------------
    # Vérifier les valeurs uniques dans la colonne 'target' pour identifier la faute d'orthographe
    print("\nValeurs uniques avant correction dans la colonne 'target' :")
    print(df['target'].unique())

    # Remplacer la faute d'orthographe dans la colonne 'target'
    df['target'] = df['target'].replace('Vin éuilibré', 'Vin équilibré')

    # Vérifier après remplacement
    print("\nValeurs uniques après correction de la faute d'orthographe dans 'target' :")
    print(df['target'].unique())

    # ---------------------------------------------------------
    # 5. Conversion en données catégorielles
    # ---------------------------------------------------------
    # Conversion de la colonne 'target' en type catégoriel
    df['target'] = df['target'].astype('category')

    # Vérifier le type de la colonne après conversion
    print("\nType de la colonne 'target' après conversion en catégorielle :")
    print(df['target'].dtypes)

    # Afficher les catégories disponibles
    print("\nCatégories uniques dans la colonne 'target' :")
    print(df['target'].cat.categories)

    # ---------------------------------------------------------
    # 6. Gestion des doublons
    # ---------------------------------------------------------
    # Vérifier et afficher le nombre de lignes dupliquées dans le DataFrame
    print("\nNombre de doublons dans le DataFrame :", df.duplicated().sum())

    # Supprimer les doublons pour nettoyer les données
    df = df.drop_duplicates()

    # Vérification finale
    print("\nDimensions finales du DataFrame après nettoyage :")
    print(df.shape)

    # ---------------------------------------------------------
    # 7. Sauvegarde du DataFrame
    # ---------------------------------------------------------
    # Sauvegarder le DataFrame dans un fichier CSV pour utilisation future
    df.to_csv("data/data_cleaned.csv", index=False)
    print("Jeu de données nettoyé sauvegardé sous le nom 'data_cleaned.csv'")

    # Sauvegarder le DataFrame sous format pickle
    df.to_pickle("data/data_cleaned.pkl")
    print("Jeu de données nettoyé sauvegardé sous le nom 'data_cleaned.pkl'")

    return df
