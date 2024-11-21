import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import os

def feature_selection(data_file, output_dir="sections/classification/classification_visualization/feature_selection",
                      threshold=0.8, k_best=5):
    """
    Sélection des features à partir des données nettoyées et prétraitées.

    Arguments:
    - data_file: Chemin du fichier CSV contenant les données prétraitées.
    - output_dir: Répertoire pour sauvegarder les visualisations.
    - threshold: Seuil de corrélation pour filtrer les variables fortement corrélées.
    - k_best: Nombre de meilleures features à sélectionner avec ANOVA.

    Retour:
    - df_reduced: DataFrame réduit après la sélection des features.
    - k_best_features: Liste des meilleures features sélectionnées par ANOVA.
    """

    # Lire le fichier CSV contenant les données nettoyées et prétraitées
    df = pd.read_csv(data_file)

    # Vérifier si la colonne cible 'target' existe
    if 'target' not in df.columns:
        raise ValueError("La colonne 'target' est absente du jeu de données.")

    print(f"Colonnes disponibles dans le DataFrame : {df.columns}")

    # 1. Sélection des features par corrélation
    print("\nÉtape 1 : Sélection des features par corrélation...")

    # Calcul de la matrice de corrélation (sans la colonne cible 'target')
    print("Calcul de la matrice de corrélation...")
    corr_matrix = df.drop(columns=['target']).corr()
    print("Matrice de corrélation calculée.")

    # Afficher la matrice de corrélation sous forme de heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Matrice de Corrélation")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sauvegarder la heatmap de corrélation
    heatmap_path = f"{output_dir}/correlation_heatmap.pdf"
    plt.savefig(heatmap_path)
    print(f"Matrice de corrélation sauvegardée sous : {heatmap_path}")
    plt.close()

    # Identification des paires fortement corrélées
    print("Recherche des corrélations élevées...")
    high_corr_pairs = corr_matrix.where((abs(corr_matrix) > threshold) & (corr_matrix != 1)).stack()

    # Si des paires sont trouvées, les afficher et réduire les features
    if not high_corr_pairs.empty:
        print(f"{len(high_corr_pairs)} paires de variables fortement corrélées trouvées.")
        print(high_corr_pairs)

        # Sélection des colonnes à supprimer
        to_drop = set()
        for var1, var2 in high_corr_pairs.index:
            to_drop.add(var2)  # Supprimer une des deux variables corrélées
        print(f"Colonnes à supprimer : {to_drop}")

        # Réduction des features
        df_reduced = df.drop(columns=to_drop)
    else:
        print("Aucune paire fortement corrélée trouvée.")
        df_reduced = df.copy()

    # 2. Sélection univariée avec ANOVA
    print("\nÉtape 2 : Sélection univariée avec ANOVA...")

    # Séparer les features explicatives et la variable cible
    X = df_reduced.drop(columns='target')
    y = df_reduced['target']

    # Utilisation de SelectKBest avec le test ANOVA
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)

    # Créer un DataFrame pour afficher les scores des features
    scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })

    # Trier les features par score
    scores = scores.sort_values(by='Score', ascending=False)

    # Afficher les scores des features
    print("\nScores des features (ANOVA) :")
    print(scores)

    # Sélectionner les k meilleures features
    k_best_features = scores.head(k_best)['Feature'].values
    print(f"\nLes {k_best} meilleures features selon ANOVA sont : {k_best_features}")

    # Sauvegarder le DataFrame réduit après sélection des features
    reduced_data_path = f"{output_dir}/data_cleaned_preprocessed_reduced.csv"
    df_reduced.to_csv(reduced_data_path, index=False)
    print(f"\nJeu de données après sélection des features sauvegardé sous : {reduced_data_path}")

    # Sauvegarder les scores ANOVA
    anova_scores_path = f"{output_dir}/feature_scores_anova.csv"
    scores.to_csv(anova_scores_path, index=False)
    print(f"\nScores des features (ANOVA) sauvegardés sous : {anova_scores_path}")

    # Graphique des scores ANOVA
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Score', y='Feature', data=scores)
    plt.title("Importance des Features selon le test ANOVA")
    plt.xlabel("Score ANOVA")
    plt.ylabel("Feature")

    # Sauvegarder le graphique des scores ANOVA
    anova_barplot_path = f"{output_dir}/anova_feature_scores.pdf"
    plt.savefig(anova_barplot_path)
    print(f"Graphique des scores ANOVA sauvegardé sous : {anova_barplot_path}")
    plt.close()

    # Retourner les résultats
    return df_reduced, k_best_features
