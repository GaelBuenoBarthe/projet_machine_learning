import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError  # Pour gérer les erreurs de modèle non entraîné

# Fonction pour entraîner un modèle
def train_model(df, selected_model_name):
    # Séparation des variables explicatives et cible
    X = df.drop(columns='target')
    y = df['target']

    # Diviser en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création du modèle selon le choix
    model = None
    if selected_model_name == "Régression Logistique":
        max_iter = st.number_input("Nombre d'itérations (max_iter)", min_value=100, value=1000)
        model = LogisticRegression(max_iter=max_iter, random_state=42)
    elif selected_model_name == "Arbres de Décision":
        max_depth = st.number_input("Profondeur maximale des arbres", min_value=1, value=10)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif selected_model_name == "Forêts Aléatoires":
        n_estimators = st.number_input("Nombre d'arbres", min_value=10, value=100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif selected_model_name == "SVM (Support Vector Machines)":
        C_value = st.number_input("Valeur de C", min_value=0.01, value=1.0)
        model = SVC(C=C_value, probability=True, random_state=42)
    elif selected_model_name == "Réseau de Neurones":
        hidden_layer_sizes = st.number_input("Nombre de neurones dans la couche cachée", min_value=1, value=100)
        model = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), max_iter=1000, random_state=42)

    # Entraînement du modèle
    if model is not None:
        model.fit(X_train, y_train)
        st.success(f"Modèle {selected_model_name} entraîné avec succès.")

        # Sauvegarde du modèle
        model_path = f"model_{selected_model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        st.write(f"Modèle sauvegardé sous : {model_path}")

        # Évaluation du modèle
        result, fig = evaluate_model(model, X_test, y_test, selected_model_name)
        return model, X_test, y_test, selected_model_name, result, fig  # Inclure les résultats dans le retour

    return None, None, None, selected_model_name, None, None  # Si le modèle est None, renvoyer également None

# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test, model_name):
    if model is None:
        st.error("Le modèle n'a pas encore été entraîné.")
        return None, None  # Ajoutez un retour pour éviter l'erreur de décomposition

    # Vérification que le modèle a bien été entraîné avant de prédire
    try:
        y_pred = model.predict(X_test)
    except NotFittedError:
        st.error(f"Le modèle {model_name} n'a pas encore été entraîné.")
        return None, None

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Affichage des résultats
    st.subheader("Évaluation du modèle")
    st.write(f"Précision du modèle {model_name} : {accuracy:.2f}")
    st.text("Rapport de classification :")
    st.text(report)

    # Affichage de la matrice de confusion
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Retourner les résultats pour qu'ils puissent être utilisés ailleurs
    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": cm}, fig

# Charger les données à partir d'une session Streamlit
if 'df_resampled' in st.session_state:
    selected_model_name = st.selectbox("Choisissez le modèle à entraîner", [
        "Régression Logistique",
        "Arbres de Décision",
        "Forêts Aléatoires",
        "SVM (Support Vector Machines)",
        "Réseau de Neurones"
    ])
    model, X_test, y_test, model_name, result, fig = train_model(st.session_state.df_resampled, selected_model_name)

    if result is not None and fig is not None:
        st.write(f"Précision : {result['accuracy'] * 100:.2f}%")
        st.write("Rapport de classification :")
        st.text(result["classification_report"])

else:
    st.error("Les données ne sont pas chargées.")
