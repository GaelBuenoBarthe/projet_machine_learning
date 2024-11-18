import pandas as pd
import keras
import numpy as np
import plotly.graph_objects as go
from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Fonction pour charger et préparer les données
def load_and_preprocess_data(file_path, test_size=0.3):
    df = pd.read_csv(file_path)

    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test


# Fonction pour construire et entraîner le modèle
def build_and_train_nn(X_train, y_train, hidden_layers, neurons_per_layer, epochs, batch_size, learning_rate):
    model = Sequential()

    model.add(layers.Dense(neurons_per_layer, input_dim=X_train.shape[1], activation='relu'))

    for _ in range(hidden_layers - 1):
        model.add(layers.Dense(neurons_per_layer, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    return model, history


# Fonction pour dessiner la structure du réseau de neurones
def draw_nn(layers, neurons_per_layer):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Positionner les neurones
    x_positions = range(len(layers))
    for i, layer in enumerate(layers):
        y_positions = np.linspace(-1, 1, neurons_per_layer[i])
        for y in y_positions:
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[y],
                mode="markers",
                marker=dict(size=20, color="blue"),
                name=f"Layer {i + 1}"
            ))

            # Ajouter des connexions
            if i > 0:
                prev_y_positions = np.linspace(-1, 1, neurons_per_layer[i - 1])
                for prev_y in prev_y_positions:
                    fig.add_trace(go.Scatter(
                        x=[x_positions[i - 1], x_positions[i]],
                        y=[prev_y, y],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        showlegend=False
                    ))

    fig.update_layout(
        title="Structure du Réseau de Neurones",
        xaxis=dict(title="Couches", showticklabels=False),
        yaxis=dict(title="Neurones", showticklabels=False),
        width=800,
        height=600
    )
    return fig