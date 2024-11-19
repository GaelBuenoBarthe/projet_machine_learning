import pandas as pd
import keras
import numpy as np
import plotly.graph_objects as go
from keras import Sequential
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scikeras.wrappers import KerasRegressor



l2 = keras.regularizers.l2
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam

# Fonction pour charger et préparer les données
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


# Fonction pour construire et entraîner le modèle avec régularisation L2 et Dropout
def build_and_train_nn(X_train, y_train, X_val, y_val, layers, neurons_per_layer, max_iter, batch_size, learning_rate, progress_bar):
    input_dim = X_train.shape[1]
    model = create_keras_model(layers, neurons_per_layer, learning_rate, input_dim)()
    r2_history = []

    class R2History(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            y_pred = self.model.predict(X_val).flatten()
            r2 = r2_score(y_val, y_pred)
            r2_history.append(r2)
            progress_bar.progress((epoch + 1) / max_iter)

    history = model.fit(X_train, y_train, epochs=max_iter, batch_size=batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=[R2History()])
    return model, history, r2_history

# Fonction pour créer le modèle Keras
def create_keras_model(layers, neurons_per_layer, learning_rate, input_dim):
    def model_fn():
        model = Sequential()
        for i in range(layers):
            if i == 0:
                model.add(Dense(neurons_per_layer[i], input_dim=input_dim, activation='relu'))
            else:
                model.add(Dense(neurons_per_layer[i], activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model
    return model_fn

# Fonction pour dessiner la structure du réseau de neurones
def draw_nn(layers, neurons_per_layer):
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
