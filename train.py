import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Générer des données d'exemple
data = np.array([[i for i in range(100)]])
target = np.array([[i for i in range(1, 101)]])

data = np.reshape(data, (1, 1, 100))
target = np.reshape(target, (1, 100))

# Créer le modèle LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 100)))
model.add(Dense(100))

# Compiler le modèle
model.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
model.fit(data, target, epochs=1000, verbose=2)

# Générer des prédictions
test_input = np.array([[i for i in range(100)]])
test_input = np.reshape(test_input, (1, 1, 100))
predictions = model.predict(test_input, verbose=0)

print(predictions)
