import numpy as np
import tensorflow as tf
from tensorflow import keras

sample_size = 10000
# Generate training data
X_train = np.array([[i] for i in range(sample_size)])  # Numbers from 0 to 999
y_train = np.array([[i % 2] for i in range(sample_size)])  # Labels: 0 if even, 1 if odd

# Define the model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)


# Save the model
model.save('model.h5')