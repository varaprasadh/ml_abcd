import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

# Generate training data
num_samples = 10
a = np.random.rand(num_samples) * 10
b = np.random.rand(num_samples) * 10
X_train = np.column_stack((a, b))
y_train = a + b

# Define the neural network model
model = Sequential([
    Dense(10, input_shape=(2,), activation='relu'), # Input layer
    Dense(1) # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10000, batch_size=32)

model.save('model.h5')

# Evaluate the model on the training data
y_train_pred = model.predict(X_train)
r2 = r2_score(y_train, y_train_pred)
print("R^2 score on training data:", r2)


