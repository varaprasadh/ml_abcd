import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable

# Generate training data
num_samples = 1000000

# Generate random input numbers and operations
X_numbers = np.random.randint(0, 1000, (num_samples, 2))
X_operations = np.random.randint(0, 4, (num_samples, 1))  # 0: add, 1: sub, 2: mul, 3: div



@register_keras_serializable()
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# Function to perform the specified operation
def perform_operation(numbers, ops):
    results = []
    for num_pair, op in zip(numbers, ops):
        if op == 0:  # Addition
            result = np.sum(num_pair)
        elif op == 1:  # Subtraction
            result = np.abs(num_pair[0] - num_pair[1])
        elif op == 2:  # Multiplication
            result = np.prod(num_pair)
        elif op == 3:  # Division
            result = num_pair[0] / max(num_pair[1], 1)  # Avoid division by zero
        results.append(result)
    return np.array(results)

# Generate corresponding labels
y_train = perform_operation(X_numbers, X_operations)

# Combine input features
X_train = np.concatenate((X_numbers, X_operations), axis=1)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(3,), activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam',
              loss='mse',  # Mean Squared Error for regression task
              metrics=['mae'])  # Mean Absolute Error

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32)


#save the model
model.save('basic_algebra_model.h5')

# Test the model
test_input_numbers = np.array([[5, 10]])  # Numbers to multiply: 5 * 10
test_input_operation = np.array([[2]])  # Operation: 2 (multiplication)
test_input = np.concatenate((test_input_numbers, test_input_operation), axis=1)
prediction = model.predict(test_input)
print(f"The predicted result is: {prediction[0][0]}")

