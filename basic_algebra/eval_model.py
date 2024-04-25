import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('basic_algebra_model.h5')

# take input from user
a = int(input("Enter value for 'a': "))
b = int(input("Enter value for 'b': "))
operator = int(input("Enter operator (0: add, 1: sub, 2: mul, 3: div): "))

# Test the model
test_data = np.array([[a, b, operator]])
prediction = model.predict(test_data)

# Print the prediction
print("Prediction:", prediction[0][0])
