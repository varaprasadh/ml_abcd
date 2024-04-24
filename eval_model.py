import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Take input from the user for one sample
a = float(input("Enter value for 'a': "))
b = float(input("Enter value for 'b': "))
test_data = np.array([[a, b]])

# Make prediction using the loaded model
prediction = model.predict(test_data)

# Print the prediction
print("Prediction:", prediction[0][0])
