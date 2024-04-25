import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')


# take the input from user
test_number = int(input("Enter a number: "))

# Test the model
prediction = model.predict(np.array([[test_number]]))
prediction_label = "odd" if prediction[0][0] > 0.5 else "even"
print(f"The number {test_number} is predicted to be {prediction_label}.")



