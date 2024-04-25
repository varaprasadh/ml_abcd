import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the existing model
model = load_model('model.h5')

# Take input from the user for additional training data
num_samples = int(input("Enter the number of additional samples: "))
additional_X = []
additional_y = []
for i in range(num_samples):
    a = float(input("Enter value for 'a': "))
    b = float(input("Enter value for 'b': "))
    additional_X.append([a, b])
    additional_y.append(a + b)
additional_X = np.array(additional_X)
additional_y = np.array(additional_y)

# Compile the model if necessary
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Continue training with additional data
model.fit(additional_X, additional_y, epochs=10, batch_size=32)

# Save the updated model
model.save('model.h5')
