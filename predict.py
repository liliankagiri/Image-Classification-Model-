
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("dog_cat_classifier.h5")

# Load and preprocess image
img_path = "your_image.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
result = model.predict(img_array)
prediction = "dog" if result[0][0] > 0.5 else "cat"

# Display result
print(f"Prediction: {prediction}")
plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.axis('off')
plt.show()
