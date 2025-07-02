# 🐶🐱 CNN Image Classifier - Dogs vs Cats

This project demonstrates an image classification model using a **Convolutional Neural Network (CNN)** to distinguish between **dogs and cats**. Built with TensorFlow and trained on image data, the model can accurately predict whether a new image is a dog or a cat.

---

## 📁 Dataset

The dataset consists of:
- Training images: `/training_set/training_set/`
- Validation images: `/test_set/test_set/`

Each folder contains subfolders:
- `/dogs/`
- `/cats/`

The images were sourced from [Dropbox links] and automatically unzipped in Colab.

---

## 🧠 Model Architecture

The model is a **custom CNN** built using the Keras Sequential API:

- `Conv2D` layer with ReLU activation  
- `MaxPooling2D` for downsampling  
- `Flatten` layer to flatten 2D features  
- Fully connected `Dense` layers  
- `Sigmoid` output for binary classification (Dog vs Cat)

---

## ⚙️ Features

- 📦 Data loading and preprocessing with `ImageDataGenerator`
- 🧪 Training with `model.fit()`
- 📈 Accuracy and loss visualization using Matplotlib
- 🔍 External image upload and prediction in Colab
- 📤 Easy deployment on GitHub

---

## 🧪 How to Use

### 🧑‍💻 Run in Google Colab:
1. Clone or open the notebook in Colab.
2. Download and unzip the dataset using `wget` and `unzip` commands.
3. Train the model using `model.fit()`.
4. Upload your own image using:
   ```python
   from google.colab import files
   uploaded = files.upload()

---

plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
These plots help visualize how the model improves with training and whether it generalizes well to unseen validation data.

📷 Example Output
<!-- Replace with image link or GitHub-hosted file -->

🔍 Sample Prediction Code
python
Copy
Edit
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img = image.load_img('your_image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

result = model.predict(img_array)
prediction = 'dog' if result[0][0] > 0.5 else 'cat'

plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.axis('off')
plt.show()
✅ Dependencies
TensorFlow

NumPy

Matplotlib

Keras (via TensorFlow)
