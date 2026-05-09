import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model_bunga.h5")

# Nama kelas
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load gambar
img_path = "test_bunga.jpg"
img = cv2.imread(img_path)

# Resize
img = cv2.resize(img, (32, 32))

# Normalisasi
img = img.astype("float32") / 255.0

# Tambah dimensi batch
img = np.expand_dims(img, axis=0)

# Prediksi
prediction = model.predict(img)

# Ambil kelas tertinggi
predicted_class = np.argmax(prediction)

print("Prediksi bunga:", class_names[predicted_class])
print("Confidence:", np.max(prediction) * 100, "%")

