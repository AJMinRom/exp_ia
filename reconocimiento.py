pip install opencv-python numpy scikit-learn
from sklearn.model_selection import train_test_split
import cv2

# Cargar las imágenes etiquetadas
images = load_images('dogs_vs_cats.h5')
labels = extract_labels(images)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)



# Normalizar los píxeles
X_train = normalize_pixels(X_train)
X_test = normalize_pixels(X_test)

# Extraer características utilizando HOG
hog = cv2.HOGDescriptor()
X_train_hog = hog.compute(X_train)
X_test_hog = hog.compute(X_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# Construir la CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
