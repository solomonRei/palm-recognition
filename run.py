import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt

M = 408
N = 306
input_shape = (M, N, 3)

model_path = './models/model_checkpoint.h5'
siamese_model = load_model(model_path)

def load_and_preprocess_image(image_path, mean_rgb):
    print(f"Загрузка изображения: {image_path}")
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    print(f"Оригинальные значения изображения (до предобработки):\n{image.numpy()[:5, :5, :]}")

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, mean_rgb)
    image = tf.image.resize(image, [M, N])

    print(f"Предобработанные значения изображения:\n{image.numpy()[:5, :5, :]}")

    plt.imshow(image)
    plt.title(f"Предобработанное изображение: {image_path}")
    plt.show()

    return image


mean_rgb = np.array([0.641855879, 0.523251229, 0.51696453])

image1_path = ''
image2_path = ''

if not os.path.exists(image1_path):
    print(f"Ошибка: Изображение {image1_path} не найдено.")
if not os.path.exists(image2_path):
    print(f"Ошибка: Изображение {image2_path} не найдено.")

image1 = load_and_preprocess_image(image1_path, mean_rgb)
image2 = load_and_preprocess_image(image2_path, mean_rgb)

image1 = tf.expand_dims(image1, axis=0)
image2 = tf.expand_dims(image2, axis=0)

print(f"Форма первого изображения: {image1.shape}")
print(f"Форма второго изображения: {image2.shape}")

try:
    prediction = siamese_model.predict([image1, image2])
    print(f"Результат предсказания: {prediction}")

    if prediction >= 0.5:
        print("Изображения совпадают (match).")
    else:
        print("Изображения не совпадают (no match).")
except Exception as e:
    print(f"Произошла ошибка: {e}")

print(f"Версия TensorFlow: {tf.__version__}")
