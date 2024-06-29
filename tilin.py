import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from customtkinter import *
import customtkinter as ctk

# Interfaz Gráfica
root = CTk()
set_appearance_mode("dark")
root.geometry("400x300")
root.title("PDI")
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Mantener referencia a la imagen
imagen_tk = None

# Función para cargar una imagen
def cargar_imagen():
    global imagen_tk  # Usar la referencia global
    archivo = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if archivo:
        image = load_img(archivo, target_size=(256, 256))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Añade una dimensión para el batch
        image_array /= 255.0  # Normaliza los píxeles

        imagenNorm = Image.open(archivo)
        # Redimensionar si el tamaño excede los 2000x1400 píxeles
        if imagenNorm.width > 256 or imagenNorm.height > 256:
            imagenNorm = imagenNorm.resize((256, 256))
        
        imagen_tk = ImageTk.PhotoImage(imagenNorm)
        label_imagen.config(image=imagen_tk)
        label_imagen.image = imagen_tk  # Mantener referencia para evitar garbage collection

        # Realiza la predicción
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])  # predictions[0] porque asumo que predictions es un array de arrays
        predicted_class = selected_labels[predicted_class_index]
        
        resultado.set(f"La clase predicha es: {predicted_class}")

# Definir la arquitectura del modelo
model = Sequential()

model.add(tf.keras.layers.Input(shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(4, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Cargar los pesos desde el archivo .h5
try:
    model.load_weights('cnn_weights.best.weights.h5')
    print("Pesos cargados correctamente.")
except OSError as e:
    print(f"Error al cargar los pesos: {e}")

selected_labels = ['bellflower', 'common_daisy', 'rose', 'sunflower']

# Variables de interfaz
resultado = StringVar()
resultado.set("Cargue una imagen para predecir la clase.")

# Botón para cargar imagen
btn_cargar = CTkButton(root, text="Cargar Imagen", command=cargar_imagen)
btn_cargar.pack(pady=20)

# Etiqueta para mostrar la imagen
label_imagen = tk.Label(root)
label_imagen.pack(pady=20)

# Etiqueta para mostrar el resultado
lbl_resultado = CTkLabel(root, textvariable=resultado)
lbl_resultado.pack(pady=20)

# Iniciar el loop de la interfaz
root.mainloop()