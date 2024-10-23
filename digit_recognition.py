import PIL
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
import threading


def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def create_model(learning_rate):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save_model():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(train_images)

    learning_rate = 0.001
    model = create_model(learning_rate)

    epochs = 10
    batch_size = 64
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                            epochs=1, validation_data=(test_images, test_labels))

        val_acc = history.history['val_accuracy'][0]
        val_loss = history.history['val_loss'][0]

        print(f"Epoch {epoch + 1}/{epochs} - Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save('digit_recognition_model.h5')
            print(f"Model improved and saved. Best Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= 3:
            learning_rate *= 0.5
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            print(f"Learning rate reduced to {learning_rate}.")

        if patience_counter >= patience:
            print("Training stopped due to no improvement.")
            break

    print("Training complete. Final model saved as 'digit_recognition_model.h5'.")


def predict_digit(img):
    img = ImageOps.invert(img.convert('L'))
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img = img.resize((28, 28), PIL.Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    result = model.predict([img])[0]
    return np.argmax(result), max(result)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.title("Handwritten Digit Recognition")
        self.configure(bg="#f0f0f0")

        self.canvas = tk.Canvas(self, width=400, height=400, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 24), bg="#f0f0f0")
        self.classify_btn = tk.Button(self, text="Classify", command=self.classify_handwriting,
                                      font=("Helvetica", 16), bg="#4CAF50", fg="white")
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all,
                                      font=("Helvetica", 16), bg="#f44336", fg="white")


        self.canvas.grid(row=0, column=0, pady=10, padx=10, rowspan=2, sticky=W)
        self.label.grid(row=0, column=1, pady=10, padx=10)
        self.classify_btn.grid(row=1, column=1, pady=10, padx=10)
        self.button_clear.grid(row=2, column=0, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new(mode="RGB", size=(400, 400), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

    def clear_all(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 400, 400), fill=(255, 255, 255))
        self.label.configure(text="Draw a digit")

    def paint(self, event):
        self.x = event.x
        self.y = event.y
        r = 12
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
        self.draw.ellipse([self.x - r, self.y - r, self.x + r, self.y + r], fill='black')

    def classify_handwriting(self):
        # Disable classify button and start progress bar
        self.classify_btn.config(state="disabled")

        # Run classification in a separate thread to avoid freezing
        threading.Thread(target=self.run_classification).start()

    def run_classification(self):
        digit, acc = predict_digit(self.image1)
        self.label.configure(text=f'Digit: {digit}, Acc: {int(acc * 100)}%')
        self.classify_btn.config(state="normal")


if __name__ == '__main__':
    try:
        model = tf.keras.models.load_model('digit_recognition_model.h5')
        print("Model loaded successfully.")
    except:
        print("No saved model found. Training a new model...")
        train_and_save_model()
        model = tf.keras.models.load_model('digit_recognition_model.h5')

    app = App()
    mainloop()
