# Digit Recognition

This project implements handwritten digit recognition (0–9) using a Convolutional Neural Network (CNN) and a graphical user interface.

## Description

The model is trained on the MNIST dataset and can recognize handwritten digits drawn by the user on a canvas.  
On startup, the application automatically loads a saved model or trains a new one if no model is found.

## Technologies Used

- Python  
- TensorFlow / Keras  
- MNIST dataset  
- Pillow (PIL)  
- CustomTkinter  

## How It Works

- The MNIST dataset is loaded and normalized  
- A CNN is trained with data augmentation  
- The best model is saved as `digit_recognition_model.h5`  
- The GUI allows the user to draw a digit  
- The drawn image is preprocessed and passed to the model  
- The predicted digit and confidence are displayed  

## How to Run

1. Install the required dependencies:
   ```bash
   pip install tensorflow pillow customtkinter
   
2. Run the application   
   ```bash
   python main.py

