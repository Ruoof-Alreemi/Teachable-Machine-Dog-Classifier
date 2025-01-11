# Teachable-Machine-Dog-Classifier

# Introduction

This project uses a model trained with Teachable Machine to classify images into the following types:

1. Pomeranian dog
2. Golden Retriever
3. Husky
4. Chow Chow
5. not dog

# Objective

Determine the type of dog (or if the image is not of a dog) using a TensorFlow model.

Requirements

1. Google Colab account.

2. TensorFlow model exported from Teachable Machine:

* Export the model as TensorFlow.

* Download the model.zip file.

3. TensorFlow library installed (it will be installed automatically in Colab).

Project Setup Steps

1. Train the model on Teachable Machine

* Go to Teachable Machine.

* Select Image Project.
‎ • Add the following classes:
* Pomeranian dog
* Golden Retriever
* Husky
* Chow Chow
* not dog
‎ • Upload training images for each class.
‎ • After training, click Export Model and choose TensorFlow.
‎ • Download the model.zip file.

‎2. Run the model in Google Colab

‎ 1. Open Google Colab.
‎ 2. Copy and paste the following code into a new Notebook file.
‎ 3. Follow the instructions to upload the model and images.

‎ Full code:

import tensorflow as tf
print(tf.__version__)

# Install a specific version of TensorFlow if needed
!pip install tensorflow==2.17.1

import numpy as np
from tensorflow.keras.preprocessing import image
import zipfile
import os

# Define the path to the ZIP file and output directory
zip_file_path = '/content/model_zip_path.zip'  # Replace with your actual ZIP file name
output_dir = 'model_dir'

# Unzip the model files
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

# Custom DepthwiseConv2D layer fix
class DepthwiseConv2D_fix(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super(DepthwiseConv2D_fix, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(DepthwiseConv2D_fix, self).get_config()
        config.pop('groups', None)
        return config

# Load the Keras model with custom objects
model = tf.keras.models.load_model(os.path.join(output_dir, 'keras_model.h5'),
                                   custom_objects={'DepthwiseConv2D': DepthwiseConv2D_fix})

# Load class labels
with open(os.path.join(output_dir, 'labels.txt'), 'r') as f:
    class_labels = f.read().splitlines()

# Function to predict class
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size as needed
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    return class_index

# Example usage
img_path = '/content/testt.jpg'  # Replace with your image path
result = predict_image(img_path)
print("Predicted class:", class_labels[result[0]])


# Finally test 
I tested it with two images, a picture of a lion so that it appears that it is(not dog)  and I also tested the type (Golden Retriever) and it was recognized.
