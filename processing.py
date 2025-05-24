from PIL import Image
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = preprocess_input(img_array)        # prétraitement MobileNetV2
    return img_array
