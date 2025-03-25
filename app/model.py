import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class ImageClassifier:
    def __init__(self):
        # Použijeme předtrénovaný MobileNetV2 model
        self.model = MobileNetV2(weights='imagenet')
    
    def predict(self, image_path):
        # Načtení a příprava obrázku
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Rozšíření dimenzí a preprocessing
        img_batch = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(img_batch)
        
        # Predikce
        predictions = self.model.predict(processed_img)
        
        # Dekódování top 3 předpovědí
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        return decoded_predictions