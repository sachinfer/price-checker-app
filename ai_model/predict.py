from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

model = load_model("ai_model/trained_model.h5")
classes = ['t-shirt', 'dress', 'jeans', 'shirt', 'jacket']

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return classes[np.argmax(pred)]
