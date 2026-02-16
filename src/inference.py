from src.model_loader import get_model, config
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
def preprocess_image(image_bytes: bytes, target_size=(224,224)):
    """
    Take Image Bytes & Return an Image Array Through:
    - convert the channels into >> RGB
    - resize the image to (1,224,224,3) >> Expected size by the model 
    - Math (normalization) is handled internally by the models.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    # Resize the image into (224x224)
    image = image.resize(target_size)
    img_array = img_to_array(image)
    # expand the dim
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_bytes: bytes, model_name: str = "CNN"):
    """
    This Function predict which class the image belongs to 
    Parameters:
    - image_bytes: the uploaded image by the user via streamlit.
    - model_name: The available models name are ("CNN", "EfficientNetB0")
    """
    # load the model 
    model = get_model(model_name=model_name)
    # preprocess the input image (image_bytes) >> (check the size )
    img_array = preprocess_image(image_bytes=image_bytes)
    predictions = model.predict(img_array)
    # The np.argmax() get the index of the largest probabilty so in this case
    predicted_class_index = np.argmax(predictions)
    # match the predicted index with the class name 
    return config["classes"][predicted_class_index]

