from src.model_loader import get_model
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

