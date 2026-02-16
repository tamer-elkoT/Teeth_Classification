import tensorflow as tf 
import yaml
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input  
# We import the preprocess_input because inside the effiecentnet mode we use a custom lambda function 
# so the mode will raise an error because it doesn't now what is preprecess_input came from 
# To solve the error when we try to load the model it can't recognize the preprocess_input function 
# Didn't now what it do so we use this line of code to tell keras
# Whenever you see the name 'preprocess_input' in the model file, use this specific function that I just imported from the EfficientNet library.
custom_dict = {'preprocess_input': preprocess_input}

# 1. Define the Project Root globally so we can use it everywhere
# __file__ is src/model_loader.py. 
# .parent is src/ 
# .parent.parent is the main project folder!
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config():
    """
    Loads the project configuration settings from the config.yaml file.

    
    Returns:
        dict: A dictionary containing the parsed configuration key-value pairs 
              (e.g., model paths, class labels).
    """
   
    # Use the global PROJECT_ROOT to find the config 
    config_path = PROJECT_ROOT / "config" / "config.yaml" # curren_dir.parent >> /mnt/d/01_Projects/CV/teeth_classification
    # Open the config file in read only mode and save it tempioraly to the RAM to the file variable
    with open(config_path, "r") as file:
        # yaml.safe_load return a dict of Key-value pair of the yaml file
        return yaml.safe_load(file)
config = load_config()

# load models globally
rel_path_cnn_model = config["model_paths"]["cnn"]
rel_path_eff_model = config["model_paths"]["efficientnet"]

CNN_Path = PROJECT_ROOT / rel_path_cnn_model
EFF_Path = PROJECT_ROOT / rel_path_eff_model
models = {
    "CNN": load_model(CNN_Path),
    "EfficientNetB0": load_model(EFF_Path, custom_objects=custom_dict)# We pass the custom_dict to the effiecenet model to solve the error of preprocess_input not regonized inside the model
}
def get_model(model_name: str):
    """
    Retrieves a pre-loaded machine learning model from the global memory cache.

    This function acts as a fast-access getter, pulling the fully instantiated 
    Keras model directly from the server's RAM rather than reloading the heavy 
    .keras files from the hard drive. This guarantees lightning-fast inference.

    Args:
        model_name (str): The exact name of the model to retrieve. 
                          Available options are "CNN" or "EfficientNetB0".

    Returns:
        tf.keras.Model or None: The compiled Keras model object ready for 
                                predictions, or None if the model_name is invalid.
    """
    # Get the value of the dictionary (models) >> The model itself after loading it.
    return models.get(model_name)




