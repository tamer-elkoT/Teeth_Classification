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

def load_config():
    # Get the absolute path to the root of the project
    current_dir = Path(__file__).resolve().parent # return >> /mnt/d/01_Projects/CV/teeth_classification/src
    # get the config path by 
    config_path = current_dir.parent / "config" / "config.yaml" # curren_dir.parent >> /mnt/d/01_Projects/CV/teeth_classification
    # Open the config file in read only mode and save it tempioraly to the RAM to the file variable
    with open(config_path, "r") as file:
        # yaml.safe_load return a dict of Key-value pair of the yaml file
        return yaml.safe_load(file)
config = load_config()

# load models globally
CNN_path = config["model_paths"]["cnn"]
Eff_net_path = config["model_paths"]["efficientnet"]
models = {
    "CNN": load_model(CNN_path),
    "EfficientNetB0": load_model(Eff_net_path, custom_objects=custom_dict)# We pass the custom_dict to the effiecenet model to solve the error of preprocess_input not regonized inside the model
}
def get_model(model_name: str):
    """
    Docstring for get_model
    Load the Model from the RAM
    :param model_name: Available model names is : (CNN, EfficientNetB0 )
    :type model_name: str
    """
    # Get the value of the dictionary (models) >> The model itself after loading it.
    return models.get(model_name)




