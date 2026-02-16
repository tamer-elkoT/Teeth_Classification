from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.inference import predict_image
from enum import Enum

# Intialize the router
router = APIRouter()

# Define a droplist that contain the available models
class AvailableModels(str, Enum):
    cnn = "CNN"
    efficientnet = "EfficientNetB0"

# create the predict endpoint
@router.post("/predict")
async def predict(
    # The instance (image)
    # the class (UploadFile) inherit from the (File)
    image: UploadFile = File(...), # Expect a file (image) from the user
    # The instance (model_name)
    # the class (AvailableModels) inherit from the base class (Enum) 
    model_name: AvailableModels = Form(...) # recieve the model name (CNN or EfficientnetB0)

):

    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG or jpeg")
    
    # read the Image into the memory as image_bytes
    image_bytes = await image.read()

    try:
        prediction = predict_image(image_bytes, model_name)
        return {"filename": image.filename, "prediction": prediction, "model_used": model_name.value } # we set model_name.value because the model_name is a Enum object <AvailableModels.cnn: 'CNN'> we need the (CNN) by using (.value)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))