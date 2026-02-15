from src.model_loader import get_model, models

cnn_model = get_model("CNN")
print(cnn_model)

def test_get_cnn_model():
    """Test that the CNN model loads successfully."""
    model = get_model("CNN")
    
    # Assert checks if a statement is True. If it's False, the test fails.
    assert model is not None, "CNN model failed to load. It returned None."
    
    # Double-check that what we loaded is actually a TensorFlow model 
    # by verifying it has a 'predict' function inside it.
    assert hasattr(model, "predict"), "The loaded CNN object does not have a predict method."

def test_get_efficientnet_model():
    """Test that the EfficientNetB0 model loads successfully."""
    model = get_model("EfficientNetB0")
    assert model is not None, "EfficientNetB0 model failed to load."
    assert hasattr(model, "predict"), "The loaded EfficientNet object does not have a predict method."

def test_get_invalid_model():
    """Test that asking for a fake model safely returns None without crashing."""
    model = get_model("SomeFakeModel")
    assert model is None, "Expected None for an invalid model, but got something else."