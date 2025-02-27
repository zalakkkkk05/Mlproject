import pickle
import os

# Define paths to model and preprocessor
model_path = "artifacts/model.pkl"
preprocessor_path = "artifacts/preprocessor.pkl"

def load_object(file_path):
    """Load the model or preprocessor object using pickle."""
    try:
        print(f"Trying to load object from: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        print(f"Error loading object from {file_path}: {str(e)}")
        return None

# Try to load the model and preprocessor
print("Loading model...")
model = load_object(model_path)

if model:
    print("Model loaded successfully.")
else:
    print("Failed to load model.")

print("Loading preprocessor...")
preprocessor = load_object(preprocessor_path)

if preprocessor:
    print("Preprocessor loaded successfully.")
else:
    print("Failed to load preprocessor.")