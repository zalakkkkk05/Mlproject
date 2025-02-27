import pickle
import os

preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

# Ensure the file exists before trying to load it
if os.path.exists(preprocessor_file_path):
    with open(preprocessor_file_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Print the preprocessor object or check its details
    print("Preprocessor loaded successfully:", preprocessor)
else:
    print(f"File {preprocessor_file_path} does not exist.")

