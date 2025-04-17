# import h5py
# import tensorflow as tf
# import tensorflow as tf

# with h5py.File("your_model_path.h5", "r") as f:
#     model = tf.keras.models.load_model(f)
# # Removed redundant import and used tf.keras.models.load_model directly
# model = tf.keras.models.load_model("your_model_path.h5")
# import sys
# import os

# # Ensure src folder is accessible
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# try:
#     from src.data_preprocessing import load_data
# except ModuleNotFoundError:
#     print("Error: 'src.data_preprocessing' module not found. Ensure the 'src' folder has an '__init__.py' file.")
#     exit(1)

# def evaluate_model():
#     model_path = os.path.join(os.getcwd(), "data", "sentiment_model.h5")

#     # Check if model file exists
#     if not os.path.exists(model_path):
#         print("Error: Model file not found at {model_path}")
#         return

#     print("Loading model...")
#     model = tf.keras.models.load_model(model_path)

#     print("Loading dataset...")
#     try:
#         (x_train, y_train), (x_test, y_test) = load_data()
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return

#     print("Evaluating model...")
#     loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

#     print(f"\nTest Loss: {loss:.4f}")
#     print(f"Test Accuracy: {accuracy:.4f}")

# if __name__ == "__main__":
#     evaluate_model()
import os
import sys
import h5py
import tensorflow as tf

# Ensure 'src' folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    try:
        from .data_preprocessing import load_data  # Relative import # Ensure this module exists
    except ModuleNotFoundError:
        print("Error: 'src.data_preprocessing' module not found. Ensure the 'src' folder has an '__init__.py' file and is in the correct location.")
        sys.exit(1)
except ModuleNotFoundError:
    print("Error: 'src.data_preprocessing' module not found. Ensure the 'src' folder has an '__init__.py' file.")
    sys.exit(1)  # Use sys.exit instead of exit for better practice

def evaluate_model():
    model_path = os.path.join(os.getcwd(), "data", "sentiment_model.h5")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")  # Fixed f-string formatting
        return

    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path)  # Load model directly
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = load_data()  # Ensure load_data() is defined properly
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Evaluating model...")
    try:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")

if __name__ == "__main__":
    evaluate_model()
