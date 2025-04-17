# import tensorflow as tf
# from tensorflow import keras
# from keras.utils import pad_sequences
# import numpy as np

# # Load dataset
# imdb = keras.datasets.imdb
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# # Preprocess data
# max_length = 250
# x_train = pad_sequences(x_train, maxlen=max_length, padding="post")
# x_test = pad_sequences(x_test, maxlen=max_length, padding="post")

# # Build model
# model = keras.Sequential([
#     keras.layers.Embedding(10000, 16, input_length=max_length),
#     keras.layers.LSTM(32, return_sequences=True),
#     keras.layers.LSTM(16),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train model
# try:
#     model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
#     print("Training complete.")
# except Exception as e:
#     print(f"Error occurred during training: {e}")

# # Save model
# try:
#     model.save("sentiment_model.h5")
#     print("Model saved successfully!")
# except Exception as e:
#     print(f"Error occurred while saving the model: {e}")
# import tensorflow as tf
# from tensorflow.keras.utils import pad_sequences  # ✅ Correct
#   # Updated import for compatibility
# import numpy as np

# # Load dataset
# imdb = tf.keras.datasets.imdb
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# # Preprocess data
# max_length = 250
# x_train = pad_sequences(x_train, maxlen=max_length, padding="post")
# x_test = pad_sequences(x_test, maxlen=max_length, padding="post")

# # Build model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(10000, 16, input_length=max_length),
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     tf.keras.layers.LSTM(16),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train model
# try:
#     model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
#     print("Training complete.")
# except Exception as e:
#     print(f"Error occurred during training: {e}")

# # Save model
# try:
#     model.save("sentiment_model.h5")
#     print("Model saved successfully!")
# except Exception as e:
#     print(f"Error occurred while saving the model: {e}")
# Fix for collections.abc issue
# Add this at the very top to prevent protobuf conflicts
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import tensorflow as tf  # Import TensorFlow

# import sys
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_imports():
#     try:
#         import tensorflow as tf
#         import numpy as np
#         logger.info(f"TensorFlow {tf.__version__}, NumPy {np.__version__}")
#         return True
#     except Exception as e:
#         logger.error(f"Import failed: {e}")
#         return False

# if not check_imports():
#     sys.exit(1)

# from tensorflow.keras.preprocessing.sequence import pad_sequences

# def main():
#     try:
#         logger.info("Loading data...")
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
        
#         logger.info("Padding sequences...")
#         max_length = 250
#         x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
#         x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")
        
#         logger.info("Building model...")
#         model = tf.keras.Sequential([
#             tf.keras.layers.Embedding(10000, 16, input_length=max_length),
#             tf.keras.layers.LSTM(32, return_sequences=True),
#             tf.keras.layers.LSTM(16),
#             tf.keras.layers.Dense(16, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
#         logger.info("Training...")
#         model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))
        
#         model.save("sentiment_model.h5")
#         logger.info("Model saved")
        
#     except Exception as e:
#         logger.error(f"Runtime error: {e}")

# if __name__ == "__main__":
#     main()
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import tensorflow as tf
# import numpy as np
# import logging
# import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_imports():
#     """Verify required packages are installed"""
#     try:
#         logger.info(f"TensorFlow {tf.__version__}, NumPy {np.__version__}")
#         return True
#     except Exception as e:
#         logger.error(f"Import failed: {e}")
#         return False

# def train_model():
#     """
#     Train and save a sentiment analysis model
#     Returns:
#         str: Path to the saved model
#     """
#     if not check_imports():
#         sys.exit(1)
    
#     try:
#         # Load and preprocess data
#         logger.info("Loading IMDB dataset...")
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
        
#         logger.info("Padding sequences...")
#         max_length = 250
#         x_train = tf.keras.preprocessing.sequence.pad_sequences(
#             x_train, maxlen=max_length, padding="post", truncating="post")
#         x_test = tf.keras.preprocessing.sequence.pad_sequences(
#             x_test, maxlen=max_length, padding="post", truncating="post")
        
#         # Build model
#         logger.info("Building model architecture...")
#         model = tf.keras.Sequential([
#             tf.keras.layers.Embedding(10000, 16, input_length=max_length),
#             tf.keras.layers.LSTM(32, return_sequences=True),
#             tf.keras.layers.LSTM(16),
#             tf.keras.layers.Dense(16, activation='relu'),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(
#             optimizer='adam',
#             loss='binary_crossentropy',
#             metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
#         )
        
#         # Train model
#         logger.info("Starting training...")
#         history = model.fit(
#             x_train, y_train,
#             epochs=3,
#             batch_size=32,
#             validation_data=(x_test, y_test),
#             verbose=1
#         )
        
#         # Save model
#         model_path = "models/sentiment_model.h5"
#         os.makedirs("models", exist_ok=True)
#         model.save(model_path)
#         logger.info(f"Model saved to {model_path}")
        
#         return model_path
        
#     except Exception as e:
#         logger.error(f"Error during model training: {str(e)}")
#         raise

# if __name__ == "__main__":
#     train_model()
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import tensorflow as tf
# import numpy as np
# import logging
# import sys
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def check_imports():
#     """Verify required packages are installed"""
#     try:
#         logger.info(f"TensorFlow {tf.__version__}, NumPy {np.__version__}")
#         return True
#     except Exception as e:
#         logger.error(f"Import failed: {e}")
#         return False

# def train_model():
#     """
#     Train and save an improved sentiment analysis model
#     Returns:
#         str: Path to the saved model
#     """
#     if not check_imports():
#         sys.exit(1)
    
#     try:
#         # Load and preprocess data
#         logger.info("Loading IMDB dataset...")
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
#             num_words=10000,
#             index_from=3  # Start word indices from 3 (0=padding, 1=start, 2=unknown)
#         )
        
#         logger.info("Padding sequences...")
#         max_length = 250
#         x_train = tf.keras.preprocessing.sequence.pad_sequences(
#             x_train, maxlen=max_length, padding="post", truncating="post")
#         x_test = tf.keras.preprocessing.sequence.pad_sequences(
#             x_test, maxlen=max_length, padding="post", truncating="post")
        
#         # Build improved model
#         logger.info("Building improved model architecture...")
#         model = tf.keras.Sequential([
#             tf.keras.layers.Embedding(10000, 64, input_length=max_length),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
        
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             loss='binary_crossentropy',
#             metrics=[
#                 'accuracy',
#                 tf.keras.metrics.Precision(name='precision'),
#                 tf.keras.metrics.Recall(name='recall'),
#                 tf.keras.metrics.AUC(name='auc')
#             ]
#         )
        
#         # Callbacks
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=2),
#             ModelCheckpoint(
#                 'models/best_model.h5',
#                 save_best_only=True,
#                 monitor='val_accuracy',
#                 mode='max'
#             )
#         ]
        
#         # Train model with validation split
#         logger.info("Starting training...")
#         history = model.fit(
#             x_train, y_train,
#             epochs=10,
#             batch_size=64,
#             validation_split=0.2,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Save final model
#         model_path = "models/sentiment_model.h5"
#         os.makedirs("models", exist_ok=True)
#         model.save(model_path)
#         logger.info(f"Model saved to {model_path}")
        
#         # Evaluate on test set
#         logger.info("Evaluating on test set...")
#         results = model.evaluate(x_test, y_test, verbose=0)
#         logger.info(f"Test Loss: {results[0]:.4f}")
#         logger.info(f"Test Accuracy: {results[1]*100:.2f}%")
#         logger.info(f"Test Precision: {results[2]*100:.2f}%")
#         logger.info(f"Test Recall: {results[3]*100:.2f}%")
#         logger.info(f"Test AUC: {results[4]*100:.2f}%")
        
#         return model_path
        
#     except Exception as e:
#         logger.error(f"Error during model training: {str(e)}")
#         raise

# if __name__ == "__main__":
#     train_model()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU (optional)

import tensorflow as tf
import numpy as np
import logging
import sys
from tensorflow.keras.callbacks import EarlyStopping

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_imports():
    """Verify required packages"""
    try:
        logger.info(f"TensorFlow {tf.__version__}, NumPy {np.__version__}")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

def train_model():
    """
    Train and save a sentiment analysis model using the IMDB dataset.
    """
    if not check_imports():
        sys.exit(1)
    
    try:
        # Load IMDB dataset
        logger.info("Loading IMDB dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
            num_words=10000, index_from=3
        )

        logger.info("Padding sequences to uniform length...")
        max_length = 250
        x_train = tf.keras.preprocessing.sequence.pad_sequences(
            x_train, maxlen=max_length, padding="post", truncating="post"
        )
        x_test = tf.keras.preprocessing.sequence.pad_sequences(
            x_test, maxlen=max_length, padding="post", truncating="post"
        )

        # Build model
        logger.info("Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 64, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # Train model
        logger.info("Training model...")
        history = model.fit(
            x_train, y_train,
            epochs=15,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Create directory and save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/sentiment_model.h5"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Evaluate on test set
        logger.info("Evaluating model...")
        results = model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]*100:.2f}%")
        logger.info(f"Test Precision: {results[2]*100:.2f}%")
        logger.info(f"Test Recall: {results[3]*100:.2f}%")

        return model_path

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
