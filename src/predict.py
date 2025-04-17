# import tensorflow as tf
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.datasets import imdb
# import numpy as np

# def predict_sentiment(review):
#     # Load model
#     model = load_model('data/sentiment_model.h5')

#     # Preprocess review (convert words to integer indices)
#     word_index = imdb.get_word_index()
#     review = review.lower().split()
#     review = [word_index.get(word, 0) for word in review]
#     review = pad_sequences([review], maxlen=250, padding='post')

#     # Predict sentiment
#     prediction = model.predict(review)
#     sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
#     return sentiment

# if __name__ == "__main__":
#     review = input("Enter a movie review: ")
#     sentiment = predict_sentiment(review)
#     print(f"The sentiment is: {sentiment}")
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb
# import numpy as np
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global variables to avoid reloading
# model = None
# word_index = None

# def load_assets():
#     """Load model and word index only once for better performance"""
#     global model, word_index
    
#     try:
#         # Load model
#         if model is None:
#             model_path = os.path.join('models', 'sentiment_model.h5')
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model file not found at {model_path}")
#             model = load_model(model_path)
#             logger.info("Model loaded successfully")
        
#         # Load word index
#         if word_index is None:
#             word_index = imdb.get_word_index()
#             logger.info("Word index loaded")
            
#     except Exception as e:
#         logger.error(f"Error loading assets: {e}")
#         raise

# def preprocess_review(review_text):
#     """Convert raw text to padded sequence"""
#     try:
#         # Convert text to sequence
#         review = review_text.lower().split()
#         review = [word_index.get(word, 0) for word in review]  # 0 for unknown words
        
#         # Pad sequence
#         max_length = 250
#         review = pad_sequences([review], maxlen=max_length, padding='post', truncating='post')
#         return review
#     except Exception as e:
#         logger.error(f"Error preprocessing review: {e}")
#         raise

# def predict_sentiment(review_text):
#     """
#     Predict sentiment of a movie review
#     Args:
#         review_text (str): Raw review text
#     Returns:
#         dict: {'sentiment': 'Positive/Negative', 'confidence': float}
#     """
#     try:
#         # Lazy loading of assets
#         if model is None or word_index is None:
#             load_assets()
        
#         # Preprocess and predict
#         processed_review = preprocess_review(review_text)
#         prediction = model.predict(processed_review, verbose=0)[0][0]
        
#         # Format results
#         sentiment = "Positive" if prediction > 0.5 else "Negative"
#         confidence = float(prediction if sentiment == "Positive" else 1 - prediction)
        
#         return {
#             'sentiment': sentiment,
#             'confidence': round(confidence, 4),
#             'raw_score': float(prediction)
#         }
        
#     except Exception as e:
#         logger.error(f"Prediction failed: {e}")
#         return {
#             'sentiment': 'Error',
#             'confidence': 0.0,
#             'error': str(e)
#         }

# if __name__ == "__main__":
#     # Command-line interface
#     try:
#         load_assets()
#         review = input("Enter a movie review: ").strip()
#         if not review:
#             print("Error: Review cannot be empty!")
#         else:
#             result = predict_sentiment(review)
#             print(f"Sentiment: {result['sentiment']}")
#             print(f"Confidence: {result['confidence']*100:.2f}%")
#     except KeyboardInterrupt:
#         print("\nOperation cancelled")
#     except Exception as e:
#         print(f"Error: {e}")
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load model and word index
# model = tf.keras.models.load_model('models/sentiment_model.h5')
# word_index = imdb.get_word_index()

# def predict_sentiment(review_text):
#     try:
#         # Convert text to sequence
#         words = review_text.lower().split()
#         sequence = [word_index.get(word, 0) for word in words]
        
#         # Pad sequence
#         padded = pad_sequences([sequence], maxlen=250, padding='post', truncating='post')
        
#         # Predict
#         prediction = model.predict(padded, verbose=0)[0][0]
#         confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
#         return {
#             'sentiment': 'Positive' if prediction > 0.5 else 'Negative',
#             'confidence': confidence,
#             'raw_score': float(prediction)
#         }
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         return {
#             'sentiment': 'Error',
#             'error': str(e)
#         }
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb
# import re
# import numpy as np

# # Initialize model and word index
# model = None
# word_index = None

# def initialize_model():
#     global model, word_index
#     try:
#         # Load model with validation
#         model = tf.keras.models.load_model('models/sentiment_model.h5')
        
#         # Load word index with proper offset handling
#         word_index = imdb.get_word_index()
#         word_index = {k:(v+3) for k,v in word_index.items()}
#         word_index["<PAD>"] = 0
#         word_index["<START>"] = 1
#         word_index["<UNK>"] = 2
        
#     except Exception as e:
#         print(f"Model loading failed: {str(e)}")
#         raise

# def safe_predict(prediction):
#     """Ensure prediction is within valid range and calculate proper confidence"""
#     try:
#         # Clip prediction to valid range [0,1]
#         prediction = float(np.clip(prediction, 0.0, 1.0))
        
#         # Calculate sentiment and confidence
#         if prediction > 0.6:
#             return ("Positive", prediction)
#         elif prediction < 0.4:
#             return ("Negative", 1 - prediction)
#         else:
#             # For neutral, confidence is based on distance from 0.5
#             distance = abs(prediction - 0.5)
#             return ("Neutral", 0.5 - distance)
    
#     except Exception as e:
#         print(f"Prediction processing error: {str(e)}")
#         return ("Error", 0.0)

# def predict_sentiment(review_text):
#     try:
#         # Initialize if not done
#         if model is None or word_index is None:
#             initialize_model()
        
#         # Validate input
#         if not review_text or not isinstance(review_text, str):
#             return {
#                 'sentiment': 'Invalid',
#                 'confidence': 0.0,
#                 'error': 'Empty or invalid input'
#             }
        
#         # Clean and tokenize with proper handling
#         text = re.sub(r'[^\w\s]', '', review_text.lower()).strip()
#         if not text:
#             return {
#                 'sentiment': 'Invalid',
#                 'confidence': 0.0,
#                 'error': 'No valid text after cleaning'
#             }
        
#         # Convert to sequence with validation
#         sequence = [word_index.get(word, word_index["<UNK>"]) for word in text.split()]
#         sequence = [min(idx, 9999) for idx in sequence]  # Ensure within vocab range
        
#         # Pad sequence
#         padded = pad_sequences([sequence], maxlen=250, padding='post', truncating='post')
        
#         # Predict with validation
#         prediction = model.predict(padded, verbose=0)[0][0]
#         sentiment, confidence = safe_predict(prediction)
        
#         # Final validation
#         confidence = min(100.0, max(0.0, confidence * 100))  # Ensure 0-100 range
        
#         return {
#             'sentiment': sentiment,
#             'confidence': round(float(confidence), 1),
#             'raw_score': float(prediction)
#         }
        
#     except Exception as e:
#         return {
#             'sentiment': 'Error',
#             'confidence': 0.0,
#             'error': str(e)
#         }

# # Test cases
# if __name__ == "__main__":
#     initialize_model()
#     test_samples = [
#         "This movie was terrible",
#         "I loved this film!",
#         "It was okay",
#         "",  # Empty
#         123,  # Invalid type
#         "This movie was so bad it made me angry",
#         "The best cinematic experience ever"
#     ]
    
#     for text in test_samples:
#         print(f"Input: {text}")
#         result = predict_sentiment(text)
#         print(f"Result: {result}\n")
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import numpy as np

# Initialize model and word index
model = None
word_index = None

def initialize_model():
    global model, word_index
    try:
        model = tf.keras.models.load_model('models/sentiment_model.h5')
        
        word_index = imdb.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2

    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise

def predict_sentiment(review_text):
    try:
        # Initialize model and word index if not already done
        if model is None or word_index is None:
            initialize_model()

        if not review_text or not isinstance(review_text, str):
            return {
                'sentiment': 'Invalid',
                'confidence': 0.0,
                'error': 'Empty or invalid input'
            }

        # Tokenize properly using Keras text_to_word_sequence
        tokens = text_to_word_sequence(review_text)
        encoded = [1]  # <START> token

        for word in tokens:
            index = word_index.get(word, 2)  # 2 = <UNK>
            index = min(index, 9999)         # clip to vocabulary limit
            encoded.append(index)

        padded = pad_sequences([encoded], maxlen=250, padding='post', truncating='post')
        prediction = model.predict(padded, verbose=0)[0][0]

        # Assign sentiment label
        if prediction >= 0.6:
            sentiment = "Positive"
        elif prediction <= 0.4:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            'sentiment': sentiment,
            'confidence': round(float(prediction) * 100, 1),
            'raw_score': float(prediction)
        }

    except Exception as e:
        return {
            'sentiment': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }

# Run tests
if __name__ == "__main__":
    initialize_model()
    test_samples = [
        "This movie was terrible",
        "I loved this film!",
        "It was okay",
        "",  # Empty
        123,  # Invalid type
        "This movie was so bad it made me angry",
        "The best cinematic experience ever"
    ]

    for text in test_samples:
        print(f"\n📝 Input: {text}")
        result = predict_sentiment(text)
        print(f"📊 Result: {result}")
