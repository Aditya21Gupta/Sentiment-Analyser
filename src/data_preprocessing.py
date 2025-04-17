# src/data_preprocessing.py
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_data(num_words=10000, max_length=250):
    """Load and preprocess IMDB dataset"""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
    
    return (x_train, y_train), (x_test, y_test)