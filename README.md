# Sentiment Analysis on IMDB Movie Reviews

This project uses a deep learning model to classify IMDB movie reviews as positive or negative.

## Requirements

- Python 3.x
- TensorFlow
- Numpy
- Pandas
- Matplotlib

## Setup

1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Run the training script to train the model: `python src/model.py`.
4. The trained model will be saved as `sentiment_model.h5`.

## Files

- `src/model.py`: Defines the model architecture and training process.
- `src/data_preprocessing.py`: Contains functions for preprocessing the IMDB dataset.
- `src/evaluate_model.py`: Evaluate the model's performance on test data.
- `src/predict.py`: Used for predicting sentiment on new reviews.

## Usage

To make predictions, run the following command:
```bash
python src/predict.py "This movie was fantastic!"
