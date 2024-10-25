# Song Popularity Prediction

This project uses data mining and machine learning techniques, including **sentiment analysis**, **feature extraction**, and the **k-Nearest Neighbors (k-NN)** algorithm, to predict song popularity based on YouTube comments.

## Overview

This repository demonstrates a predictive model that evaluates YouTube comments on songs to estimate popularity. By analyzing comment volume, sentiment, and other features, we explore patterns that correlate with song popularity.

## Technologies

- **Python**: Core programming language
- **Pandas**: Data manipulation and preprocessing
- **k-NN**: Machine learning model for classification
- **Sentiment Analysis**: Extracts sentiment from comments for feature creation

## Project Structure

- `data/`: Contains raw YouTube comment data for analysis
- `notebooks/`: Jupyter notebooks for data exploration and modeling
- `src/`: Core scripts for data preprocessing and modeling
- `results/`: Model performance metrics and visualizations

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tranhoangminh1412/song-popularity-prediction
   cd song-popularity-prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Usage
  - Data Preprocessing: Run preprocess.py in src/ to clean and prepare data.
  - Model Training: Execute train_model.py to train and evaluate the k-NN model.
  - Prediction: Use predict.py to make popularity predictions on new data.
4. Results
  - The k-NN model achieved high accuracy in predicting song popularity based on comment sentiment and engagement levels, highlighting the value of social feedback in popularity forecasting.

## Future Work
Experiment with additional models (e.g., SVM, Random Forest)
Integrate other features, such as song metadata and engagement rates

## License
This project is licensed under the MIT License.
