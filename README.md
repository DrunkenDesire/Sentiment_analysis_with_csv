# Sentiment_analysis_with_csv
For sentiment analysis on text data stored in a CSV file using NLTK’s VADER sentiment analyzer. IT includes data cleaning, sentiment prediction, accuracy evaluation  and visualization of actual vs predicted sentiment distributions. It is a straightforward sentiment analysis using Python libraries like pandas, sklearn, matplotlib and nltk.

## Features

- Text cleaning (lowercasing, removing punctuation)
- Sentiment classification using VADER
- Accuracy calculation against true sentiment labels
- Visualization of actual vs predicted sentiment distributions via bar charts

## Project Files

- `Sentiment_analysis_with_csv.py`: Main Python script for sentiment analysis on CSV data.
- `sentiment_analysis.csv`: Example CSV file expected to contain at least `text` and `sentiment` columns.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 
- pandas
- nltk
- scikit-learn
- matplotlib

## Installation

Install the required dependencies using pip:

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

## Usage

1. Prepare your CSV file named `sentiment_analysis.csv` in the project directory. The CSV must have these columns:
   - `text`: The text to analyze
   - `sentiment`: The true sentiment labels (`positive`, `negative`, or `neutral`)

2. Run the script:
python Sentiment_analysis_with_csv.py

3. The script will output:
   - The model accuracy on the terminal
   - Bar charts of actual vs predicted sentiment counts





