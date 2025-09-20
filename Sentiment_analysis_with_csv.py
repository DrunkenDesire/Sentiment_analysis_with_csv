
import string
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Download NLTK packages if not already present
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# ---------- Step 1: Load dataset ----------
df = pd.read_csv(r"C:\Users\DRUNK\OneDrive\Documents\sentiment_analysis.csv")

# ---------- Step 2: Clean text ----------
df["cleaned_text"] = df["text"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# ---------- Step 3: Define sentiment function ----------
def get_sentiment_label(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(str(sentiment_text))
    neg = score['neg']
    pos = score['pos']

    # Match predicted labels with CSV labels
    if neg > pos:
        return "negative"
    elif pos > neg:
        return "positive"
    else:
        return "neutral"

# ---------- Step 4: Apply sentiment analysis ----------
df["predicted"] = df["cleaned_text"].apply(get_sentiment_label)

# ---------- Step 5: Calculate accuracy ----------
if "sentiment" in df.columns:
    # Convert both to lowercase so they match perfectly
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df["predicted"] = df["predicted"].str.lower().str.strip()

    accuracy = accuracy_score(df["sentiment"], df["predicted"])
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
else:
    print("No 'sentiment' column in CSV. Skipping accuracy calculation.")

# ---------- Step 6: Plot Actual vs Predicted ----------
if "sentiment" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["sentiment"].value_counts().plot(kind='bar', ax=axes[0], title="Actual Sentiments")
    df["predicted"].value_counts().plot(kind='bar', ax=axes[1], title="Predicted Sentiments")
    plt.tight_layout()
    plt.show()
else:
    df["predicted"].value_counts().plot(kind='bar', title="Predicted Sentiments")
    plt.show()
print(df["sentiment"].unique())
