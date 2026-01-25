"""Utils Module used for Data Analysis and EDA"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def visualize_ngrams(df: pd.DataFrame, category_name: str, ngram_range: Tuple[int, int] = (2, 2), top_k: int = 10):
    """
    Visualizes the top N-grams for a specific return category.
    Args:
        df: The DataFrame containing the reviews and categories
        category_name: The class to analyze (e.g., 'Defect', 'Sizing')
        n_gram_range: (2,2) for bigrams, (3,3) for trigrams
    """

    # 1. Filter the specific category
    subset = df[df["return_category"] == category_name]

    if subset.empty:
        print(f"No data for category: {category_name}")
        return 
    
    # 2. Initialize Vectorizer (Remove English Stop words)
    # We use a custom token pattern to keep it clean
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=5000
    ) # Bag of Words Model

    # 3. Fit and Transform
    try:
        X = vectorizer.fit_transform(subset["review_text"])
    except ValueError:
        # Handle cases where stop words might remove everything
        print(f"Skipping {category_name}: Not enough valid text data.")
        return

    # 4. Sum up the counts of each n-gram
    counts = np.asarray(X.sum(axis=0)).flatten()

    vocab = vectorizer.get_feature_names_out()

    freq_distribution = (
        pd.DataFrame({"ngram": vocab, "count": counts})
        .sort_values("count", ascending=False)
        .head(top_k)
    )

    # 5. Ploting the Bar Chart
    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="count",
        y="ngram",
        data=freq_distribution
    )

    plt.title(f"Top {top_k} {'Bigrams' if ngram_range==(2,2) else 'Trigrams'} for: {category_name}")
    plt.xlabel("Frequency")
    plt.ylabel("Phrase")
    plt.show()




def plot_wordcloud(df: pd.DataFrame, category_name: str):
    """
    Plots a WordCloud for a specific return category.
    Args: 
        df: The DataFrame containing the reviews and categories
        category_name: The class to analyze (e.g., "Defect", "Sizing")
    """
    subset = df[df["return_category"] == category_name]
    text_combined = " ".join(review for review in subset.review_text)

    # Generate
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="magma").generate(text_combined)


    # Plotting the WordCloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud: {category_name}", fontweight="bold", fontsize=16)
    plt.show()
    