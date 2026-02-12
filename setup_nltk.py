import nltk
import os

def setup_nltk():
    """Downloads necessary NLTK data for production."""
    print("Setting up NLTK data...")
    # List of resources used in preprocessing.py and etl.py
    resources = ['stopwords', 'punkt_tab']
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")

if __name__ == "__main__":
    setup_nltk()
