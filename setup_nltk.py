import nltk
import os

def setup_nltk():
    """Downloads necessary NLTK data to a specific local directory for deployment reliability."""
    print("Starting NLTK setup...")
    
    # Define a local path for NLTK data within the project
    # This ensures Render persistence and explicit path mapping
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
        print(f"Created directory: {nltk_data_path}")
    
    # Add this path to NLTK's search path for the current process
    nltk.data.path.append(nltk_data_path)
    
    # Resources needed for preprocessing.py
    resources = ["stopwords", "punkt_tab"]
    
    for resource in resources:
        try:
            print(f"Downloading {resource} to {nltk_data_path}...")
            nltk.download(resource, download_dir=nltk_data_path, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {e}")
            
    print("NLTK setup complete.")

if __name__ == "__main__":
    setup_nltk()
