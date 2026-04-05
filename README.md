# ReturnX Intelligent Returns Classifier 📦

🚀 **[Live Dashboard](https://returnx-dashboard.onrender.com)** | 📂 **[API Documentation](https://returnx-api.onrender.com/docs)** | 📊 **[GitHub Repository](https://github.com/GFFB0314/ReturnX-Intelligent-Returns-Classifier)**

Welcome to the **ReturnX Intelligent Returns Classifier** – a production-grade machine learning system designed to automate the triaging of e-commerce product returns. This project demonstrates the end-to-end lifecycle of an NLP application, from raw text processing to financial impact analysis, solving a critical operational bottleneck.

## Table of Contents
- [ReturnX Intelligent Returns Classifier 📦](#returnx-intelligent-returns-classifier-)
  - [Table of Contents](#table-of-contents)
  - [About the Project 📖](#about-the-project-)
    - [✨ Key Features:](#-key-features)
  - [Data Strategy \& Engineering Pipeline 🏗️](#data-strategy--engineering-pipeline-️)
    - [1. Data Sourcing](#1-data-sourcing)
    - [2. ETL \& Preprocessing](#2-etl--preprocessing)
    - [3. Business-Aware Feature Engineering](#3-business-aware-feature-engineering)
    - [4. Financial Impact Analysis](#4-financial-impact-analysis)
  - [Technologies \& Libraries Used 🛠️](#technologies--libraries-used-️)
  - [Getting Started 🚀](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [⚙️ Installation](#️-installation)
  - [Usage Guide 🖥️](#usage-guide-️)
  - [Results \& Impact 📊](#results--impact-)
  - [Contributing 🤝](#contributing-)
  - [Contact ✉️](#contact-️)
  - [License ©️](#license-️)

## About the Project 📖

The goal of this project is to automate the classification of customer return reasons (Defect, Sizing, Style, Other) to reduce manual processing costs and improve routing efficiency. It moves beyond simple keyword matching by using advanced Natural Language Processing (NLP) to understand customer intent from unstructured text and metadata.

This project was built to demonstrate **robust programming skills and Data Science expertise**, focusing on reproducibility, statistical rigor, and business ROI.

### ✨ Key Features:
*   **Business-Aware NLP:** Custom text cleaning pipeline that handles "poison phrases" (e.g., "true to size" in a sizing complaint) and preserves critical negations.
*   **Modular Architecture:** Clean Python package structure (`src/`) separating ETL, Modeling, and Inference logic.
*   **Mixed-Data Modeling:** Combines high-dimensional TF-IDF text features with normalized numeric metadata (Age, Rating, Word Count).
*   **Financial Impact Analysis:** Evaluates the model based on **Net Savings ($)** and **ROI (%)**, not just F1-score.
*   **Integrated Testing:** Includes unit tests (`pytest`) to verify text cleaning logic and prediction integrity.
*   **Production Dashboard:** Interactive Streamlit interface providing real-time predictions and confidence scores.

---

## Data Strategy & Engineering Pipeline 🏗️

This project follows a strict **MLOps** workflow to ensure data quality and scalability.

### 1. Data Sourcing
*   **Return Data:** Aggregated customer reviews and return labels from the retail database (`retail_returns.labeled_reviews_v`).
*   **Metadata:** Ingested customer validation metrics including Age and Product Ratings to provide context to the unstructured text.

### 2. ETL & Preprocessing
Instead of relying solely on raw dumps, I utilized **SQL** and Python for robust data lifting.
*   **Ingestion:** Automated extraction mechanism with CSV fallback for offline development.
*   **Cleaning:** Implemented a rigorous text cleaning function `clean_review_text` to remove noise while keeping sentiment-bearing words and negations ("not", "never").

### 3. Business-Aware Feature Engineering
Exploratory Data Analysis (EDA) revealed that text alone wasn't enough; context matters.
*   **The Intent:** A 3-star rating with the word "small" implies something different than a 5-star rating with the word "small".
*   **The Engineering:** I implemented a `ColumnTransformer` to normalize numeric data (Age, Rating) via `MaxAbsScaler` and combined it with a 5,000-feature TF-IDF matrix (unigrams/bigrams) to capture complex customer intent.

### 4. Financial Impact Analysis
A machine learning model is only as good as the value it creates.
*   **Scenario:** A production batch of 4,529 returns.
*   **Strategy:** The system compares "Business As Usual" (Manual Review at ~$2.00/item) against Feature-Automated costs ($0.10/item), factoring in a $5.00 penalty for every misclassification to calculate true ROI.

---

## Technologies & Libraries Used 🛠️

This project leverages a modern Python Data Science stack.

*   🐍 **Python 3.10+:** The core language.
*   🐼 **Pandas & NumPy:** For high-performance data manipulation and vectorization.
*   🤖 **Scikit-Learn & XGBoost:** For machine learning pipelines, gradient boosting, and cross-validation.
*   ⚖️ **Imbalanced-Learn:** For SMOTE synthetic over-sampling to handle rare defect classes.
*   📊 **Streamlit:** For the production dashboard interface.
*   ⚡ **FastAPI:** For building high-performance, production-ready APIs.
*   🖤 **Black:** For automated and consistent code formatting.
*   🔍 **Pylint:** For static code analysis and ensuring code quality.
*   🗄️ **SQLAlchemy:** For robust database interaction.
*   🧪 **Pytest:** For unit testing and verifying pipeline integrity.

---

## Getting Started 🚀

Follow these steps to get a local copy up and running.

### Prerequisites

*   **Python 3.10+**
*   **Pip**

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/GFFB0314/ReturnX-Intelligent-Returns-Classifier.git
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Tests:**
    Verify the logic by running the test suite.
    ```bash
    pytest tests/
    ```

---

## Usage Guide 🖥️

You can interact with the project via Jupyter Notebooks for exploration or CLI for execution.

*   **Notebooks (`notebooks/`):**
    *   `01_extraction.ipynb`: SQL Data Extraction.
    *   `02_eda.ipynb`: Exploratory Data Analysis.
    *   `03_nlp_feature_engineering.ipynb`: Text Cleaning, TF-IDF Vectorization & Modeling.
    *   `04_modelling.ipynb`: Executive Summary.

*   **Live Deployment (Render):**
    *   **Dashboard:** [https://returnx-dashboard.onrender.com](https://returnx-dashboard.onrender.com)
    *   **API (Swagger UI):** [https://returnx-api.onrender.com/docs](https://returnx-api.onrender.com/docs)

*   **Command Line Interface (`main.py`):**
    
    To run the full training pipeline (ETL -> Train -> Save Artifacts):
    ```bash
    python main.py train
    ```

    To launch the interactive dashboard locally:
    ```bash
    python main.py dashboard
    ```

*   **Dashboard Features:**
    *   Real-time classification of return comments.
    *   Confidence score visualization.
    *   Pre-loaded example scenarios for testing.

---

## Results & Impact 📊

**Simulated cost impact on a batch of 4,529 returns** (modeled on historical data, not live deployment):
*   **Capture Efficiency:** **94.1%** 📈
*   **Operational Cost Reduction:** **80%** (From $9,058 to $1,807) 📉
*   **Net Savings:** **$7,250.10** 💰
*   **Return on Investment (ROI):** **401%** 🚀

**Real metric validated on actual data:**
*   **Macro F1-Score:** **0.93** (XGBoost Champion Model)

**Note:** These savings are modeled based on stated cost assumptions (manual review: $2/item, 
automated: $0.10/item, misclassification penalty: $5/error). Real impact requires live 
warehouse integration and cost validation. Real metric validated on actual data: **Macro F1-score 0.93** ✓

The system successfully automates the vast majority of returns while acting as a "Safety Valve" by routing ambiguous cases to the "Other" category for human review, minimizing expensive errors.

---

## Contributing 🤝

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  **Fork** the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a **Pull Request**

---

## Contact ✉️
For any questions, issues, or suggestions, please feel free to contact:
- Email: gbetnkom.bechir@gmail.com
- LinkedIn: [Fares Fahim Bechir Gbetnkom](https://www.linkedin.com/in/fares-fahim-bechir-gbetnkom-386a782a6)
- GitHub Issues: [Project Issues](https://github.com/GFFB0314/ReturnX-Intelligent-Returns-Classifier/issues)

---

## License ©️
**MIT License** 📝

**© 2026 Fares Gbetnkom**. This project is licensed under the **MIT License** — feel free to use, modify, and distribute it. See the full license text [here](LICENSE).

Happy Classifying! 🎯
