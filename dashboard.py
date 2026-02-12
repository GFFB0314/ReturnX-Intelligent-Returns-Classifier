"""
ReturnX Intelligent Returns Classifier - Streamlit Dashboard
Main entry point for the production application.
"""

import streamlit as st
import pandas as pd
from src.inference import load_model_artifacts, predict_category

# Page configuration
st.set_page_config(
    page_title="ReturnX - Returns Classifier",
    page_icon="📦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .category-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .defect { background-color: #ff4444; color: white; }
    .sizing { background-color: #ff9800; color: white; }
    .style { background-color: #9c27b0; color: white; }
    .other { background-color: #2196f3; color: white; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📦 ReturnX Intelligent Returns Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated routing of product returns using Machine Learning</div>', unsafe_allow_html=True)

# Load all artifacts (cached)
try:
    model, label_encoder, tfidf = load_model_artifacts()
    st.success("✅ Model system loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model system: {str(e)}")
    st.stop()

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This dashboard classifies product returns into four categories:
    
    - **🔴 Defect**: Product has manufacturing defects
    - **🟠 Sizing**: Product doesn't fit properly
    - **🟣 Style**: Customer doesn't like the design/color
    - **🔵 Other**: Changed mind, duplicate order, etc.
    """)
    
    st.header("📊 Model Performance")
    st.metric("F1-Score", "0.93")
    st.metric("Cost Savings", "80%")
    st.metric("ROI", "401%")
    
    st.header("🎯 Example Complaints")
    example_type = st.selectbox(
        "Load example:",
        ["Select...", "Defect", "Sizing", "Style", "Other"]
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Customer Complaint")
    
    # Example complaints mapping
    examples = {
        "Defect": {
            "text": "The zipper broke after one use and there's a hole in the fabric. Very disappointed with the quality.",
            "age": 40,
            "rating": 2
        },
        "Sizing": {
            "text": "The shirt is too small. I ordered a large but it fits like a medium. Runs very small.",
            "age": 35,
            "rating": 3
        },
        "Style": {
            "text": "The color doesn't match the picture at all. It's much darker and the design looks different.",
            "age": 45,
            "rating": 3
        },
        "Other": {
            "text": "I changed my mind and don't need this anymore. Ordered by mistake.",
            "age": 30,
            "rating": 4
        }
    }
    
    # Load example if selected
    default_text = ""
    default_age = 40
    default_rating = 3
    
    if example_type != "Select..." and example_type in examples:
        default_text = examples[example_type]["text"]
        default_age = examples[example_type]["age"]
        default_rating = examples[example_type]["rating"]
    
    # Input fields
    complaint_text = st.text_area(
        "Enter customer complaint:",
        value=default_text,
        height=150,
        placeholder="Example: The zipper broke and there's a hole in the fabric..."
    )
    
    col_age, col_rating = st.columns(2)
    
    with col_age:
        age = st.number_input(
            "Customer Age:",
            min_value=18,
            max_value=99,
            value=default_age,
            step=1
        )
    
    with col_rating:
        rating = st.selectbox(
            "Product Rating:",
            options=[1, 2, 3, 4, 5],
            index=default_rating - 1
        )
    
    # Predict button
    predict_button = st.button("🔍 Classify Return", type="primary", use_container_width=True)

with col2:
    st.header("📊 Prediction Results")
    
    if predict_button:
        if not complaint_text.strip():
            st.error("⚠️ Please enter a customer complaint")
        else:
            with st.spinner("Analyzing complaint..."):
                try:
                    # Make prediction
                    result = predict_category(
                        complaint_text=complaint_text,
                        age=age,
                        rating=rating,
                        model=model,
                        label_encoder=label_encoder,
                        tfidf=tfidf
                    )
                    
                    # Display predicted category
                    category = result["category"]
                    confidence = result["confidence"]
                    
                    # Color mapping
                    color_map = {
                        "Defect": "defect",
                        "Sizing": "sizing",
                        "Style": "style",
                        "Other": "other"
                    }
                    
                    st.markdown(
                        f'<div class="category-badge {color_map[category]}">{category}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Show all probabilities
                    st.subheader("All Categories")
                    probs = result["probabilities"]
                    
                    for cat in ["Defect", "Sizing", "Style", "Other"]:
                        prob = probs.get(cat, 0)
                        st.progress(prob, text=f"{cat}: {prob:.1%}")
                    
                    # Additional info
                    st.info(f"""
                    **Routing Decision**: {category}
                    
                    This return should be routed to the **{category}** processing queue.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.exception(e)
    else:
        st.info("👈 Enter a complaint and click 'Classify Return' to see predictions")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>ReturnX Intelligent Returns Engine</strong></p>
    <p>Powered by XGBoost | F1-Score: 0.93 | 401% ROI</p>
</div>
""", unsafe_allow_html=True)