"""
Shopper Spectrum - Streamlit App
Product Recommendation & Customer Segmentation
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Product recommendation cards */
    .product-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateX(10px);
        border-color: rgba(255, 255, 255, 0.6);
        box-shadow: 0 6px 30px rgba(0, 0, 0, 0.3);
    }
    
    .product-title {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .product-code {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        font-family: 'Courier New', monospace;
    }
    
    .product-freq {
        color: #ffd700;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Cluster result card */
    .cluster-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(245, 87, 108, 0.4);
        animation: fadeInUp 0.6s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .cluster-label {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .cluster-description {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }
    
    /* Headers */
    h1 {
        color: #fff;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #fff;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: rgba(255, 255, 255, 0.95);
        font-weight: 600;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        color: #fff;
        font-size: 1rem;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
    }
    
    /* Labels */
    label {
        color: #fff !important;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_similarity_data():
    """Load product similarity data"""
    try:
        with open('models/product_similarity.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Product similarity data not found. Please run scripts/prepare_data.py first.")
        return None

@st.cache_resource
def load_segmentation_model():
    """Load customer segmentation model"""
    try:
        with open('models/customer_segmentation_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Customer segmentation model not found.")
        return None

# Helper functions
def get_product_recommendations(product_name, similarity_data, top_n=5):
    """Get product recommendations based on co-occurrence"""
    if not similarity_data:
        return []
    
    product_dict = similarity_data['product_dict']
    co_occurrence = similarity_data['co_occurrence']
    product_frequency = similarity_data['product_frequency']
    
    # Find matching products
    product_name = product_name.strip().upper()
    matching_products = [(code, desc) for code, desc in product_dict.items() 
                        if product_name in desc]
    
    if not matching_products:
        return None
    
    # Use the first match
    stock_code = matching_products[0][0]
    
    if stock_code not in co_occurrence:
        return []
    
    # Get recommendations
    similar_products = co_occurrence[stock_code]
    sorted_products = sorted(similar_products.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    recommendations = []
    for prod_code, count in sorted_products:
        if prod_code in product_dict:
            recommendations.append({
                'code': prod_code,
                'description': product_dict[prod_code],
                'co_occurrence': count,
                'frequency': product_frequency.get(prod_code, 0)
            })
    
    return recommendations

def get_cluster_label(cluster_num):
    """Map cluster number to business label"""
    labels = {
        0: ("🌟 High-Value Customers", "Premium customers with frequent purchases and high spending. These are your most valuable customers."),
        1: ("👥 Regular Customers", "Steady customers with moderate purchase frequency and spending. Core customer base."),
        2: ("🛒 Occasional Shoppers", "Infrequent shoppers with lower spending. Potential for growth with targeted campaigns."),
        3: ("⚠️ At-Risk Customers", "Customers with declining activity. Require re-engagement strategies.")
    }
    return labels.get(cluster_num, ("Unknown Cluster", "No description available"))

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🛍️ Shopper Spectrum</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8); font-size: 1.2rem;'>AI-Powered Product Recommendations & Customer Insights</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        module = st.radio(
            "Select Module:",
            ["🎯 Product Recommendations", "👥 Customer Segmentation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app provides:
        - **Product Recommendations**: Find similar products based on purchase patterns
        - **Customer Segmentation**: Classify customers using RFM analysis
        """)
    
    # Module 1: Product Recommendations
    if module == "🎯 Product Recommendations":
        st.markdown("## 🎯 Product Recommendation Module")
        st.markdown("Find similar products based on collaborative purchase patterns")
        
        # Load data
        similarity_data = load_similarity_data()
        
        if similarity_data:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                product_input = st.text_input(
                    "Enter Product Name:",
                    placeholder="e.g., WHITE HANGING HEART T-LIGHT HOLDER",
                    help="Enter a product name or partial name to search"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                search_button = st.button("🔍 Get Recommendations", use_container_width=True)
            
            if search_button and product_input:
                with st.spinner("Finding similar products..."):
                    recommendations = get_product_recommendations(product_input, similarity_data)
                    
                    if recommendations is None:
                        st.warning(f"❌ No products found matching '{product_input}'. Please try a different search term.")
                    elif len(recommendations) == 0:
                        st.info("ℹ️ No recommendations available for this product yet.")
                    else:
                        st.success(f"✅ Found {len(recommendations)} similar products!")
                        st.markdown("### 🎁 Recommended Products")
                        
                        for i, prod in enumerate(recommendations, 1):
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-title">#{i} {prod['description']}</div>
                                <div class="product-code">Stock Code: {prod['code']}</div>
                                <div class="product-freq">⭐ Bought together {prod['co_occurrence']} times | Total purchases: {prod['frequency']}</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show popular products
            if not product_input:
                st.markdown("### 🔥 Popular Products")
                st.info("Enter a product name above to get personalized recommendations!")
                
                top_products = similarity_data['top_products'][:10]
                product_dict = similarity_data['product_dict']
                
                cols = st.columns(2)
                for idx, code in enumerate(top_products):
                    with cols[idx % 2]:
                        if code in product_dict:
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-title">{product_dict[code]}</div>
                                <div class="product-code">{code}</div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Module 2: Customer Segmentation
    else:
        st.markdown("## 👥 Customer Segmentation Module")
        st.markdown("Predict customer segment based on RFM (Recency, Frequency, Monetary) analysis")
        
        # Load model
        model = load_segmentation_model()
        
        if model:
            st.markdown("### 📝 Enter Customer Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recency = st.number_input(
                    "📅 Recency (days)",
                    min_value=0,
                    max_value=1000,
                    value=30,
                    help="Days since last purchase"
                )
            
            with col2:
                frequency = st.number_input(
                    "🔄 Frequency",
                    min_value=1,
                    max_value=10000,
                    value=50,
                    help="Number of purchases"
                )
            
            with col3:
                monetary = st.number_input(
                    "💰 Monetary (£)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=1000.0,
                    step=10.0,
                    help="Total spend in GBP"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                predict_button = st.button("🎯 Predict Cluster", use_container_width=True)
            
            if predict_button:
                with st.spinner("Analyzing customer profile..."):
                    # Prepare input
                    input_data = np.array([[recency, frequency, monetary]])
                    
                    # Predict
                    cluster = model.predict(input_data)[0]
                    label, description = get_cluster_label(cluster)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display result
                    st.markdown(f"""
                    <div class="cluster-result">
                        <div class="cluster-label">{label}</div>
                        <div class="cluster-description">{description}</div>
                        <br>
                        <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                            Cluster ID: {cluster}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("### 📊 Customer Metrics Summary")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Recency", f"{recency} days", 
                                 delta="Recent" if recency < 30 else "Inactive",
                                 delta_color="normal" if recency < 30 else "inverse")
                    
                    with metric_col2:
                        st.metric("Frequency", f"{frequency} purchases",
                                 delta="High" if frequency > 50 else "Low",
                                 delta_color="normal" if frequency > 50 else "inverse")
                    
                    with metric_col3:
                        st.metric("Monetary", f"£{monetary:,.2f}",
                                 delta="High Value" if monetary > 1000 else "Standard",
                                 delta_color="normal" if monetary > 1000 else "inverse")

if __name__ == "__main__":
    main()
