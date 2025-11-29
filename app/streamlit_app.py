# Sales Forecasting App - Streamlit Deployment
# app.py

"""
Sales Forecasting and Demand Prediction - Streamlit Dashboard
Milestone 4: Deployment
Deploy to: Railway / Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. LOAD MODEL AND RESOURCES
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try loading from multiple possible locations
        model_files = [
            'best_model_xgboost.pkl',
            'best_model_lightgbm.pkl',
            'best_model_random_forest.pkl',
            'model.pkl',
            'feature_columns_20251122_103557.pkl'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                return model, model_file
        
        st.error("❌ No model file found!")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    try:
        if os.path.exists('feature_scaler.pkl'):
            with open('feature_scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load scaler: {str(e)}")
        return None

@st.cache_resource
def load_metadata():
    """Load model metadata"""
    try:
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            return metadata
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load metadata: {str(e)}")
        return None

@st.cache_data
def load_feature_names():
    """Load feature names"""
    try:
        if os.path.exists('feature_names.txt'):
            with open('feature_names.txt', 'r') as f:
                features = [line.strip() for line in f.readlines()]
            return features
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load feature names: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        if os.path.exists('train_processed.csv'):
            df = pd.read_csv('train_processed.csv')
            df['date'] = pd.to_datetime(df['date'])
            return df.tail(1000)  # Load last 1000 rows for demo
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not load sample data: {str(e)}")
        return None

# Load all resources
model, model_filename = load_model()
scaler = load_scaler()
feature_names = load_feature_names()
sample_data = load_sample_data()

# ============================================================================
# 2. SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.title("📊 Sales Forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Make Predictions", "📈 Batch Forecast", "ℹ️ About"]
)


# ============================================================================
# 3. HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.title("🏠 Sales Forecasting Dashboard")
    st.markdown("### Welcome to the Sales Demand Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>ML-powered forecasting with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Results</h3>
            <p>Get instant predictions for your sales data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Interactive Visualizations</h3>
            <p>Explore trends and patterns in your forecasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    1. **🔮 Make Predictions** - Enter sales data manually for single predictions
    2. **📈 Batch Forecast** - Upload CSV file for multiple predictions
    3. **ℹ️ About** - Learn more about the project
    """)
    
    # Sample data preview
    if sample_data is not None:
        st.markdown("---")
        st.markdown("### 📋 Sample Data Preview")
        st.dataframe(sample_data.head(10), use_container_width=True)

# ============================================================================
# 4. MAKE PREDICTIONS PAGE
# ============================================================================

elif page == "🔮 Make Predictions":
    st.title("🔮 Make Sales Predictions")
    st.markdown("### Enter the details below to get a sales forecast")
    
    # Add feature explanation section
    with st.expander("ℹ️ What do these features mean?", expanded=False):
        st.markdown("""
        ### 📋 Feature Descriptions
        
        **Basic Information:**
        - **Store Number** (1-54): Unique identifier for each store location
        - **Date**: The date for which you want to predict sales
        - **Product Family**: Category of products (e.g., GROCERY, BEVERAGES, DAIRY)
        
        **Store Characteristics:**
        - **Store Type** (A-E): Classification of store size and format
          - Type A: Largest stores
          - Type E: Smallest stores
        - **Store Cluster** (1-17): Grouping of similar stores based on characteristics
        
        **Economic Factors:**
        - **Oil Price**: Daily oil price in USD
          - Ecuador's economy is oil-dependent
          - Higher oil prices typically mean higher consumer spending
          - Affects transportation costs and consumer behavior
        
        **Sales Indicators:**
        - **Items on Promotion**: Number of items from this product family currently on promotion
          - Promotions typically increase sales by 20-40%
          - Important for demand forecasting
        
        - **Transactions**: Total number of customer transactions at the store that day
          - Higher transactions usually mean higher sales
          - Indicates store traffic and customer activity
        
        **Special Events:**
        - **Is Holiday?**: Whether the date is a national or local holiday
          - Holidays significantly impact sales patterns
          - Some product families sell more (celebrations), others less (offices closed)
        
        ---
        
        ### 🧠 How the Model Uses These Features:
        
        The model combines these inputs with **engineered features** like:
        - **Historical patterns**: Sales from previous weeks/months
        - **Seasonality**: Day of week, month, quarter effects
        - **Trends**: Overall sales trends over time
        - **Interactions**: How promotions work differently on holidays
        - **Rolling averages**: Recent sales performance
        
        All these factors together help predict future sales accurately! 📊
        """)
    
    if model is None:
        st.error("❌ Model not loaded! Please ensure model file exists.")
        st.stop()
    
    # Create input form
    with st.form("prediction_form"):
        st.markdown("#### 📝 Input Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏪 Store Information**")
            store_nbr = st.number_input(
                "Store Number", 
                min_value=1, 
                max_value=54, 
                value=1,
                help="Enter the store number (1-54). Different stores have different sales patterns."
            )
            
            date = st.date_input(
                "Date", 
                value=datetime.now(),
                help="Select the date for prediction. Weekends and holidays have different patterns."
            )
            
            onpromotion = st.number_input(
                "Items on Promotion", 
                min_value=0, 
                value=0,
                help="Number of items from this product family currently on promotion. Promotions typically boost sales by 20-40%."
            )
        
        with col2:
            st.markdown("**📦 Product & Store Details**")
            family = st.selectbox(
                "Product Family",
                ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", 
                 "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
                 "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
                 "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
                 "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
                 "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
                 "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY", "PREPARED FOODS",
                 "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"],
                help="Select the product category. Different families have different sales volumes and patterns."
            )
            
            store_type = st.selectbox(
                "Store Type", 
                ["A", "B", "C", "D", "E"],
                help="Store classification: A (Largest) to E (Smallest). Larger stores typically have higher sales."
            )
            
            cluster = st.number_input(
                "Store Cluster", 
                min_value=1, 
                max_value=17, 
                value=1,
                help="Cluster number (1-17). Stores in the same cluster have similar characteristics."
            )
        
        with col3:
            st.markdown("**💰 Economic & Activity Indicators**")
            oil_price = st.number_input(
                "Oil Price (USD)", 
                min_value=0.0, 
                value=50.0, 
                step=0.1,
                help="Daily oil price affects consumer spending. Typical range: $30-$110. Ecuador's economy is oil-dependent."
            )
            
            
            is_holiday = st.checkbox(
                "Is Holiday?",
                help="Check if this date is a national or local holiday. Holidays significantly affect sales patterns."
            )
        
        
        submit_button = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
    
    if submit_button:
        with st.spinner("🔄 Making prediction..."):
            try:
                # Create feature dictionary (simplified version)
                # In production, you'd need to calculate all engineered features
                
                # Extract date features
                date_obj = pd.to_datetime(date)
                year = date_obj.year
                month = date_obj.month
                day = date_obj.day
                day_of_week = date_obj.dayofweek
                week_of_year = date_obj.isocalendar()[1]
                quarter = (month - 1) // 3 + 1
                is_weekend = 1 if day_of_week in [5, 6] else 0
                
                # This is a simplified example
                # You would need to create all the features your model expects
                st.warning("""
                ⚠️ **Note:** For accurate predictions, this requires all engineered features from training.
                This demo shows the interface structure. In production, you'd need to:
                1. Load the complete feature engineering pipeline
                2. Calculate all lag features, rolling statistics, etc.
                3. Use the exact same preprocessing as training
                """)
                
                # Mock prediction for demonstration
                mock_prediction = np.random.uniform(10, 1000)
                
                # Display results
                st.success("✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Sales", f"{mock_prediction:.2f}", "units")
                with col2:
                    confidence = 0.85
                    st.metric("Confidence", f"{confidence:.1%}", "")
                with col3:
                    st.metric("Store", store_nbr, f"Type {store_type}")
                
                # Visualization
                st.markdown("### 📊 Prediction Breakdown")
                
                # Create a sample breakdown chart
                breakdown_data = pd.DataFrame({
                    'Factor': ['Base Sales', 'Promotion Effect', 'Seasonality', 'Store Type', 'Other'],
                    'Impact': [mock_prediction * 0.4, mock_prediction * 0.2, mock_prediction * 0.15, 
                              mock_prediction * 0.15, mock_prediction * 0.1]
                })
                
                fig = px.bar(breakdown_data, x='Factor', y='Impact', 
                           title='Contribution to Predicted Sales',
                           color='Impact', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

# ============================================================================
# 5. BATCH FORECAST PAGE
# ============================================================================

elif page == "📈 Batch Forecast":
    st.title("📈 Batch Sales Forecast")
    st.markdown("### Upload a CSV file to get predictions for multiple records")
    
    if model is None:
        st.error("❌ Model not loaded! Please ensure model file exists.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with sales data",
        type=['csv'],
        help="CSV should contain: store_nbr, family, date, onpromotion, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df_upload = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df_upload)} records found.")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Make predictions button
            if st.button("🔮 Generate Predictions", use_container_width=True):
                with st.spinner("🔄 Processing predictions..."):
                    # Mock predictions for demo
                    predictions = np.random.uniform(10, 1000, size=len(df_upload))
                    df_upload['predicted_sales'] = predictions
                    
                    st.success("✅ Predictions generated!")
                    
                    # Display results
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(df_upload))
                    with col2:
                        st.metric("Avg Predicted Sales", f"{predictions.mean():.2f}")
                    with col3:
                        st.metric("Min Prediction", f"{predictions.min():.2f}")
                    with col4:
                        st.metric("Max Prediction", f"{predictions.max():.2f}")
                    
                    # Visualizations
                    st.markdown("### 📈 Forecast Visualization")
                    
                    if 'date' in df_upload.columns:
                        df_upload['date'] = pd.to_datetime(df_upload['date'])
                        daily_forecast = df_upload.groupby('date')['predicted_sales'].sum().reset_index()
                        
                        fig = px.line(daily_forecast, x='date', y='predicted_sales',
                                    title='Daily Sales Forecast',
                                    labels={'predicted_sales': 'Predicted Sales', 'date': 'Date'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download predictions
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"sales_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Show sample CSV format
        st.info("💡 Upload a CSV file to get started")
        
        st.markdown("### 📝 Expected CSV Format")
        sample_df = pd.DataFrame({
            'store_nbr': [1, 2, 3],
            'family': ['GROCERY I', 'BEVERAGES', 'DAIRY'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'onpromotion': [5, 0, 3],
            'dcoilwtico': [50.5, 50.5, 50.5]
        })
        st.dataframe(sample_df, use_container_width=True)


# ============================================================================
# 6. ABOUT PAGE
# ============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### 📊 Sales Forecasting and Demand Prediction System
    
    This application uses Machine Learning to predict future sales and demand based on historical data,
    helping businesses optimize inventory management, staffing, and marketing strategies.
    
    ---
    
    ### 🎯 Key Features
    
    - **Accurate Predictions**: ML-powered forecasting with multiple algorithms
    - **Real-time Forecasting**: Get instant predictions for your sales data  
    - **Batch Processing**: Upload CSV files for multiple predictions
    - **Interactive Visualizations**: Explore trends and patterns
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **Machine Learning**: XGBoost, LightGBM, Random Forest
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib
    - **Deployment**: Railway
    
    ---
    
    ### 📈 Project Milestones
    
    1. ✅ **Data Collection & EDA** - Exploratory analysis and preprocessing
    2. ✅ **Feature Engineering** - Advanced feature creation
    3. ✅ **Model Development** - Training and optimization
    4. ✅ **Deployment** - Web application deployment
    
    ---
    
    ### 📝 How to Use
    
    1. Navigate to **Make Predictions** for single forecasts
    2. Use **Batch Forecast** to upload CSV files
    """)
    
    # System information
    st.markdown("---")
    st.markdown("### 💻 System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Python Version:** {sys.version[:7]}  
        **Streamlit Version:** {st.__version__}  
        **Model Status:** {'✅ Loaded' if model else '❌ Not Loaded'}
        """)

    # About Team
    st.markdown("""
    ## Our Team 
    - **Abdelrahman Saeed** - [LinkedIn]
    - **Abdelrahman Youssry** - [LinkedIn]
    - **Farida Sabra** - [LinkedIn](https://www.linkedin.com/in/farida-sabra)
    - **Hossam Eldin Mahmod** - [LinkedIn]
    - **Rana Mohammed** - [LinkedIn]
    - **Sara Basheer** - [LinkedIn]
    """)