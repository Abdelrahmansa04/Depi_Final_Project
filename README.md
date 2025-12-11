# 📊 Sales Forecasting Dashboard

A powerful, machine learning-powered web application for predicting sales and demand forecasting. Built with Streamlit and TensorFlow/Keras, this dashboard enables businesses to make data-driven decisions for inventory management, staffing, and marketing strategies.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Features

### Core Functionality
- **🔮 Single Predictions**: Interactive form-based predictions with real-time results
- **📈 Batch Forecasting**: Upload CSV files for bulk predictions with progress tracking
- **📊 Interactive Visualizations**: Dynamic Plotly charts for data exploration and analysis
- **💾 Export Capabilities**: Download predictions as CSV files and charts as PNG images
- **📚 Feature Guide**: Comprehensive documentation of all model features

### Key Highlights
- **103-Feature Neural Network Model**: Advanced deep learning architecture for accurate predictions
- **Multi-Store Support**: Handles predictions across 54 different store locations
- **33 Product Categories**: Supports diverse product families from groceries to electronics
- **Economic Factors Integration**: Incorporates oil prices and economic indicators
- **Event-Aware Forecasting**: Accounts for holidays, events, and special circumstances

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Depi_Final_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   - Make sure `app/model_nn.keras` is present in the project directory

4. **Run the application**
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Access the dashboard**
   - Open your browser and navigate to `http://localhost:8501`

---

## 📖 Usage Guide

### Single Prediction

1. Navigate to **"🔮 Make Predictions"** from the sidebar
2. Fill in the required fields:
   - Store Number (1-54)
   - Product Family (select from dropdown)
   - Date
   - Promotion status
   - Oil price (DCOILWTICO)
   - Holiday/Event/Work Day/Earthquake flags
3. Click **"Predict 🔮"** to get instant results

### Batch Forecasting

1. Navigate to **"📈 Batch Forecast"** from the sidebar
2. Prepare your CSV file with one of these formats:

   **Option 1: Raw Input Columns** (Recommended)
   ```csv
   store_nbr,family,date,onpromotion,dcoilwtico,is_holiday,is_event,is_work_day,is_earthquake
   1,GROCERY I,2024-01-15,1.0,0.25,0,0,1,0
   2,BEVERAGES,2024-01-16,0.0,0.26,0,0,1,0
   ```

   **Option 2: Preprocessed Features**
   - CSV with exactly 103 feature columns (already processed)

3. Upload your CSV file
4. Review the data preview and validation
5. Click **"🚀 Predict Sales"** to generate predictions
6. View interactive charts and download results

---

## 🏗️ Project Structure

```
Depi_Final_Project/
│
├── app/
│   ├── streamlit_app.py      # Main Streamlit application
│   └── model_nn.keras        # Trained neural network model
│
├── artifacts/
│   ├── scaler_*.pkl          # Feature scalers for preprocessing
│   └── *.pkl                 # Additional preprocessing artifacts
│
├── src/
│   └── config.py             # Configuration settings
│
├── docs/                     # Documentation files
│
├── requirements.txt          # Python dependencies
├── runtime.txt              # Python runtime version
├── Procfile                 # Deployment configuration
└── README.md                # This file
```

---

## 🔧 Technical Details

### Model Architecture

- **Type**: Deep Neural Network (Keras/TensorFlow)
- **Input Features**: 103 features including:
  - Store one-hot encoding (54 stores)
  - Product family one-hot encoding (33 categories)
  - Temporal features (day, month, year, day of week)
  - Economic indicators (oil prices)
  - Event flags (holidays, promotions, earthquakes)

### Feature Engineering

The model processes raw inputs into a 103-dimensional feature vector:
- **Index 0-8**: Numeric features (promotions, oil price, flags, normalized date components)
- **Index 9-62**: Store one-hot encoding (54 stores)
- **Index 63-95**: Product family one-hot encoding (33 categories)
- **Index 96-102**: Day of week one-hot encoding (7 days)

### Supported Product Families

- AUTOMOTIVE, BABY CARE, BEAUTY, BEVERAGES, BOOKS
- BREAD/BAKERY, CELEBRATION, CLEANING, DAIRY, DELI
- EGGS, FROZEN FOODS, GROCERY I, GROCERY II, HARDWARE
- HOME AND KITCHEN I/II, HOME APPLIANCES, HOME CARE
- LADIESWEAR, LAWN AND GARDEN, LINGERIE
- LIQUOR,WINE,BEER, MAGAZINES, MEATS, PERSONAL CARE
- PET SUPPLIES, PLAYERS AND ELECTRONICS, POULTRY
- PREPARED FOODS, PRODUCE, SCHOOL AND OFFICE SUPPLIES, SEAFOOD

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
joblib
tensorflow==2.20.0
plotly
plotly.express
kaleido
```

---

## 🎨 Features Overview

### Dashboard Pages

1. **🏠 Home**: Welcome page with quick start guide
2. **📚 Feature Guide**: Detailed documentation of model features
3. **🔮 Make Predictions**: Single prediction interface
4. **📈 Batch Forecast**: Bulk prediction with CSV upload
5. **ℹ️ About**: Project information and team details

### Visualization Features

- Interactive Plotly charts for prediction analysis
- Time-series visualization of sales forecasts
- Color-coded charts by product family
- Downloadable chart images (PNG format)

---

## 👥 Team

This project was developed by:

- **Abdelrahman Saeed** - [LinkedIn](https://www.linkedin.com/in/abdelrahman-abdelraouf004/)
- **Abdelrahman Youssry** - [LinkedIn](https://www.linkedin.com/in/abdelrahman-yousry-271816269/)
- **Farida Sabra** - [LinkedIn](https://www.linkedin.com/in/farida-sabra)
- **Hossam Eldin Mahmod** - [LinkedIn](http://linkedin.com/in/hossam-eldin-m-hmady)
- **Rana Mohammed** - [LinkedIn](https://www.linkedin.com/in/rana-mohammed1)
- **Sara Basheer** - [LinkedIn](http://linkedin.com/in/sara-basheer)

---

## 🔍 Model Features Explained

### Store Features
- **54 Store Locations**: Each store has unique characteristics affecting sales patterns
- **Geographic Variations**: Different demographics and customer bases

### Product Features
- **33 Product Categories**: Diverse product families with varying sales patterns
- **Category-Specific Trends**: Each family has unique seasonality and demand patterns

### Economic Factors
- **Oil Price (DCOILWTICO)**: Critical for Ecuador's economy, affects consumer spending
- **Correlation Impact**: Medium-high correlation (10-20%) with sales

### Time Features
- **Temporal Patterns**: Day, month, year, and day of week encoding
- **Seasonality**: Captures weekly and seasonal sales patterns
- **Holiday Effects**: Accounts for holiday-driven sales spikes

### Promotional Features
- **Promotion Flags**: Binary indicators for promotional campaigns
- **Impact**: Promotions can increase sales by 20-150% depending on product category

---

## 📝 Notes

- The model expects normalized inputs and handles feature engineering automatically
- Predictions are constrained to non-negative values (sales cannot be negative)
- The application caches the model for improved performance
- CSV files should follow the specified format for optimal results

---

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `app/model_nn.keras` exists in the correct location
- Check that TensorFlow is properly installed
- Verify Python version compatibility (3.11+)

### CSV Upload Issues
- Verify column names match exactly (case-sensitive)
- Ensure date format is YYYY-MM-DD
- Check that numeric values are properly formatted

### Chart Download Not Working
- Install kaleido package: `pip install kaleido`
- Ensure plotly is up to date: `pip install --upgrade plotly`

---

## 📄 License

This project is part of an academic/research initiative. Please refer to the project documentation for licensing details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [TensorFlow/Keras](https://www.tensorflow.org/)
- Visualizations by [Plotly](https://plotly.com/)

---

## 📧 Contact

For questions or support, please contact the development team through their LinkedIn profiles listed above.

---

