import streamlit as st
import numpy as np
import joblib

# ==========================================
# 1. LOAD SAVED MODELS
# ==========================================
try:
    scaler = joblib.load('scaler.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    rf_model = joblib.load('revenue_model.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found! Please run 'training.ipynb' first.")
    st.stop()

# ==========================================
# 2. APP UI (User Interface)
# ==========================================
st.set_page_config(page_title="Revenue Predictor", layout="centered")

st.title("💰 Customer Revenue Prediction")
st.write("Enter customer details to predict their potential revenue value.")

# Input Form
with st.form("prediction_form"):
    st.subheader("Customer Details (RFM)")
    
    # Recency input
    recency = st.number_input("Days Since Last Purchase (Recency)", 
                              min_value=0, value=10, help="Kam din matlab accha customer")
    
    # Frequency input
    frequency = st.number_input("Total Transactions (Frequency)", 
                                min_value=1, value=5, help="Zyada baar matlab loyal customer")
    
    # Monetary input
    monetary = st.number_input("Total Spent So Far (Monetary)", 
                               min_value=0.0, value=500.0, format="%.2f")
    
    submit_btn = st.form_submit_button("Predict Future Value")

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if submit_btn:
    try:
        # Step A: Data ko format karna
        raw_data = np.array([[recency, frequency, monetary]])
        
        # Step B: Scaling
        scaled_data = scaler.transform(raw_data)
        
        # Step C: Clustering
        cluster = kmeans_model.predict(scaled_data)[0]
        
        # Step D: Regression Prediction
        # Model input: Recency, Frequency, Cluster
        model_input = np.array([[recency, frequency, cluster]])
        
        prediction = rf_model.predict(model_input)[0]
        
        # ==========================================
        # 4. DISPLAY RESULTS
        # ==========================================
        st.success(f"✅ Prediction Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Assigned Cluster", value=f"Cluster {cluster}")
            
        with col2:
            st.metric(label="Predicted Revenue Value", value=f"₹ {prediction:.2f}")
            
        if prediction > monetary:
            st.info("📈 This customer is expected to spend MORE in the future!")
        else:
            st.warning("📉 This customer might reduce spending. Requires attention.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Built with Random Forest & K-Means Clustering")
