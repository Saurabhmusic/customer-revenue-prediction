#  Customer Revenue Prediction & Segmentation App

An End-to-End Machine Learning project that segments customers based on their buying behavior and predicts their future revenue potential. Built using **Python, Scikit-Learn, and Streamlit**.

## 🔗 Live Demo
Check out the live app here: https://customer-revenue-prediction-ic2krklxewth4pssktbvaw.streamlit.app/

---

##  Project Overview
In e-commerce, treating every customer the same doesn't work. Businesses need to identify who their **Loyal Customers** are and who might churn (stop buying).

This project solves that problem by:
1.  **Segmenting Customers:** Grouping them into clusters based on purchasing behavior.
2.  **Predicting Revenue:** Estimating how much a customer is likely to spend in the future.

##  My Approach (The Workflow)

I didn't just fit a model on raw data. I followed a proper Data Science lifecycle:

### 1. Data Cleaning & RFM Analysis
The raw data contained transaction logs. I converted this into a customer-level dataset using **RFM Analysis**:
* **Recency:** How many days ago was their last purchase?
* **Frequency:** How many times have they purchased?
* **Monetary:** How much have they spent in total?

### 2. Preprocessing & Scaling (Important!)
Since 'Monetary' values (e.g., 5000) are much larger than 'Frequency' (e.g., 5), algorithms like K-Means get biased.
* **Solution:** I applied **Standard Scaler** to bring all features to the same scale before clustering.

### 3. Machine Learning Models
* **Unsupervised Learning (Clustering):** Used **K-Means Clustering** to group similar customers.
* **Supervised Learning (Regression):** Used **Random Forest Regressor** to predict the LTV (Lifetime Value) of a customer.

### 4. Deployment
Built an interactive web app using **Streamlit** so that non-technical users can also use this model to get insights.

---

##  Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Model Saving:** Joblib
* **Deployment:** Streamlit Community Cloud

---

##  Project Structure
