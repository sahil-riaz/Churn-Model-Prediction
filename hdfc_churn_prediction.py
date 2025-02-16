import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("HSBC_Customer_Churn_Dataset.csv")
    return df

df = load_data()

# Define features and target variable
X = df.drop(columns=["Customer_ID", "Churn_Risk"])
y = df["Churn_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Streamlit UI
st.set_page_config(page_title="HDFC Bank - Churn Prediction", layout="wide")
st.image("https://upload.wikimedia.org/wikipedia/en/thumb/9/99/HDFC_Bank_logo.svg/1200px-HDFC_Bank_logo.svg.png", width=300)
st.title("🔍 HDFC Bank - AI-Powered Customer Churn Prediction")
st.subheader("Identify at-risk customers and implement personalized retention strategies.")

# Model selection
st.sidebar.header("⚙️ Select Machine Learning Model")
model_type = st.sidebar.radio("Choose Model", ["Random Forest", "Gradient Boosting"])

# Train the selected model
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_type == "Gradient Boosting":
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the model
joblib.dump(model, "hdfc_churn_model.pkl")

# Display accuracy
st.sidebar.metric(label="📊 Model Accuracy", value=f"{accuracy:.2%}")

# Event-Driven Prediction Section
st.sidebar.header("⚡ Automated Churn Detection")
use_event_triggers = st.sidebar.checkbox("Enable Event-Based Prediction")

if use_event_triggers:
    st.sidebar.write("🔹 System will auto-detect churn risk from banking events.")
    df['Churn_Prediction'] = model.predict(X)
    high_risk_customers = df[df['Churn_Prediction'] == 1].head(10)
    st.write("### 🚨 High-Risk Customers Detected")
    st.dataframe(high_risk_customers)
else:
    # User Input Form
    st.sidebar.header("📝 Enter Customer Details")
    transaction_frequency = st.sidebar.slider("Transaction Frequency (per month)", 1, 50, 10)
    complaint_history = st.sidebar.slider("Complaint History (last year)", 0, 10, 2)
    service_interactions = st.sidebar.slider("Service Interactions", 0, 20, 5)
    product_usage = st.sidebar.slider("Products Used", 1, 6, 3)
    account_balance = st.sidebar.number_input("Account Balance (INR)", 1000, 500000, 100000)
    credit_limit = st.sidebar.number_input("Credit Limit (INR)", 10000, 500000, 250000)

    if st.sidebar.button("🔮 Predict Churn"):
        input_data = np.array([[transaction_frequency, complaint_history, service_interactions, product_usage, account_balance, credit_limit]])
        prediction = model.predict(input_data)[0]
        churn_probability = model.predict_proba(input_data)[0][1] * 100

        st.subheader("🛠 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Churn ({churn_probability:.2f}%)")
            st.write("### Suggested Retention Actions:")
            st.write("✅ Relationship Manager Outreach 📞")
            st.write("✅ Exclusive Deposit & Credit Limit Offers 💰")
            st.write("✅ Tailored Wealth Management Plans 📊")
        else:
            st.success(f"✅ Low Risk of Churn ({churn_probability:.2f}%)")
            st.write("Customer is likely to remain engaged with the bank.")

st.sidebar.info("🔹 Powered by HDFC Bank AI-driven analytics.")
