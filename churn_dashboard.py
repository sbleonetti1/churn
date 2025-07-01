import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Title
st.title("📉 Telco Customer Churn Dashboard")

# Upload CSV or use default
df = pd.read_csv("Telco-Customer-Churn.csv")  # Replace with your local path

# Basic cleaning
# Convert 'TotalCharges' to numeric (some are missing or blank)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Sidebar filters
st.sidebar.header("📊 Filters")
contract_filter = st.sidebar.multiselect(
    "Select Contract Types", options=df['Contract'].unique(), default=df['Contract'].unique()
)
df = df[df['Contract'].isin(contract_filter)]

# Show basic stats
st.subheader("📈 Churn Rate")
churn_rate = df['Churn'].mean() * 100
st.metric(label="Overall Churn Rate", value=f"{churn_rate:.2f}%")

# Visuals: Distribution of numerical features
st.subheader("📊 Feature Distributions")
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in num_cols:
    fig, ax = plt.subplots()
    sns.histplot(df, x=col, hue='Churn', bins=30, kde=True, ax=ax)
    ax.set_title(f"{col} distribution by churn")
    st.pyplot(fig)

# Heatmap for correlations
st.subheader("🔍 Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Optional: Show full data
with st.expander("📄 View Data Table"):
    st.dataframe(df)
    
#Confusion matrix
cm_df = pd.read_csv('confusion_matrix.csv')

#Plot confusion matrix heatmap
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

#Classification report

with open('classification_report.json') as f:
    report = json.load(f)

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)
