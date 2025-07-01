import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Title
st.title("📉 Customer Churn Dashboard")

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload your churn CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Telco-Customer-Churn.csv")  # Replace with your local path

# Basic cleaning
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
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