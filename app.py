import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import os

# Load data (ensure 'adult.csv' and 'adult.test.csv' are in the same directory as app.py or provide full paths)
try:
    train_data = pd.read_csv("adult.csv")
    test_data = pd.read_csv("adult.test.csv")
except FileNotFoundError:
    st.error("Error: 'adult.csv' or 'adult.test.csv' not found. Please ensure they are in the same directory as app.py.")
    st.stop()


# Data Preprocessing
columns = train_data.columns
test_data.columns = columns
df = pd.concat([train_data, test_data], axis=0)

for col in columns:
    if df[col].dtype in ['O']:
        df[col] = df[col].str.strip()

df.replace("?'", np.nan, inplace=True)
df[df['Workclass'] == 'Never-worked'] = df[df['Workclass'] == 'Never-worked'].fillna('No-occupation')
df['Native Country'].fillna('United-States', inplace=True)
df.dropna(inplace=True)
df.drop(columns=['Capital Gain', 'capital loss', 'Native Country'], inplace=True)
df.replace({'Income': {">50K.": ">50K", "<=50K.": "<=50K"}}, inplace=True)

mask = df['Final Weight'] > 0.6 * 1000000
df.loc[mask, 'Final Weight'] = np.ceil(df['Final Weight'].mean())


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ['Income', 'Gender']:
    df = label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    encoded_data = dataframe.copy()
    for col in categorical_cols:
        dumm = pd.get_dummies(dataframe[col], prefix=col, dtype=int, drop_first=drop_first)
        del encoded_data[col]
        encoded_data = pd.concat([encoded_data, dumm], axis=1)
    return encoded_data

categorical_cols_to_encode = ['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race']
df = one_hot_encoder(df, categorical_cols_to_encode)

X = df.drop("Income", axis=1)
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit App
st.title("Census Income Prediction")

# Sidebar for Model Parameters
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Max Depth:", min_value=1, max_value=50, value=40)

# Sidebar for Input Features (Example)
st.sidebar.header("Input Features for Prediction")
age = st.sidebar.slider("Age:", min_value=int(train_data['Age'].min()), max_value=int(train_data['Age'].max()), value=int(train_data['Age'].mean()))
hours_per_week = st.sidebar.slider("Hours per Week:", min_value=1, max_value=99, value=40)

# Create selectbox widgets for one-hot encoded categorical features
workclass_cols = [col for col in df.columns if col.startswith('Workclass_')]
selected_workclass = st.sidebar.selectbox("Workclass", workclass_cols)

education_cols = [col for col in df.columns if col.startswith('Education_')]
selected_education = st.sidebar.selectbox("Education", education_cols)

marital_status_cols = [col for col in df.columns if col.startswith('Marital Status_')]
selected_marital_status = st.sidebar.selectbox("Marital Status", marital_status_cols)

occupation_cols = [col for col in df.columns if col.startswith('Occupation_')]
selected_occupation = st.sidebar.selectbox("Occupation", occupation_cols)

relationship_cols = [col for col in df.columns if col.startswith('Relationship_')]
selected_relationship = st.sidebar.selectbox("Relationship", relationship_cols)

race_cols = [col for col in df.columns if col.startswith('Race_')]
selected_race = st.sidebar.selectbox("Race", race_cols)


DT_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

if st.button("Predict"):
    # Train the model
    DT_model.fit(X_train, y_train)
    y_pred_proba_test = DT_model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred_proba_test[:, 1])
    st.write(f"AUC Score (on test data): {auc_score:.3f}")

    # Plotting ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    st.pyplot(plt)

    # Make a single prediction based on user input
    user_input = pd.DataFrame([[age, hours_per_week] + [0] * (len(X.columns) - 2)], columns=X.columns)

    # Set the selected categorical features to 1
    user_input[selected_workclass] = 1
    user_input[selected_education] = 1
    user_input[selected_marital_status] = 1
    user_input[selected_occupation] = 1
    user_input[selected_relationship] = 1
    user_input[selected_race] = 1

    prediction_proba = DT_model.predict_proba(user_input)
    predicted_class = DT_model.predict(user_input)[0]
    prediction_label = "Income > $50K" if predicted_class == 1 else "Income <= $50K"

    st.subheader("Prediction based on your input:")
    st.write(f"Probability of Income > $50K: {prediction_proba[0][1]:.2f}")
    st.write(f"Predicted Income: {prediction_label}")

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(df.head())