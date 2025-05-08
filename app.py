import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Untuk mengatasi imbalance kelas (jika diperlukan)

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stAppViewContainer"] {
        overflow: hidden; /* Untuk mencegah scrollbar akibat footer */
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: black; /* Atau warna lain yang sesuai dengan tema Anda */
        text-align: left;
        padding: 10px;
        font-size: small;
        z-index: 9999; /* Pastikan di atas elemen lain */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- End Konfigurasi Halaman ---

# Load data
try:
    train_data = pd.read_csv("adult.csv")
    test_data = pd.read_csv("adult.test.csv")
except FileNotFoundError:
    st.error("Error: 'adult.csv' or 'adult.test.csv' not found.")
    st.stop()

# Preprocessing
columns = train_data.columns
test_data.columns = columns
df = pd.concat([train_data, test_data], axis=0)
for col in columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
df.replace("?", np.nan, inplace=True)
df['Workclass'] = df['Workclass'].fillna('Unknown')
df['Native Country'] = df['Native Country'].fillna('United-States')
df.dropna(inplace=True)
df.drop(columns=['Capital Gain', 'capital loss', 'Education'], inplace=True)
df.replace({'Income': {">50K.": ">50K", "<=50K.": "<=50K"}}, inplace=True)
for col in ['Income', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Native Country'], drop_first=True)
X = df.drop('Income', axis=1)
y = df['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tangani Imbalance Kelas (SMOTE - Oversampling kelas minoritas)
# Aktifkan bagian ini jika distribusi kelas sangat tidak seimbang
try:
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    X_train_for_model = X_train_resampled
    y_train_for_model = y_train_resampled
    # st.write("Menggunakan SMOTE untuk mengatasi imbalance kelas.")
except ImportError:
    X_train_for_model = X_train
    y_train_for_model = y_train
    st.warning("Library imbalanced-learn tidak terinstal. Tidak dapat menggunakan SMOTE.")

# Train model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_for_model, y_train_for_model)

# Evaluasi Model pada Data Uji (dipindahkan ke sini)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit App
st.title("Census Income Prediction using XGBoost")
st.subheader("Kelompok 5 - RAYAN")
# Initialize session state for default values
if 'age' not in st.session_state:
    st.session_state['age'] = int(train_data['Age'].mean())
if 'education_level' not in st.session_state:
    st.session_state['education_level'] = "Bachelors"
if 'hours_per_week' not in st.session_state:
    st.session_state['hours_per_week'] = int(train_data[[col for col in train_data.columns if 'Hours' in col and 'Week' in col][0]].mean())
if 'occupation' not in st.session_state:
    st.session_state['occupation'] = sorted(train_data['Occupation'].unique().tolist())[0]
if 'marital_status' not in st.session_state:
    st.session_state['marital_status'] = sorted(train_data['Marital Status'].unique().tolist())[0]
if 'relationship' not in st.session_state:
    st.session_state['relationship'] = sorted(train_data['Relationship'].unique().tolist())[0]

# Layout using columns to center the input form
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.subheader("Input Features for Prediction")
    age = st.number_input("Age:", min_value=int(train_data['Age'].min()), max_value=int(train_data['Age'].max()), value=st.session_state['age'], key='age_input')

    # Dropdown untuk Education Level
    education_map = {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
        "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
        "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
    }
    education_options = list(education_map.keys())
    education_level = st.selectbox("Education Level:", education_options, index=education_options.index(st.session_state['education_level']) if st.session_state['education_level'] in education_options else 0, key='education_level_input')
    education_num = education_map[education_level]

    hours_per_week_col = [col for col in train_data.columns if 'Hours' in col and 'Week' in col][0]
    hours_per_week = st.number_input("Hours per Week:", min_value=1, max_value=99, value=st.session_state['hours_per_week'], key='hours_per_week_input')

    # Dropdown untuk Occupation
    occupation_options = sorted(train_data['Occupation'].unique().tolist())
    occupation_map = {option: 'Occupation_' + option.replace(' ', '_').lower() for option in occupation_options}
    occupation_list = list(occupation_map.keys())
    occupation_selection = st.selectbox("Occupation:", occupation_list, index=occupation_list.index(st.session_state['occupation']) if st.session_state['occupation'] in occupation_list else 0, key='occupation_input')
    occupation_encoded = occupation_map[occupation_selection]

    # Dropdown untuk Marital Status
    marital_status_options = sorted(train_data['Marital Status'].unique().tolist())
    marital_status_map = {option: 'Marital Status_' + option.replace(' ', '_').lower() for option in marital_status_options}
    marital_status_list = list(marital_status_map.keys())
    marital_status_selection = st.selectbox("Marital Status:", marital_status_list, index=marital_status_list.index(st.session_state['marital_status']) if st.session_state['marital_status'] in marital_status_list else 0, key='marital_status_input')
    marital_status_encoded = marital_status_map[marital_status_selection]

    # Dropdown untuk Relationship
    relationship_options = sorted(train_data['Relationship'].unique().tolist())
    relationship_map = {option: 'Relationship_' + option.replace(' ', '_').lower() for option in relationship_options}
    relationship_list = list(relationship_map.keys())
    relationship_selection = st.selectbox("Relationship:", relationship_list, index=relationship_list.index(st.session_state['relationship']) if st.session_state['relationship'] in relationship_list else 0, key='relationship_input')
    relationship_encoded = relationship_map[relationship_selection]

    # Prediction and Reset buttons centered
    predict_button_col, reset_button_col = st.columns(2)
    with predict_button_col:
        if st.button("Predict"):
            user_input = pd.DataFrame(0, index=[0], columns=X_train.columns)
            user_input['Age'] = age
            user_input['EducationNum'] = education_num
            user_input[hours_per_week_col] = hours_per_week

            if occupation_encoded in user_input.columns:
                user_input[occupation_encoded] = 1
            if marital_status_encoded in user_input.columns:
                user_input[marital_status_encoded] = 1
            if relationship_encoded in user_input.columns:
                user_input[relationship_encoded] = 1

            prediction_proba = xgb_model.predict_proba(user_input)[:, 1]
            predicted_class = xgb_model.predict(user_input)[0]
            prediction_label = "Income > $50K" if predicted_class == 1 else "Income <= $50K"
            st.subheader("Prediction Result:")
            st.write(f"Predicted Income: {prediction_label}")

    with reset_button_col:
        if st.button("Reset"):
            st.session_state['age'] = int(train_data['Age'].mean())
            st.session_state['education_level'] = "Bachelors"
            st.session_state['hours_per_week'] = int(train_data[[col for col in train_data.columns if 'Hours' in col and 'Week' in col][0]].mean())
            st.session_state['occupation'] = sorted(train_data['Occupation'].unique().tolist())[0]
            st.session_state['marital_status'] = sorted(train_data['Marital Status'].unique().tolist())[0]
            st.session_state['relationship'] = sorted(train_data['Relationship'].unique().tolist())[0]
            st.rerun()

# Evaluation metrics (displayed in the main area)
# st.subheader("Model Evaluation on Test Data:")
# st.write(f"Akurasi: {accuracy:.2f}")
# st.text(report)

st.markdown(
    """
    <div class="footer">
        Dibuat Oleh Kelompok 5 - RAYAN
    </div>
    """,
    unsafe_allow_html=True,
)

# st.subheader("Average Income Distribution in Training Data:")
# average_income = train_data['Income'].value_counts(normalize=True)
# st.write(f"<=50K: {average_income[0]:.2f}")
# st.write(f">50K: {average_income[1]:.2f}")