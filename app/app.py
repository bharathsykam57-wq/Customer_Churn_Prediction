import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, '..', 'data', 'Customer-Churn.csv')
MODEL_PATH = os.path.join(APP_DIR, '..', 'models', 'ada_boost_churn_model.pkl')


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache_resource
def build_preprocessor(df):
    telco = df.copy()
    telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
    telco.dropna(how='any', inplace=True)

    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['1-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    telco['tenure_bin'] = pd.cut(telco['tenure'], bins=bins, labels=labels, include_lowest=True)

    X_template = telco.drop(columns=['customerID', 'Churn', 'tenure'])
    X_template = pd.get_dummies(X_template, drop_first=True)

    scaler = StandardScaler()
    scaler.fit(X_template)

    return {
        'template_columns': list(X_template.columns),
        'scaler': scaler,
        'tenure_bins': (bins, labels),
        'sample_df': telco
    }


def preprocess_input(user_input, prep):
    df_in = pd.DataFrame([user_input])

    bins, labels = prep['tenure_bins']
    df_in['tenure_bin'] = pd.cut(df_in['tenure'], bins=bins, labels=labels, include_lowest=True)
    df_in = df_in.drop(columns=['tenure'])

    df_in_enc = pd.get_dummies(df_in, drop_first=True)
    df_in_enc = df_in_enc.reindex(columns=prep['template_columns'], fill_value=0)

    X_scaled = prep['scaler'].transform(df_in_enc)
    return X_scaled


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def is_categorical(series):
    """
    Returns True if the column should be rendered as a selectbox.
    Uses pd.api.types.is_string_dtype to handle both older pandas 'object'
    and newer pandas 'str' dtypes correctly.
    Also treats low-cardinality numeric columns (e.g. SeniorCitizen 0/1) as categorical.
    """
    if pd.api.types.is_string_dtype(series):
        return True
    if str(series.dtype) == 'category':
        return True
    if pd.api.types.is_numeric_dtype(series) and series.nunique() < 5:
        return True
    return False


def main():
    st.set_page_config(page_title='Churn Prediction', layout='centered')
    st.title('Telecom Customer Churn Prediction')

    df = load_data(DATA_PATH)
    prep = build_preprocessor(df)
    model = load_model(MODEL_PATH)

    st.markdown('### Provide customer details to get churn prediction')

    sample = prep['sample_df'].copy()
    sample['TotalCharges'] = pd.to_numeric(sample['TotalCharges'], errors='coerce')

    cols_to_ask = [
        c for c in sample.columns
        if c not in ['customerID', 'Churn', 'tenure', 'tenure_bin']
    ]

    with st.form('input_form'):
        user_input = {}

        user_input['tenure'] = st.slider(
            'Tenure (months)',
            min_value=int(sample['tenure'].min()),
            max_value=int(sample['tenure'].max()),
            value=12
        )

        for col in cols_to_ask:
            if is_categorical(sample[col]):
                opts = sorted(sample[col].dropna().astype(str).unique().tolist())
                user_input[col] = st.selectbox(col, opts)
            else:
                if pd.api.types.is_integer_dtype(sample[col]):
                    minv = int(sample[col].min())
                    maxv = int(sample[col].max())
                    default = int(sample[col].median())
                    user_input[col] = st.number_input(col, value=default, min_value=minv, max_value=maxv)
                else:
                    minv = float(sample[col].min())
                    maxv = float(sample[col].max())
                    default = float(sample[col].median())
                    user_input[col] = st.number_input(col, value=default, min_value=minv, max_value=maxv)

        submitted = st.form_submit_button('Predict')

    if submitted:
        X_in = preprocess_input(user_input, prep)
        pred_proba = model.predict_proba(X_in)[0][1]
        pred_class = model.predict(X_in)[0]

        st.write('### Prediction Result')
        churn_text = 'Yes' if pred_class == 1 else 'No'
        st.write(f'**Churn:** {churn_text}')
        st.write(f'**Churn probability:** {pred_proba:.2f}')

        if st.checkbox('Show debug info'):
            st.write('**Raw input**', user_input)

    st.markdown('---')
    st.markdown(
        '**Notes:** This app replicates preprocessing used in the notebook: '
        'tenure is binned into tenure_bin, categorical variables are one-hot encoded '
        'with drop_first=True, and features are scaled with StandardScaler.'
    )


if __name__ == '__main__':
    main()