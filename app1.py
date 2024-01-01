import streamlit as st
import pandas as pd
import numpy as np
from phe import paillier
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# def convert_df_to_csv_download_link(df):
#     csv = df.to_csv(index=False).encode('utf-8-sig')
#     b64 = base64.b64encode(csv).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" target="_blank">Download predictions.csv</a>'
#     return href

# Initialize global variables for public and private keys
public_key, private_key = None, None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Function to preprocess data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Function to train the RandomForest model
def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    y = df['salary']  # Replace with your target variable
    X = df.drop('salary', axis=1)  # Replace with your features
    X_processed = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    #st.write(f'Test Score: {test_score}')
    return model

# Function to generate keys and save them
def generate_and_save_keys():
    global public_key, private_key
    public_key, private_key = paillier.generate_paillier_keypair()
    keys = {
        'public_key': {'n': str(public_key.n)},  # Convert large number to string
        'private_key': {'p': str(private_key.p), 'q': str(private_key.q)}  # Convert large numbers to string
    }
    with open('public_key.json', 'w') as f:
        json.dump(keys['public_key'], f)
    with open('private_key.json', 'w') as f:
        json.dump(keys['private_key'], f)
    return keys

# Streamlit interface
st.set_page_config(page_title='Secure ML Prediction', layout='wide')
st.title('Secure Machine Learning Prediction with Homomorphic Encryption')

# Sidebar for key generation and model training
with st.sidebar:
    st.header('Setup')
    if st.button('Generate Encryption Keys'):
        keys = generate_and_save_keys()
        st.session_state['public_key'] = keys['public_key']
        st.session_state['private_key'] = keys['private_key']
        st.success('Encryption keys generated and saved.')
        # Display the generated keys
        st.subheader('Public Key:')
        st.code(f"n = {keys['public_key']['n']}")
        st.subheader('Private Key (Hidden for security):')
        st.code(f"p = {'*' * 10}")  # Masking private key
        st.code(f"q = {'*' * 10}")  # Masking private key

    if st.button('Train Model'):
        with st.spinner('Training in progress...'):
            model = train_model('employee_data.csv')  # Ensure the dataset path is correct
            st.session_state['model'] = model
            st.success('Model training completed successfully!')

# Main Page for Data Prediction
st.header('Data Prediction')
st.markdown('Enter the data you want to predict on using the trained model.')

# Assuming there are four features to input
feature_1 = st.number_input('Enter age:', value=60)
feature_2 = st.number_input('Enter healthy_eating:', value=8)
feature_3 = st.number_input('Enter active_lifestyle:', value=1)
feature_4 = st.number_input('Enter Gender:', value=1)

if st.button('Encrypt Data and Predict'):
    if 'public_key' in st.session_state and 'private_key' in st.session_state and 'model' in st.session_state:
        with st.spinner('Processing your data...'):
            try:
                # Convert the stored keys back to their appropriate types
                public_key = paillier.PaillierPublicKey(n=int(st.session_state['public_key']['n']))
                private_key = paillier.PaillierPrivateKey(public_key, 
                                                          int(st.session_state['private_key']['p']), 
                                                          int(st.session_state['private_key']['q']))
                # Prepare the data for prediction
                data = np.array([[feature_1, feature_2, feature_3, feature_4]])
                data_processed = preprocess_data(data)  # Preprocess the data
                # Encrypt and decrypt the processed data
                encrypted_data = [public_key.encrypt(x) for x in data_processed.flatten()]
                decrypted_data = np.array([private_key.decrypt(x) for x in encrypted_data]).reshape(data_processed.shape)
                # Predict
                prediction = st.session_state['model'].predict(decrypted_data)
                st.success('Prediction successful!')
                st.write(f'Predicted Value: {prediction[0]}')
            

            except Exception as e:
                st.error(f'An error occurred: {e}')
    else:
        st.error('Please complete the setup steps first.')
