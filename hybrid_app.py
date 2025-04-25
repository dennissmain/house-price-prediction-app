# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie, st_lottie_spinner
import json
import joblib
from utils import merge_images, preprocess_image
from tensorflow import sqrt, square, reduce_mean
from keras.models import load_model
import os
import streamlit as st

st.set_page_config(page_title = 'House Price Prediction', page_icon = ':cityscape:', layout = 'wide')

# Load categorical feature categories from JSON file
f = open('categories.json')
categories = json.load(f)

# Function to save an uploaded file to a specified path
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

# Custom loss function for root mean squared error
def root_mean_squared_error(y_true, y_pred):
    return sqrt(reduce_mean(square(y_true - y_pred)))

# Check if models are loaded in session state, if not, load them
if 'loaded_xgboost_model' not in st.session_state.keys():
    st.session_state['loaded_xgboost_model'] = joblib.load('xgboost_model.pkl')
    st.session_state['loaded_catboost_model'] = joblib.load('catboost_model.pkl')
    booster = st.session_state['loaded_xgboost_model'].get_booster()
    st.session_state['feature_names'] = booster.feature_names
    with open("lottie_animation.json", "r") as f:
        st.session_state['lottie_animation'] = json.loads(f.read())
    with open("loading_animation.json", "r") as f:
        st.session_state['loading_animation'] = json.loads(f.read())

# Display initial animation
lottie_anim = st.session_state.get('lottie_animation')
if lottie_anim:
    st_lottie(lottie_anim, speed=1, height=430, key="initial")
else:
    st.warning("⚠️ Could not load the Lottie animation.")

# Display headers and information
st.header('House Price Prediction')
st.subheader('Using Machine Learning to predict house prices in the UK')

# Define categorical and continuous columns
categorical_columns = ['Location', 'Bills Included', 'Student Friendly', 'Families Allowed', 'Pets Allowed',
                      'Smokers Allowed', 'DSS/LHA Covers Rent', 'House Code', 'Garden', 'Parking', 'Fireplace',
                      'Furnishing', 'EPC Not Required', 'DSS Income Accepted', 'House Type', 'EPC Rating',
                      'Online Viewings', 'Students Only']
cont_cols = ['Bathrooms', 'Bedrooms', 'Max Tenants', 'Minimum Tenancy']

features_dict = {}

# Create a form for user input
with st.form('my_form'):
    st.write("Enter House features below")
    with st.container():
        # Display input fields for continuous columns
        for num, col in enumerate(st.columns(2, gap='large')):
            ind_feats = list(range(len(cont_cols)))
            kk = [ind for ind in ind_feats if ind % 2 == num]
            for ind in kk:
                features_dict[cont_cols[ind]] = col.number_input(f'**{cont_cols[ind].capitalize()}**',
                                                                 step=1, min_value=1, max_value=12)
        # Display input fields for categorical columns
        for num, col in enumerate(st.columns(3, gap='large')):
            ind_feats = list(range(len(categorical_columns)))
            kk = [ind for ind in ind_feats if ind % 3 == num]
            for ind in kk:
                features_dict[categorical_columns[ind]] = col.selectbox(f"**{categorical_columns[ind]}**",
                                                                        categories[categorical_columns[ind]])
    st.write("**Upload House Images below**")
    col1, col2 = st.columns(2)
    kitchen = col1.file_uploader('Upload Kitchen image', ['.jpg'], False)
    bedroom = col1.file_uploader('Upload bedroom image', ['.jpg'], False)
    bathroom = col2.file_uploader('Upload bathroom image', ['.jpg'], False)
    living_room = col2.file_uploader('Upload Living room image', ['.jpg'], False)

    # When form is submitted
    submitted = st.form_submit_button("Predict")

if submitted:
    # Display loading animation
    with st_lottie_spinner(st.session_state['loading_animation'], height=300, width=300):
        # Encode categorical features
        for col in categories.keys():
            for value in categories[col]:
                if features_dict[col] == value:
                    features_dict[f'{col}_{value}'] = 1
                else:
                    features_dict[f'{col}_{value}'] = 0
            features_dict.pop(col)

        # Create DataFrame with features
        features_df = pd.DataFrame(features_dict, index=[0])

        # Predict using XGBoost model
        xgb_preds = st.session_state['loaded_xgboost_model'].predict(features_df[st.session_state['feature_names']])[0]

        # Save uploaded images
        save_uploaded_file(kitchen, 'uploaded images/kitchen.jpg')
        save_uploaded_file(bathroom, 'uploaded images/bathroom.jpg')
        save_uploaded_file(bedroom, 'uploaded images/bedroom.jpg')
        save_uploaded_file(living_room, 'uploaded images/living_room.jpg')
        st.success("Images Uploaded successfully!")

        # Merge uploaded images
        image_paths = ['bedroom', 'bathroom', 'kitchen', 'living_room']
        base_folder = f'uploaded images'
        output_name = f'merged_image.jpg'
        merge_images(base_folder, image_paths, output_name, target_size=(480, 480), border_color="white",
                     border_size=10)

        # Preprocess merged image and make prediction using CNN model
        X = np.array([preprocess_image('merged_image.jpg')])
        model = load_model('mobilenet_model.h5', custom_objects={'root_mean_squared_error': ''})
        cnn_model_pred = model.predict(X)[0][0]

        # Combine predictions using ensemble
        ensemble_weight = 0.9
        weighted_ensemble = xgb_preds * ensemble_weight + cnn_model_pred * (1 - ensemble_weight)
        st.write(f"This house should cost around £{weighted_ensemble:.0f}")