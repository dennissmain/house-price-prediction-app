import pandas as pd
import numpy as np 

import streamlit as st 
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
st.set_page_config(page_title = 'House Price Prediction', page_icon = ':cityscape:', layout = 'wide')
import json
import joblib


f = open('categories.json')
categories = json.load(f)





if 'loaded_xgboost_model' not in st.session_state.keys():
    # Load the XGBoost model from the file
    st.session_state['loaded_xgboost_model'] = joblib.load('xgboost_model.pkl')

    # Load the CatBoost model from the file
    st.session_state['loaded_catboost_model'] = joblib.load('catboost_model.pkl')

    booster = st.session_state['loaded_xgboost_model'].get_booster()

    # Get feature names
    st.session_state['feature_names'] = booster.feature_names

    with open("lottie_animation.json", "r") as f:
        st.session_state['lottie_animation'] = json.loads(f.read())


st_lottie(st.session_state['lottie_animation'], speed=1, height=430, key="initial")

st.header('House Price Prediction')

st.subheader('Using Machine Learning to predict house prices in the UK')


categorical_columns = ['Location', 'Bills Included', 'Student Friendly', 'Families Allowed', 'Pets Allowed',
                      'Smokers Allowed', 'DSS/LHA Covers Rent', 'House Code', 'Garden', 'Parking', 'Fireplace',
                      'Furnishing', 'EPC Not Required', 'DSS Income Accepted', 'House Type', 'EPC Rating',
                      'Online Viewings', 'Students Only']

cont_cols = ['Bathrooms', 'Bedrooms', 'Max Tenants', 'Minimum Tenancy']

features_dict = {}

with st.form('my_form'):
        st.write("Enter House features below")
        with st.container():
            for num,col in enumerate( st.columns(2, gap ='large')):
                ind_feats = list(range(len(cont_cols)))
                kk = [ind for ind in ind_feats if ind%2 == num ]
                for ind in kk :
                    features_dict[cont_cols[ind]] = col.number_input(f'**{cont_cols[ind].capitalize()}**',step =1, min_value=1, max_value=10)
            for num,col in enumerate( st.columns(3, gap ='large')):
                ind_feats = list(range(len(categorical_columns)))
                kk = [ind for ind in ind_feats if ind%3 == num ]
                for ind in kk :
                    features_dict[categorical_columns[ind]] = col.selectbox(f"**{categorical_columns[ind]}**", categories[categorical_columns[ind]])
        submitted = st.form_submit_button("Predict")

if submitted: 
    for col in categories.keys():
        for value in categories[col]:
            if features_dict[col] ==value:
                features_dict[f'{col}_{value}'] = 1
            else: 
                features_dict[f'{col}_{value}'] = 0
        features_dict.pop(col)

    features_df = pd.DataFrame(features_dict, index = [0])
    xgb_preds = st.session_state['loaded_xgboost_model'].predict(features_df[st.session_state['feature_names']])

    st.write(f"This house should cost between £{xgb_preds[0]-300:.0f} and £{xgb_preds[0]+300:.0f} ")



