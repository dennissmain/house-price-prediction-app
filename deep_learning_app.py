import pandas as pd
import numpy as np 
import requests

import streamlit as st 
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import os
from utils import merge_images,preprocess_image
from tensorflow import sqrt,square, reduce_mean
from keras.models import load_model

st.set_page_config(page_title = 'House Price Prediction', page_icon = ':cityscape:', layout = 'wide')

st.header('House Price Prediction')

st.subheader('Using Machine Learning to predict house prices in the UK')

col1,col2 = st.columns(2)

def save_uploaded_file(uploaded_file, save_path):
  """
  Saves the uploaded file to the specified path.

  Args:
      uploaded_file: A Streamlit UploadedFile object.
      save_path: The path to save the file.
  """
  with open(save_path, "wb") as f:
      f.write(uploaded_file.getvalue())

with st.form('my_form'):
    st.write("Upload House Images below")
    kitchen  = col1.file_uploader('Upload Kitchen image', ['.jpg'],False)
    bedroom  = col1.file_uploader('Upload bedroom image', ['.jpg'],False)
    bathroom  = col2.file_uploader('Upload bathroom image', ['.jpg'],False)
    living_room  = col2.file_uploader('Upload Living room image', ['.jpg'],False)
    uploaded = st.form_submit_button()


if uploaded: 
    save_uploaded_file(kitchen, 'uploaded images/kitchen.jpg')
    save_uploaded_file(bathroom, 'uploaded images/bathroom.jpg')
    save_uploaded_file(bedroom, 'uploaded images/bedroom.jpg')
    save_uploaded_file(living_room, 'uploaded images/living_room.jpg')
    st.success("Images Uploaded successfully!")

    image_paths = ['bedroom','bathroom','kitchen', 'living_room']

    base_folder = f'uploaded images'
    output_name = f'merged_image.jpg'

    merge_images(base_folder, image_paths, output_name, target_size=(480, 480), border_color="white", border_size=10)

    X = np.array([preprocess_image('merged_image.jpg')])

    # Compile model with RMSE as metric
    def root_mean_squared_error(y_true, y_pred):
        return sqrt(reduce_mean(square(y_true - y_pred)))
    model = load_model('mobilenet_model.h5', custom_objects={'root_mean_squared_error': '' })

    st.write(f"This house should cost around £{model.predict(X)[0][0]:.0f}")