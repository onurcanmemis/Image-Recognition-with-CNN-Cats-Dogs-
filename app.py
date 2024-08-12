import streamlit as st
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# Load your trained CNN model
cnn_model = load_model('cnn_model.h5')

# Setting the page configuration with a wide layout and a title
st.set_page_config(page_title="Cat and Dog Image Predictor", page_icon="🐱🐶", layout="wide")

# Adding a wallpaper at the top of the page
wallpaper_path = 'cats_and_dogs.jpg'
wallpaper = Image.open(wallpaper_path)
new_width = wallpaper.width // 2
new_height = wallpaper.height // 4
# Check Pillow version compatibility
try:
    wallpaper = wallpaper.resize((new_width, new_height), Image.Resampling.LANCZOS)
except AttributeError:
    wallpaper = wallpaper.resize((new_width, new_height), Image.LANCZOS)

# Create columns with specified width proportions
left_col, center_col, right_col = st.columns([0.2, 0.6, 0.2])

# Use the center column to display the image
with center_col:
    st.image(wallpaper)  # Display the resized wallpaper

left1_col, center1_col, right1_col = st.columns([0.2, 0.7, 0.15])
with center1_col:
    st.title('Cat and Dog Image Predictor')
    st.write("""
    This simple application uses a ***Convolutional Neural Network (CNN)*** to distinguish between images of cats and dogs.

    **To use this app:**

    1. Click on the **Browse files** button below to upload an image of a cat or a dog.
    2. After the image has been uploaded, click the **Guess** button to the right.
    3. The model will analyze the image and tell you whether it's a cat or a dog.

    Please only upload images in JPG, PNG, or JPEG format.
    """)

    # Create columns for the uploader and the button
    col1, space_col, col2 = st.columns([0.8, 0.2, 1]) 

    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
    with space_col:
        st.write("")
    with col2:
        # Button to guess the image
        if st.button('Guess') and uploaded_file is not None:
            my_bar = st.progress(0)

            for percent_complete in range(0, 101, 10):
                time.sleep(0.1)  # simulate a delay
                my_bar.progress(percent_complete)
            # Convert the file to an image
            test_image = image.load_img(uploaded_file, target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            # Make prediction
            result = cnn_model.predict(test_image/255.0)
            
            # Interpret the results
            if result[0][0] > 0.5:
                prediction = 'DOG'
            else:
                prediction = 'CAT'
            # Ensure the progress bar reaches 100%
            my_bar.progress(100)
            # Display the result as a flashy heading
            st.markdown(f"<h1 style='text-align: left; color: white;'>{prediction}</h1>", unsafe_allow_html=True)      
            # Display the uploaded image
            st.image(uploaded_file, width=350)
