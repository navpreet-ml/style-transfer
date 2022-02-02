import streamlit as st
import os
import tensorflow_hub as hub
from utils import load_img, transform_img, tensor_to_image, imshow
import tensorflow as tf
import numpy as np
from PIL import Image

# For supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.title("Neural Art Style Transfer")

#display demo image
st.image('demo_image.png')


@st.cache
def load_model():
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return hub_model


model_load_state = st.text('Loading Model...')
model = load_model()
# Notify the reader that the data was successfully loaded.
model_load_state.text('Loading Model...done!')

content_image, style_image = st.columns(2)

with content_image:
    st.write('## Content Image...')
    chosen_content = st.radio('How are you going to upload the content image?  ', ("Upload", "URL"))
    if chosen_content == 'Upload':
        st.write(f"You have chosen to upload the image!")
        content_image_file = st.file_uploader(
            "Pick a Content image (accepted formats: PNG, JPG, JPEG)", type=("png", "jpg", "jpeg"))
        try:
            content_image_file = content_image_file.read()
            content_image_file = transform_img(content_image_file)
        except:
            pass
    elif chosen_content == 'URL':
        st.write(f"You have chosen to provide a URL!")
        url = st.text_input('Enter the URL for the content image.')
        try:
            content_path = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'content.jpg'), url)
        except:
            pass
        try:
            content_image_file = load_img(content_path)

        except:
            pass
    try:
        st.write('Content Image')
        st.image(imshow(content_image_file))
    except:
        pass

with style_image:
    st.write('## Style Image...')
    chosen_style = st.radio('How are you going to upload the style image?', ("Upload", "URL"))
    if chosen_style == 'Upload':
        st.write(f"You have chosen to upload the image!")
        style_image_file = st.file_uploader(
            "Pick a Style image (accepted formats: PNG, JPG, JPEG)", type=("png", "jpg", "jpeg"))
        try:
            style_image_file = style_image_file.read()
            style_image_file = transform_img(style_image_file)
        except:
            pass
    elif chosen_style == 'URL':
        st.write(f"You have chosen to provide a URL!")
        url = st.text_input('Enter the URL for the style image.')
        try:
            style_path = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'style.jpg'), url)
        except:
            pass
        try:
            style_image_file = load_img(style_path)

        except:
            pass
    try:
        st.write('Style Image')
        st.image(imshow(style_image_file))
    except:
        pass

predict = st.button('Start Neural Style Transfer...')

if predict:
    try:
        stylized_image = model(tf.constant(content_image_file), tf.constant(style_image_file))[0]
        final_image = tensor_to_image(stylized_image)
    except:
        stylized_image = model(tf.constant(tf.convert_to_tensor(content_image_file[:, :, :, :3])),
                               tf.constant(tf.convert_to_tensor(style_image_file[:, :, :, :3])))[0]

        final_image = tensor_to_image(stylized_image)

    st.write('Output Image')
    st.image(final_image)

    try:
        # Delete style.jpg and content.jpg
        os.remove("style.jpg")
        os.remove("content.jpg")
    except:
        pass
