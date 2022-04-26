import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.resnet import ResNet50, preprocess_input as preprocess_RN
from keras.models import load_model, model_from_json
from keras.preprocessing.image import load_img, save_img, img_to_array
#import joblib
#import pickle
from PIL import Image, ImageOps
import streamlit as st
import json

st.title('COVID-19 DETECTION USING LUNG X-RAY')
st.subheader('If you are experiencing any of the following symptoms listed below, please visit your general physician immediately.')
st.image('COVID19-symptoms-thumb.png')
st.write('Please upload your PA Lung X-ray:')

model = model_from_json(open('model_arch.json').read())

file = st.file_uploader('Please upload the image file below:')


def import_and_predict(img, model):
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    pre_img = img
    size = (256,256)
    pre_img = ImageOps.fit(pre_img, size=size, method=Image.ANTIALIAS)
    img_arr = img_to_array(pre_img)
    img_arr = np.expand_dims(img_arr, axis=0)
    data[0] = img_arr
    rslt = model.predict(data)

    if rslt[0][0] < 0.6:  # comparison
        st.error('''The probability of the patient having COVID-19 is **{}%**.
        Please visit your GP immediately.'''.format((1 - rslt[0][0])*100))
    else:
        st.success('''The probability of the patient NOT having COVID-19 is **{}%**.
        But if you are experiencing any of the above symptoms, please consult a doctor.'''.format(rslt[0][0]*100))



if file is None:
    st.text("Please upload an Image file")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    img = img.resize((256,256))
    model.load_weights('my_model_weights.hdf5')
    import_and_predict(img, model)







