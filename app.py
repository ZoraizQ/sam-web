import os
import streamlit as st 
from PIL import Image
import numpy as np
import requests
import sys
import matplotlib.pyplot as plt

pred_url = ""

def predict(img_name):
    img_path = "exp/"+img_name
    with open(img_path, 'wb') as f:
        f.write(bytes_data)

    r = requests.post(pred_url, files={'media': open(img_path, 'rb')})

    pred_path = "exp/p_"+img_name

    if r.status_code == 200:
        with open(pred_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        st.success("Predictions:")

        return Image.open(pred_path)
    else:
        return Image.open(img_path)


# Keep the state of the button press between actions
@st.cache(allow_output_mutation=True)
def get_states():
    return {"pred_url": None, "pressed": None, "pred": None}


st.set_page_config("SAM GAN Web", layout='centered')
st.title("SAM StyleGAN for Age Regression")
curr_state = get_states()  # gets our cached dictionary

if curr_state["pred_url"] == None:
    pred_url = st.text_input("Server URI @ngrok.io:")
    if len(pred_url) > 0:
        if pred_url[-1] == '/': # remove / at end if entered
            pred_url = pred_url[:-1]

        pred_url += "/predict"
        curr_state.update({"pred_url": pred_url})
else:
    pred_url = curr_state["pred_url"]


st.subheader("Upload an image to predict for 11 ages (0-100)")   


uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))



if uploaded_file is not None:

    st.write("uploading...") # upload to local directory of server first
    bytes_data = uploaded_file.getvalue()
    img_path = uploaded_file.name.lower()

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image')
    st.write("")


    press_button = st.button("Predict Now")

    if press_button: # any changes need to be performed in place
        with st.spinner("now time travelling with SAM..."):
            pred = predict(img_path)
        curr_state.update({"pressed": True, "pred": pred})

    if st.button("Clear") and curr_state["pressed"]:
        curr_state.update({"pressed": None, "pred": None})

    if curr_state["pressed"]:
        pred = curr_state["pred"]
        st.image(pred, caption='Result', use_column_width=True) 
        pred = pred.resize((256*12, 256))
        
        predrow = np.array(pred)
        pred_slices = np.array_split(predrow, 12, axis=1)[1:] # ignore first image which is the original one

        age = st.slider('Interpolate through the ages', 0, 100) # interpolation factor of age
        f = age / 100 * 10.0
        li = int(np.floor(f))
        ri = int(np.ceil(f))
        alpha = f-li

        interp_img = alpha*pred_slices[li] + (1.0-alpha) * pred_slices[ri] # linear interpolation
        interp_img *= (1.0/interp_img.max())
        st.image(interp_img, caption='Age: '+str(age))   

        
