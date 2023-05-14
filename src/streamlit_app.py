import streamlit as st
import torch
import numpy as np
from PIL import Image
from model.dataset import DefaultTransform
from model.mask import get_mask_noise
from setup.logging_utils import normalize_image
from model.model import Generator

tf = DefaultTransform()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

gen = Generator(32, 8)
model_dict = torch.load('../data/model_checkpoint_best.pth')
gen.load_state_dict(model_dict['generator_he_to_p63_state_dict'])
gen = gen.to(device)
gen.eval()


def get_eigen():
    global gen, device

    _eigen = gen.interpretable_conv_2.conv.weight.cpu().detach().numpy().squeeze()
    _eigen /= np.linalg.norm(_eigen, axis=0, keepdims=True)
    _, _eigen_vectors = np.linalg.eig(_eigen.dot(_eigen.T))
    return torch.from_numpy(_eigen_vectors.T).to(device)


def run_model(_img, _eigen_range, _mod_range):
    global gen, device

    # get the eigen vectors
    eigen = get_eigen()

    # prep mask
    mask = get_mask_noise(_img).to(device)

    # get the codes
    img_codes, mask_codes = gen.get_partial_pass(_img, mask)

    # get the new img
    _img = gen.get_modified_rest_pass(_img, img_codes, mask_codes, eigen, _mod_range, _eigen_range)

    return normalize_image(_img)


def prepare_image(path):
    global tf, device

    _img = Image.open(path)
    _img = tf(_img)
    _img = _img.unsqueeze(0)
    return _img.to(device)


def re_run_model():
    img = prepare_image(st.session_state['uploaded_file'])
    img = run_model(img, st.session_state['eigen_range'], st.session_state['mod_range'])
    st.session_state['img'] = img


def main():
    # put the following two sliders next to each other
    default_file = "../data/test/he/3078_4_he.png"
    col1, col2 = st.columns(2)

    with col1:
        eigen_range = st.slider('Weight Layer Range', min_value=0, max_value=15,
                                value=(5, 10), step=1)

    with col2:
        mod_range = st.slider('Modulation Layer Range', min_value=-5.0, max_value=5.0,
                              value=0.0, step=0.1) * 1000

    if 'eigen_range' not in st.session_state:
        st.session_state['eigen_range'] = eigen_range
    if 'mod_range' not in st.session_state:
        st.session_state['mod_range'] = mod_range

    if 'eigen_range' in st.session_state and 'mod_range' in st.session_state:
        if eigen_range != st.session_state['eigen_range'] or mod_range != st.session_state['mod_range']:
            st.session_state['eigen_range'] = eigen_range
            st.session_state['mod_range'] = mod_range
            re_run_model()

    # add a file opener for images, and a button to run the model, next to each other
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is None:
        uploaded_file = default_file

    st.session_state['uploaded_file'] = uploaded_file

    # center between columns and stretch to full width
    with st.container():
        run_button = st.button('Run Model', use_container_width=True)

    # if the button is pressed, run the model, use default image if non supplied
    if run_button:
        img = prepare_image(uploaded_file)

        # run the model
        img = run_model(img, eigen_range, mod_range)
        # display the output
        st.image(img, use_column_width=True)
    else:
        # display the input
        if 'img' in st.session_state:
            st.image(st.session_state['img'], use_column_width=True)
        else:
            st.image(uploaded_file, use_column_width=True)


def test():
    img = prepare_image("../data/test/he/3078_4_he.png")
    _ = run_model(img, (5, 10), 0)


# This file is run by using the command: streamlit run src\streamlit_app.py
