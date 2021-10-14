import os
import numpy as np
import streamlit as st
from PIL import Image
from utils import *


def main():
    st.title("Cassava leaf disease classification")
    st.subheader("AI Camp - Demo streamlit")
    st.text("Project Author: thekael99 + huynhcuong98")
    model_list = ["EfficientNetB4", "MobilenetV3", "ResneXt50", "Ensemble"]
    choice = st.sidebar.selectbox("Select model", model_list)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    label_names = load_label_names()

    if choice == "EfficientNetB4":
        with st.spinner(text='Loading EfficientNetB4...'):
            model = B4_model()
        st.success('EfficientNetB4 loaded!')

    elif choice == "MobilenetV3":
        with st.spinner(text='Loading MobilenetV3...'):
            model = Mobilenet()
        st.success('MobilenetV3 loaded!')

    elif choice == "ResneXt50":
        with st.spinner(text='Loading ResneXt50...'):
            model = resneXt_model('models/resneXt/weights/cnn_res2.pt')
        st.success('ResneXt50 loaded!')

    elif choice == "Ensemble":
        with st.spinner(text='Loading Ensemble...'):
            model = Ensemble()
        st.success('Ensemble loaded!')

    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image_file is not None:
        with st.spinner(text='Loading image...'):
            img = Image.open(image_file)
            st.text("Your image:")
            st.image(img.resize((600, 400)))
            img = np.array(img)
        if st.button("Run!"):
            with st.spinner(text='In progress...'):
                res = model.predict(img)[0]
                if len(res) == 5:
                    res = np.append(res, 0)
                img_label = assign_label(res, label_names)
                print(img.shape)
            st.success('Done!')
            st.text("Most likely: " + img_label + "\n")
            print(label_names)
            print(res)
            for i in range(len(label_names)):
                st.text("{:36s} {:7.2f}%".format(label_names[i], res[i] * 100))


if __name__ == "__main__":
    main()
