import streamlit as st
import tensorflow as tf
from PIL import Image
from util import set_background, classify

set_background('bgs/bg.png')
def main():
    
    st.title("Grape Leaf Disease Classification")
    st.header('Please upload a Grape Leaf Image')
    file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])
    model = tf.keras.models.load_model('model/GrapeLeafMobileNet.h5')

    classes = ['Grape Black Measles', 'Grape Black rot', 'Grape Healthy', 'Grape Isariopsis Leaf Spot']

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        predicted_class, confidence = classify(image, model, classes)

        st.write("## {}".format(predicted_class))
        st.write("### score: {}".format(int(confidence * 10)/ 10))


if __name__ == "__main__":
    main()