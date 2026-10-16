
import streamlit as st

from clustering_page import show_clustering_page
from regression_page import show_regression_page
from classification_page import show_classification_page


st.set_page_config(
    page_title="ML Capstone Studio",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 ML Capstone Studio")
st.write(
    "Explore predictions and customer segments using "
    "machine learning models developed for our capstone project."
)

st.divider()

track = st.selectbox(
    "Choose a Machine Learning Track",
    [
        "Customer Segmentation",
        "Purchase Prediction",
        "Income Classification"
    ]
)

st.divider()

if track == "Customer Segmentation":
    show_clustering_page()

elif track == "Purchase Prediction":
    show_regression_page()

elif track == "Income Classification":
    show_classification_page()

st.divider()
st.caption("Built with Python · Streamlit · Scikit-learn")
