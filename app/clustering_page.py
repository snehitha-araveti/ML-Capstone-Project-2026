
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = (
    PROJECT_ROOT / "models" / "clustering"
    / "clustering_artifacts.joblib"
)


@st.cache_resource
def load_clustering_artifacts():
    return joblib.load(ARTIFACT_PATH)


def show_clustering_page():
    st.header("👥 Customer Segmentation")
    st.write(
        "Enter customer purchasing behaviour to identify "
        "the segment predicted by our trained K-Means model."
    )

    if not ARTIFACT_PATH.exists():
        st.error(f"Model artifact not found: {ARTIFACT_PATH}")
        return

    try:
        artifacts = load_clustering_artifacts()
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_columns = artifacts["feature_columns"]
        cluster_profiles = artifacts["cluster_profiles"]
    except Exception as error:
        st.error(f"Unable to load the clustering model: {error}")
        return

    st.divider()
    st.subheader("Enter Customer Details")

    with st.form("customer_segmentation_form"):
        frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            value=3,
            step=1
        )

        monetary = st.number_input(
            "Total Spending",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        quantity = st.number_input(
            "Total Quantity Purchased",
            min_value=1,
            value=500,
            step=10
        )

        average_price = st.number_input(
            "Average Price per Item",
            min_value=0.01,
            value=5.0,
            step=1.0
        )

        recency = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            value=90,
            step=1
        )

        submitted = st.form_submit_button(
            "Identify Customer Segment",
            use_container_width=True
        )

    if submitted:
        values = {
            "Frequency": frequency,
            "Monetary": monetary,
            "TotalQuantity": quantity,
            "AveragePrice": average_price,
            "Recency": recency
        }

        input_data = pd.DataFrame(
            [[values[column] for column in feature_columns]],
            columns=feature_columns
        )

        try:
            # Match the transformations used during training
            log_data = np.log1p(input_data)
            scaled_data = scaler.transform(log_data)
            cluster = int(model.predict(scaled_data)[0])

            st.divider()
            st.success(f"Predicted Customer Segment: Cluster {cluster}")

            st.subheader("Typical Cluster Characteristics")

            profile = cluster_profiles[cluster]
            profile_df = pd.DataFrame({
                "Feature": list(profile.keys()),
                "Training Average": list(profile.values())
            })

            st.dataframe(
                profile_df,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Values are averages from the training data. "
                "Cluster numbers identify groups; they are not rankings."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
