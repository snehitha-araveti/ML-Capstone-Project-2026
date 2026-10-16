
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = (
    PROJECT_ROOT / "models" / "regression"
    / "regression_artifacts.joblib"
)


@st.cache_resource
def load_regression_artifacts():
    return joblib.load(ARTIFACT_PATH)


def show_regression_page():
    st.header("🛍️ Purchase Prediction")
    st.write(
        "Estimate a customer's purchase amount using the trained "
        "Random Forest regression model."
    )

    if not ARTIFACT_PATH.exists():
        st.error(f"Model artifact not found: {ARTIFACT_PATH}")
        return

    try:
        artifacts = load_regression_artifacts()
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_columns = artifacts["feature_columns"]
        age_midpoints = artifacts["age_midpoints"]
    except Exception as error:
        st.error(f"Could not load regression artifacts: {error}")
        return

    st.divider()
    st.subheader("Enter Customer and Product Details")

    with st.form("purchase_prediction_form"):
        gender = st.selectbox("Gender", ["F", "M"])
        age_group = st.selectbox(
            "Age Group",
            list(age_midpoints.keys()),
            index=1
        )
        occupation = st.selectbox(
            "Occupation Code", list(range(21)), index=4
        )
        marital_status = st.selectbox("Marital Status", [0, 1])
        city_category = st.selectbox("City Category", ["A", "B", "C"])
        city_years = st.selectbox(
            "Years in Current City",
            ["0", "1", "2", "3", "4+"]
        )

        st.markdown("**Product Details**")
        product_category_1 = st.number_input(
            "Product Category 1", min_value=1, max_value=20, value=5
        )
        product_category_2 = st.number_input(
            "Product Category 2 (0 if unavailable)",
            min_value=0, max_value=20, value=0
        )
        product_category_3 = st.number_input(
            "Product Category 3 (0 if unavailable)",
            min_value=0, max_value=20, value=0
        )

        submitted = st.form_submit_button(
            "Predict Purchase Amount",
            use_container_width=True
        )

    if submitted:
        # Build the numeric and one-hot encoded features.
        # The feature order comes from the saved model artifact.
        values = {
            "Occupation": occupation,
            "Marital_Status": marital_status,
            "Product_Category_1": product_category_1,
            "Product_Category_2": (
                product_category_2 if product_category_2 != 0 else 0
            ),
            "Product_Category_3": (
                product_category_3 if product_category_3 != 0 else 0
            ),
            "Age_Midpoint": age_midpoints[age_group],
            "Gender_M": 1 if gender == "M" else 0,
            "Age_18-25": int(age_group == "18-25"),
            "Age_26-35": int(age_group == "26-35"),
            "Age_36-45": int(age_group == "36-45"),
            "Age_46-50": int(age_group == "46-50"),
            "Age_51-55": int(age_group == "51-55"),
            "Age_55+": int(age_group == "55+"),
            "City_Category_B": int(city_category == "B"),
            "City_Category_C": int(city_category == "C"),
            "Stay_In_Current_City_Years_1": int(city_years == "1"),
            "Stay_In_Current_City_Years_2": int(city_years == "2"),
            "Stay_In_Current_City_Years_3": int(city_years == "3"),
            "Stay_In_Current_City_Years_4+": int(city_years == "4+"),
        }

        input_data = pd.DataFrame(
            [[values[column] for column in feature_columns]],
            columns=feature_columns
        )

        try:
            scaled_data = scaler.transform(input_data)
            prediction = float(model.predict(scaled_data)[0])

            st.divider()
            st.success(f"Predicted Purchase Amount: {prediction:,.2f}")
            st.caption(
                "This is a model estimate, not a guaranteed purchase value."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
