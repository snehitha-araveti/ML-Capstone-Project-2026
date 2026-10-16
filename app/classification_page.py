from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "classification_artifacts.joblib"
)


def show_classification_page():
    st.header("💰 Income Classification")
    st.write(
        "Predict whether a person's annual income is above or at most "
        "$50K using the trained machine learning model."
    )

    if not ARTIFACT_PATH.exists():
        st.error(f"Model artifact not found: {ARTIFACT_PATH}")
        return

    artifacts = joblib.load(ARTIFACT_PATH)

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_columns = artifacts["feature_columns"]
    class_labels = artifacts["class_labels"]

    with st.form("income_classification_form"):
        st.subheader("Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=17, max_value=100, value=30)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox(
                "Race",
                [
                    "White",
                    "Black",
                    "Asian-Pac-Islander",
                    "Amer-Indian-Eskimo",
                    "Other",
                ],
            )
            education = st.selectbox(
                "Education",
                [
                    "HS-grad",
                    "Some-college",
                    "Bachelors",
                    "Masters",
                    "Assoc-voc",
                    "11th",
                    "Assoc-acdm",
                    "10th",
                    "7th-8th",
                    "Prof-school",
                    "9th",
                    "12th",
                    "Doctorate",
                    "5th-6th",
                    "1st-4th",
                    "Preschool",
                ],
            )
            education_num = st.number_input(
                "Education level (numeric)",
                min_value=1,
                max_value=16,
                value=10,
            )
            marital_status = st.selectbox(
                "Marital Status",
                [
                    "Never-married",
                    "Married-civ-spouse",
                    "Divorced",
                    "Separated",
                    "Widowed",
                    "Married-spouse-absent",
                    "Married-AF-spouse",
                ],
            )
            relationship = st.selectbox(
                "Relationship",
                [
                    "Not-in-family",
                    "Husband",
                    "Wife",
                    "Own-child",
                    "Unmarried",
                    "Other-relative",
                ],
            )

        with col2:
            workclass = st.selectbox(
                "Workclass",
                [
                    "Private",
                    "Self-emp-not-inc",
                    "Self-emp-inc",
                    "Federal-gov",
                    "Local-gov",
                    "State-gov",
                    "Without-pay",
                    "Never-worked",
                    "?",
                ],
            )
            occupation = st.selectbox(
                "Occupation",
                [
                    "Prof-specialty",
                    "Craft-repair",
                    "Exec-managerial",
                    "Adm-clerical",
                    "Sales",
                    "Other-service",
                    "Machine-op-inspct",
                    "Transport-moving",
                    "Handlers-cleaners",
                    "Farming-fishing",
                    "Tech-support",
                    "Protective-serv",
                    "Priv-house-serv",
                    "Armed-Forces",
                    "?",
                ],
            )
            native_country = st.selectbox(
                "Native Country",
                [
                    "United-States",
                    "Mexico",
                    "Philippines",
                    "Germany",
                    "Canada",
                    "India",
                    "Puerto-Rico",
                    "El-Salvador",
                    "Cuba",
                    "England",
                    "Jamaica",
                    "South",
                    "China",
                    "Italy",
                    "Dominican-Republic",
                    "Vietnam",
                    "Guatemala",
                    "Japan",
                    "Poland",
                    "Columbia",
                    "Taiwan",
                    "Haiti",
                    "Iran",
                    "Portugal",
                    "Nicaragua",
                    "Peru",
                    "Greece",
                    "France",
                    "Ecuador",
                    "Ireland",
                    "Hong",
                    "Cambodia",
                    "Trinadad&Tobago",
                    "Laos",
                    "Thailand",
                    "Yugoslavia",
                    "Outlying-US(Guam-USVI-etc)",
                    "Honduras",
                    "Hungary",
                    "Scotland",
                    "Holand-Netherlands",
                    "?",
                ],
            )
            fnlwgt = st.number_input(
                "Final sampling weight (fnlwgt)",
                min_value=1,
                value=100000,
            )
            hours_per_week = st.number_input(
                "Hours worked per week",
                min_value=0,
                max_value=100,
                value=40,
            )
            capital_gain = st.number_input(
                "Capital Gain", min_value=0, value=0
            )
            capital_loss = st.number_input(
                "Capital Loss", min_value=0, value=0
            )

        submitted = st.form_submit_button("Predict Income", type="primary")

    if submitted:
        # Derived feature: net capital = capital gain minus capital loss.
        input_values = {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education.num": education_num,
            "marital.status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital.gain": capital_gain,
            "capital.loss": capital_loss,
            "hours.per.week": hours_per_week,
            "native.country": native_country,
            "capital.net": capital_gain - capital_loss,
        }

        input_df = pd.DataFrame([input_values])
        input_df = input_df[feature_columns]

        try:
            transformed_input = preprocessor.transform(input_df)
            prediction = model.predict(transformed_input)[0]

            st.divider()
            st.subheader("Prediction Result")

            if prediction == ">50K":
                st.success("Predicted income: Above $50K per year")
            else:
                st.info("Predicted income: $50K or less per year")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed_input)[0]
                probability_map = dict(zip(model.classes_, probabilities))

                st.write("Model prediction probabilities")
                for label in class_labels:
                    if label in probability_map:
                        st.write(
                            f"{label}: "
                            f"{probability_map[label] * 100:.2f}%"
                        )

            st.caption(
                "This is a machine learning prediction, not a guarantee "
                "of a person's actual income."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "Check that the model artifact and input preprocessing "
                "match the classification notebook."
            )
