import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def display_grade_descriptions():
    grade_descriptions = {
        "A+": "Outstanding",
        "A": "Excellent",
        "A-": "Very Good",
        "B+": "Good",
        "B": "Above Average",
        "B-": "Average",
        "C+": "Below Average",
        "C": "Poor",
        "C-": "Very Poor",
        "D+": "Extremely Poor",
        "D": "Just Pass",
        "F": "Fail",
    }
    st.markdown(
        """
        <div style="background-color: #000015; padding: 10px; border-radius: 10px;">
        <p style="color: white;">Grade Descriptions:</p>
        <ul>
        """
        + "".join(
            f"<li style='color: white;'>{grade}: {description}</li>"
            for grade, description in grade_descriptions.items()
        )
        + """
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_file_upload():
    st.write("## Upload your file here")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xlsx"):
            st.write("### Data Preview")
            st.dataframe(df.head(10))

            # accessing the columns of the data
            st.write(df.columns)

            # identify the numeric columns and non-numeric columns
            numeric_columns = df.select_dtypes(include=["number"]).columns

            # handling missing values
            imputer = SimpleImputer()
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

            try:
                # selecting the target column
                target = st.selectbox(
                    "Select the target column", options=numeric_columns
                )

                # selecting the feature columns
                features = st.multiselect("Select the feature columns", numeric_columns)

                if not target or not features:
                    st.error("Please select both target and feature columns.")
                    return
            except Exception as e:
                st.error(f"An error occurred while selecting columns: {e}")
                return

            # Split the data into training and testing sets
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # train the model
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)

            # predict the the target column
            y_pred = lin_reg.predict(X_test)

            # Display the predicted values
            st.write("### Predicted Values")
            predicted_df = pd.DataFrame(
                {f"{target}": y_test, f"Predicted {target}": y_pred}
            )
            st.dataframe(predicted_df)
