import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
import joblib
from main import display_grade_descriptions, handle_file_upload
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Page lay?out
st.set_page_config(page_title="Online Grade Predictor", page_icon="🎓", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Settings",
        options=["Home", "Upload"],
        icons=["house", "upload"],
        default_index=0,
    )

    if selected == "Home":
        display_grade_descriptions()
    elif selected == "Upload":
        handle_file_upload()


# Main content
st.subheader("Online Grade Predictor 📊", anchor="top", divider="grey")


# Input fields for user data
col1, col2, col3 = st.columns(3)
with col1:
    study_hours = st.number_input("Study Hours", min_value=0, max_value=24, value=0)
    attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=0)
    assignments = st.number_input(
        "Assignments (%)", min_value=0, max_value=100, value=0
    )

with col2:
    midterm = st.number_input("Midterm Grade", min_value=0, max_value=100, value=0)
    projects = st.number_input("Group Project (%)", min_value=0, max_value=100, value=0)
    quizzes = st.number_input("Quiz Avg (%)", min_value=0, max_value=100, value=0)
with col3:
    sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=0)
    extra_curricular = st.number_input(
        "Extracurricular (%)", min_value=0, max_value=100, value=0
    )
    participation = st.number_input(
        "Participation (%)", min_value=0, max_value=100, value=0
    )

# Predict button
predict_button = st.button("Predict", key="predict_button")

if predict_button:
    # Check if any input is zero
    if any(
        value == 0
        for value in [
            study_hours,
            attendance,
            assignments,
            midterm,
            projects,
            quizzes,
            sleep_hours,
            extra_curricular,
            participation,
        ]
    ):
        st.warning("Please enter all values first to get your GPA. ")
        predicted_grade_value = 0
    else:
        # Load the model
        model = joblib.load("grade_predictor.pkl")

        # Create a DataFrame from the user input
        user_data = pd.DataFrame(
            {
                "Study Hours": [study_hours],
                "Attendance (%)": [attendance],
                "Assignments (%)": [assignments],
                "Midterm Grade": [midterm],
                "Group Project (%)": [projects],
                "Quiz Avg (%)": [quizzes],
                "Sleep Hours": [sleep_hours],
                "Extracurricular (%)": [extra_curricular],
                "Participation (%)": [participation],
            }
        )

        # Make a prediction
        prediction = model.predict(user_data)
        predicted_grade_value = prediction[0]

        # Display the prediction
        st.write(
            f"### Your GPA Grade is: **{predicted_grade_value:.2f}**",
            unsafe_allow_html=True,
        )

        # Combine all features into a single DataFrame for plotting
        plot_data = user_data.melt(var_name="Feature", value_name="Value")
        plot_data["Predicted Grade"] = predicted_grade_value


# Reset button
if st.button("Reset", key="reset_button"):
    study_hours = 0
    attendance = 0
    assignments = 0
    midterm = 0
    projects = 0
    quizzes = 0
    sleep_hours = 0
    extra_curricular = 0
    participation = 0


# Display the prediction explanation
st.write(
    """
    ### How is my GPA calculated?
    - **Study Hours**: The more time you spend studying, the higher your grade is likely to be.
    - **Assignments**: Completing assignments on time is important for your grade.
    - **Attendance**: Regular attendance is crucial for your grade.
    - **Quizzes**: Doing well on quizzes can boost your grade.
    - **Midterm**: Scoring well on the midterm can improve your grade.
    - **Projects**: Doing well on group projects can positively impact your grade.
    - **Participation**: Active participation in class can help improve your grade.
    - **Extracurricular**: Involvement in extracurricular activities can be beneficial for your grade.
    """
)


# Custom CSS for the predict button
st.markdown(
    """
    <style>
    .stButton button {
        background-color: white;
        color: black;
        float: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for the reset buttons
st.markdown(
    """
    <style>
    .stButton button {
        background-color: white;
        color: black;
        float: left;
        margin-right: 10px;
    }
    .stButton button[key="reset_button"] {
        background-color: grey;
        color: white;
        float: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if predict_button and predicted_grade_value != 0:
    # Altair bar-chart and line graph

    # Bar chart for all features
    bar_chart = (
        alt.Chart(plot_data)
        .mark_bar()
        .encode(x=alt.X("Feature", sort=None), y="Value", color="Feature")
        .properties(title="Feature Values")
    )

    # Bar chart for predicted GPA
    gpa_data = pd.DataFrame(
        {"Feature": ["Final GPA"], "Value": [predicted_grade_value]}
    )

    gpa_bar_chart = (
        alt.Chart(gpa_data)
        .mark_bar()
        .encode(x="Feature", y="Value", color=alt.value("orange"))
        .properties(title="Predicted Final GPA")
    )

    # Line graph for feature values vs predicted GPA
    line_data = user_data.copy()
    line_data["Final GPA"] = predicted_grade_value

    line_chart = (
        alt.Chart(
            line_data.melt(
                id_vars=["Final GPA"], var_name="Feature", value_name="Value"
            )
        )
        .mark_line()
        .encode(x="Feature", y="Value", color=alt.value("blue"))
        .properties(title="Feature Values vs Final GPA")
    )

    # Display the charts
    st.altair_chart(bar_chart, use_container_width=True)
    st.altair_chart(gpa_bar_chart, use_container_width=True)
    st.altair_chart(line_chart, use_container_width=True)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #000015;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="footer">
    Made with ❤️ by [Arbind kumar sah](
    [GitHub](https://github.com/Arbind-sah)]
    </div>
    """,
    unsafe_allow_html=True,
)


# horizontal line under which copy right is written

st.write(
    '<hr style="border: 0.2px solid white; border-radius: 2px; margin: 10px 0;">',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">&copy; 2020 All rights reserved</p>',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">Contact us: info@example.com</p>',
    unsafe_allow_html=True,
)
st.write(
    '<p style="color: white; text-align: center;">Follow us on <a href="https://twitter.com/example" style="color: #1DA1F2;">Twitter</a> and <a href="https://facebook.com/example" style="color: #4267B2;">Facebook</a></p>',
    unsafe_allow_html=True,
)


# train the model

df = pd.read_csv("gpa.csv")

# Convert non-numeric grades to numeric values
grade_mapping = {
    "A+": 4.0,
    "A": 3.9,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "F": 0.0,
}

df["Final Grade"] = df["Final Grade"].map(grade_mapping)

X = df[
    [
        "Study Hours",
        "Attendance (%)",
        "Assignments (%)",
        "Participation (%)",
        "Midterm Grade",
        "Group Project (%)",
        "Quiz Avg (%)",
        "Sleep Hours",
        "Extracurricular (%)",
    ]
]

y = df["Final Grade"]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

trained_pred = lin_reg.predict(X_train)
machine_pred = lin_reg.predict(X_test)

# model evaluation
r2_train = r2_score(y_train, trained_pred)
print(f"R2 Score for training data: {r2_train}")

# save the model
joblib.dump(lin_reg, "grade_predictor.pkl")
