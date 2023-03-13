import shap
import streamlit as st
import joblib

import pandas as pd
from matplotlib import pyplot as plt

model_file = open("model/best_model.pkl", "rb")
pipe = joblib.load(model_file)
model = pipe.steps[-1][-1]
preprocessing = pipe.steps[0][-1]
explainer = shap.TreeExplainer(model)


def main():
    st.image('images/img.png')
    st.markdown("<h2 style='text-align: center;'>GHG Emission ML Prediction Demo</h2>", unsafe_allow_html=True)

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    # options = ["Prediction", "Information"]
    # selection = st.sidebar.selectbox("Choose Option", options)

    # # Building out the "Information" page
    # if selection == "Information":
    #     st.info("General Information")
    #     # You can read a markdown file from supporting resources folder
    #     st.markdown("Some information here")
    #
    #     st.subheader("Raw Twitter data and label")
    # if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
    #     st.write([['sentiment', 'message']])  # will write the df to the page

    # st.info("Enter your company's data to predict yours GHG emissions")

    import numpy as np

    choice = np.array([1, 2, 3, 5])
    full_choice = np.concatenate([choice * 10 ** i for i in range(0, 9)]).astype(int)
    empl_full_choice = full_choice[(full_choice >= 1) & (full_choice <= 20_000)].astype(int)
    income_full_choice = full_choice[(full_choice >= 1e4) & (full_choice < 1e9)].astype(int)
    energy_full_choice = full_choice[(full_choice >= 5000) & (full_choice < 1e5)].astype(int)

    col1, col2, col3 = st.columns(3)

    num_employees = col1.select_slider('Number of Employees', options=empl_full_choice,
                                       format_func=lambda x: '{:,}'.format(x))
    income = col2.select_slider('Income ($)', options=income_full_choice,
                                format_func=lambda x: '${:,}'.format(x))
    energy_usage = col3.select_slider('Energy Usage (kWh)', options=energy_full_choice,
                                      format_func=lambda x: '{:,}kWh'.format(x))
    col4, col5 = st.columns((1, 5))
    if col4.button("Predict"):
        input_data = pd.DataFrame([{
            'num_employees': num_employees,
            'income': income,
            'energy_usage': energy_usage
        }])

        prediction = pipe.predict(input_data)[0]
        col5.success(
            f"{prediction.round(-2).astype(int)} tCO2e (tons of CO2 equivalent)"
        )
        with st.expander(label='Feature Importance'):
            show_values = {'num_employees': '{:,}'.format(input_data['num_employees'][0]),
                           'income': '${:,}'.format(input_data['income'][0]),
                           'energy_usage': '{:,}kWh'.format(input_data['energy_usage'][0])}
            show_values = pd.DataFrame([show_values])
            input_data_transformed = preprocessing.transform(input_data)
            input_shap_values = explainer.shap_values(input_data_transformed)
            feature_names = input_data.keys()
            expected_value = explainer.expected_value
            input_shap_values = input_shap_values[0]
            shap.force_plot(expected_value, input_shap_values, show_values, feature_names=feature_names,
                            matplotlib=True)
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            plt.tight_layout()
            st.pyplot(fig)


if __name__ == '__main__':
    main()
