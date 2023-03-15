import shap
import streamlit as st
import joblib
import plotly.graph_objects as go

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
    empl_full_choice = full_choice[(full_choice >= 1) & (full_choice <= 40_000)].astype(int)
    buildings_full_choice = full_choice[(full_choice >= 0) & (full_choice < 1000)].astype(int)
    vehicles_full_choice = full_choice[(full_choice >= 0) & (full_choice < 3000)].astype(int)

    col1, col2, col3 = st.columns(3)

    num_employees = col1.select_slider('Employees', options=empl_full_choice, value=3000,
                                       format_func=lambda x: '{:,}'.format(x))
    buildings = col2.select_slider('buildings', options=buildings_full_choice, value=20,
                                   format_func=lambda x: '{:,}'.format(x))
    vehicles = col3.select_slider('vehicles', options=vehicles_full_choice, value=300,
                                  format_func=lambda x: '{:,}'.format(x))
    col4, col5 = st.columns((1, 5))
    if col4.button("Predict"):
        input_data = pd.DataFrame([{
            'num_employees': num_employees,
            'buildings': buildings,
            'vehicles': vehicles
        }])

        prediction = pipe.predict(input_data)[0]
        col5.success(
            f"Emission: {prediction.round(-2).astype(int)} tCO2e", icon="🍀"
        )
        show_values = {'num_employees': '{:,}'.format(input_data['num_employees'][0]),
                       'buildings': '{:,}'.format(input_data['buildings'][0]),
                       'vehicles': '{:,}'.format(input_data['vehicles'][0])}
        show_values = pd.DataFrame([show_values])
        input_data_transformed = preprocessing.transform(input_data)
        input_shap_values = explainer.shap_values(input_data_transformed)
        feature_names = input_data.keys()
        input_shap_values = input_shap_values[0]
        expected_value = explainer.expected_value[0].astype(int)
        input_shap_values = input_shap_values.astype(int)
        text_values = [f"{feature_name}({value})" for feature_name, value in zip(feature_names, show_values.values[0])]
        with st.expander('Feature Importance (waterfall)'):
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=['absolute', "relative", "relative", "relative", "total"],
                x=["sector average", *feature_names, f"total emission"],
                textposition="outside",
                text=[expected_value, *['{0:+d}'.format(i) for i in input_shap_values], expected_value + sum(input_shap_values)],
                y=[expected_value, *input_shap_values, 123],
                decreasing={"marker": {"color": "#1e88e5"}},
                increasing={"marker": {"color": "#ff0d57"}},
                totals={"marker": {"color": '#C0C0C0', }},
                connector={"mode": "spanning", "line": {"width": 2, "color": "rgb(0, 0, 0)", "dash": "dot"}},
            ))

            fig.update_layout(
                #     title="Profit and loss statement 2018",
                #     # showlegend = True

                autosize=False,
                width=650,
                height=400,
                margin=dict(l=0, r=0, b=0, t=0, pad=2),
            )
            fig.update_yaxes(range=[0, 1.1 * max(expected_value + input_shap_values)])
            # fig.set_size_inches(10, 5)
            plt.tight_layout()
            st.plotly_chart(fig, config={'displayModeBar': False, "showTips": False})
        with st.expander(label='Feature Importance (force plot)'):
            shap.force_plot(expected_value, input_shap_values, show_values, feature_names=feature_names, matplotlib=True)
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            plt.tight_layout()
            st.pyplot(fig)


if __name__ == '__main__':
    main()
