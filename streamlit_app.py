import time

import shap
import streamlit as st
import joblib
import plotly.graph_objects as go
import numpy as np
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

    choice = np.array([1, 2, 3, 5])
    full_choice = np.concatenate([choice * 10 ** i for i in range(0, 9)]).astype(int)
    employees_full_choice = full_choice[(full_choice >= 1) & (full_choice <= 40_000)].astype(int)
    buildings_full_choice = full_choice[(full_choice >= 0) & (full_choice < 1000)].astype(int)
    vehicles_full_choice = full_choice[(full_choice >= 0) & (full_choice < 3000)].astype(int)

    col1, col2, col3 = st.columns(3)

    num_employees = col1.select_slider('Employees', options=employees_full_choice, value=3000,
                                       format_func=lambda x: '{:,}'.format(x))
    buildings = col2.select_slider('buildings', options=buildings_full_choice, value=20,
                                   format_func=lambda x: '{:,}'.format(x))
    vehicles = col3.select_slider('vehicles', options=vehicles_full_choice, value=300,
                                  format_func=lambda x: '{:,}'.format(x))
    col4, col5 = st.columns((1, 5))
    if col4.button("Predict"):
        with st.spinner('ML model calculation...'):
            time.sleep(2)
            input_data = pd.DataFrame([{
                    'num_employees': num_employees,
                    'buildings': buildings,
                    'vehicles': vehicles
                }])

            prediction = pipe.predict(input_data)[0]
            col5.success(
                f"Emission: {prediction.round(-2).astype(int)} tCO2e", icon="🍀"
            )
            show_values = {
                'num_employees': '{:,}'.format(input_data['num_employees'][0]),
                'buildings': '{:,}'.format(input_data['buildings'][0]),
                'vehicles': '{:,}'.format(input_data['vehicles'][0])
            }
            show_values = pd.DataFrame([show_values])
            input_data_transformed = preprocessing.transform(input_data)
            input_shap_values = explainer.shap_values(input_data_transformed)
            feature_names = input_data.keys()
            input_shap_values = input_shap_values[0]
            expected_value = explainer.expected_value[0].astype(int)
            input_shap_values = input_shap_values.astype(int)
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
                    autosize=False,
                    width=650,
                    height=400,
                    margin=dict(l=0, r=0, b=0, t=0, pad=2),
                )
                fig.update_yaxes(range=[0, 1.1 * max(expected_value + input_shap_values)])
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
