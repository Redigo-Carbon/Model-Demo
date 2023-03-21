# Predicting Companies CO2 Emissions using Machine Learning

This repository contains a ML model that predicts the CO2 emissions of companies based on a limited number of features. 
The goal of this tool is to help companies assess the environmental impact and understand the most important factors behind it.

# Dataset
The dataset used for this project contains information about the number of employees, buildings, and vehicles of companies,
as well as their CO2 emissions. This artificial demo dataset was modeled using common emission factors of these variables.

# Models
I have tested three different models: 
- DecisionTree, 
- LGBM,
- RandomForest 

To tune the hyper-parameters I've used Bayesian optimization. Cross-validation was used to validate the models. 
To measure the performance of the models I've used R2 metric.

# Feature Importance
Shapley values were used to identify the most important features in predicting CO2 emissions. 
This information can be used to help companies understand the factors that contribute
the most to a company's environmental impact.

# Interactive Web App
An interactive app was created to present the model. 
The app allows users to input the number of employees, 
buildings, and vehicles of a company and get a prediction of its CO2 emissions. 
The app also displays the most important features and their impact on the prediction.

# Results
The models achieved an R2 score of 0.85, indicating a good fit to the data. The most important features in predicting CO2 emissions were found to be the number of vehicles and the number of employees. The interactive app provides an easy-to-use interface for investors to assess the environmental impact of companies.

# Model Results

![model_results](https://github.com/Redigo-Carbon/Model-Demo/blob/main/images/model_results.png?raw=true)

# Feature Importance

![feature_importance_global](https://github.com/Redigo-Carbon/Model-Demo/blob/main/images/feature_importance_global.png?raw=true)

<p float="left">
  <img src="https://github.com/Redigo-Carbon/Model-Demo/blob/main/images/buildings.png?raw=true" width="250" />
  <img src="https://github.com/Redigo-Carbon/Model-Demo/blob/main/images/vehicles.png?raw=true" width="250" />
  <img src="https://github.com/Redigo-Carbon/Model-Demo/blob/main/images/employees.png?raw=true" width="250" />
</p>

# Conclusion
This machine learning model provides a useful tool for companies to assess
the environmental impact of companies. The model is capable of finding most important features in
predicting CO2 emissions. The interactive app provides a
user-friendly interface to use the model.