Streamlit-based web application for analyzing and visualizing forecast data, actual inbound metrics, promotional activities, and model performance. The application provides interactive visualizations and tools for decomposing time-series data, generating summary statistics, and comparing forecast models with existing predictions.

Prerequisites
Ensure you have the following installed:

Python 3.8 or higher

pip install streamlit
pandas
plotly
numpy
statsmodels
matplotlib

##################
Data Files:

combined_for_model2.csv: 

Dataset containing actual inbound metrics (you will change it for future usage of the app).

Merged_Predictions_Data.csv:

Dataset containing combined forecasted values (you will change it for future usage of the app).

RUN THE APPLICATION:

streamlit run dashboard.py

##################