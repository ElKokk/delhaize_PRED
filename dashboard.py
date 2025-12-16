import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# INITIAL PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(page_title="Data Horizon Analytics", page_icon=":bar_chart:", layout="wide")

# --------------------------------------------------------------------------
# THEME & STYLING
# --------------------------------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

toggle_theme = st.button("Toggle Dark/Light Mode")
if toggle_theme:
    st.session_state['theme'] = 'dark' if st.session_state['theme'] == 'light' else 'light'

# We do NOT change the sidebar styling based on theme. Sidebar remains vanilla on black.
if st.session_state['theme'] == 'dark':
    bg_color = '#000000'
    text_color = '#FAF3E0'
    base_css = f"""
    <style>
    body {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    .block-container {{
        background-color: {bg_color} !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6, .stSelectbox > div > div > label,
    .stRadio > div > label, .stSlider label, .st-mb, .st-df, .stTextInput label,
    .stNumberInput label, .st-df th, .st-df td {{
        color: {text_color} !important;
    }}
    </style>
    """
else:
    bg_color = '#FAF3E0'
    text_color = '#000000'
    base_css = f"""
    <style>
    body {{
        background-color: {bg_color} !important;
    }}
    .block-container {{
        background-color: {bg_color} !important;
    }}
    * {{
        color: {text_color} !important;
    }}
    </style>
    """

st.markdown(base_css, unsafe_allow_html=True)

# Sidebar and buttons always vanilla on black, no matter the theme
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] * {
        color: #FAF3E0 !important;
    }
    [data-testid="stSidebar"] *:hover {
        color: #FAF3E0 !important;
    }
    button, button * {
        color: #FAF3E0 !important;
    }
    button:hover, button:hover * {
        color: #FAF3E0 !important;
    }

    .styled-table {
        border:2px solid black !important;
        border-collapse: collapse !important;
        font-size:14px !important;
        width:100% !important;
    }
    .styled-table th, .styled-table td {
        border:1px solid black !important;
        padding:4px !important;
        vertical-align: middle !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Re-apply dropdown menu text color after global rules to ensure vanilla text on black background
st.markdown(
    """
    <style>
    .css-26l3qy-menu, .css-26l3qy-menu * {
        color: #FAF3E0 !important;
        background-color: #000000 !important;
    }
    .css-26l3qy-option:hover {
        color: #FAF3E0 !important;
        background-color: #333333 !important;
    }

    /* Ensure the difference column always red, bold, slightly larger */
    .styled-table td:last-child span {
        color: red !important;
        font-weight: bold !important;
        font-size:110% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        st.error(f"File not found: {image_path}")
        return None

@st.cache_data
def load_data():
    file_path = 'combined_for_model2.csv'
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.tz_localize(None)
    data.rename(columns=lambda x: x.replace('Fc', 'old prediction') if 'Fc' in x else x, inplace=True)
    return data

@st.cache_data
def load_predictions():
    predictions_file_path = 'Merged_Predictions_Data.csv'
    predictions_data = pd.read_csv(predictions_file_path)
    predictions_data['Date'] = pd.to_datetime(predictions_data['Date'], dayfirst=True, errors='coerce')
    if predictions_data['Date'].dtype == 'datetime64[ns, UTC]':
        predictions_data['Date'] = predictions_data['Date'].dt.tz_localize(None)
    return predictions_data

def calculate_metrics(actual, model_pred, existing_fc):
    actual = np.array(actual)
    model_pred = np.array(model_pred)
    existing_fc = np.array(existing_fc)

    mask = actual != 0
    if not np.any(mask):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    model_mae = np.mean(np.abs(actual - model_pred))
    model_rmse = np.sqrt(np.mean((actual - model_pred)**2))
    model_mape = np.mean(np.abs((actual[mask] - model_pred[mask]) / actual[mask]))
    model_accuracy = 1 - model_mape if not np.isnan(model_mape) else np.nan

    existing_mae = np.mean(np.abs(actual - existing_fc))
    existing_rmse = np.sqrt(np.mean((actual - existing_fc)**2))
    existing_mape = np.mean(np.abs((actual[mask] - existing_fc[mask]) / actual[mask]))
    existing_accuracy = 1 - existing_mape if not np.isnan(existing_mape) else np.nan

    return (model_mae, model_rmse, model_mape, model_accuracy, existing_mae, existing_rmse, existing_mape, existing_accuracy)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
data = load_data()
predictions_data = load_predictions()

cols_to_remove = ['HD Total', 'Pick Capacity', 'Collect Total', 'Pre Orders', 'Category', 'Day of Week', 'pre orders old prediction']
all_actual_cols = list(data.columns)
all_prediction_cols = list(predictions_data.columns)

excluded_cols = ['Date'] + cols_to_remove
promo_columns = [c for c in all_actual_cols if 'Promo' in c and c != 'Date']
actual_columns = [c for c in all_actual_cols if c not in excluded_cols + promo_columns and c != 'Date' and not c.startswith("Unnamed")]

prediction_columns = [c for c in all_prediction_cols if (c.startswith("Model_Prediction_") or c.startswith("Existing_Forecast_"))]

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
logo_path = 'output-onlinepngtools.png'
left_image_path = 'elef.png'
right_image_path = 'rachelle.jpg'

encoded_logo = encode_image(logo_path)
encoded_left_image = encode_image(left_image_path)
encoded_right_image = encode_image(right_image_path)

st.markdown("<h1 style='text-align: center;'>Data Horizon Analytics</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <div style="text-align: center;">
            <img src='data:image/png;base64,{encoded_left_image}' alt='Left Image' width='100'>
            <h2 style="text-align: center;">Eleftherios Kokkinis</h2>
        </div>
        <div style="text-align: center;">
            <img src='data:image/png;base64,{encoded_logo}' alt='Logo' width='200'>
        </div>
        <div style="text-align: center;">
            <img src='data:image/png;base64,{encoded_right_image}' alt='Right Image' width='220'>
            <h2 style="text-align: center;">Rachelle Natumanya</h2>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION & FILTERS
# --------------------------------------------------------------------------
st.sidebar.title("Overview")
st.sidebar.markdown("**Use the navigation below to explore:**")
page = st.sidebar.radio(
    "Go to",
    ("Overview", "Forecast Data", "Trend Analysis", "Promo Trends", "Summary Statistics", "Time Series Decomposition", "Metrics")
)

predictions_min_date = pd.to_datetime("2024-08-22")
predictions_max_date = pd.to_datetime("2024-09-11")
actual_data_min_date = data['Date'].min()
actual_data_max_date = data['Date'].max()

st.sidebar.header("Filters")

predictions_slider_range = st.sidebar.slider(
    "Select Date Range for Predictions",
    min_value=predictions_min_date.date(),
    max_value=predictions_max_date.date(),
    value=(predictions_min_date.date(), predictions_max_date.date())
)

predictions_valid = True
if len(predictions_slider_range) != 2 or predictions_slider_range[0] == predictions_slider_range[1]:
    predictions_valid = False

data_slider_range = st.sidebar.slider(
    "Select Date Range for Actuals",
    min_value=actual_data_min_date.date(),
    max_value=actual_data_max_date.date(),
    value=(actual_data_min_date.date(), actual_data_max_date.date())
)

data_valid = True
if len(data_slider_range) != 2 or data_slider_range[0] == data_slider_range[1]:
    data_valid = False

if predictions_valid:
    filtered_predictions = predictions_data[
        (predictions_data['Date'] >= pd.to_datetime(predictions_slider_range[0])) &
        (predictions_data['Date'] <= pd.to_datetime(predictions_slider_range[1]))
        ]
else:
    filtered_predictions = pd.DataFrame()

if data_valid:
    filtered_data = data[
        (data['Date'] >= pd.to_datetime(data_slider_range[0])) &
        (data['Date'] <= pd.to_datetime(data_slider_range[1]))
        ]
else:
    filtered_data = pd.DataFrame()

st.sidebar.header("Column Selections")

selected_columns_predictions = st.sidebar.multiselect(
    "Select Prediction Categories",
    options=prediction_columns
)

selected_columns_data = st.sidebar.multiselect(
    "Select Actual Inbound Metrics",
    options=actual_columns
)

st.sidebar.header("Promo Data Visualization")
selected_promo_columns = st.sidebar.multiselect(
    "Select Promo Columns",
    options=promo_columns
)

# --------------------------------------------------------------------------
# MAIN PAGE CONTENT
# --------------------------------------------------------------------------

if page == "Overview":
    st.subheader("Welcome to Data Horizon Analytics")
    st.markdown(
        """
        **Overview of the Application**

        This application allows you to:

        - **View Forecasts:** Check predicted values for various product categories.
        - **Compare Predictions and Actuals:** Visualize trends in predicted and actual inbound metrics.
        - **Analyze Promo Data:** Understand the influence of promotional activities on inbound quantities.
        - **Explore Summary Statistics:** Get descriptive statistics for selected categories.
        - **Decompose Time Series:** Break down a selected metric into trend, seasonality, and residual components.
        - **View Model Performance Metrics:** Compare the model's performance against existing forecasts.

        **How to Use:**
        1. Use the sidebar to select date ranges, categories, and promo columns.
        2. Navigate through the sections using the sidebar radio buttons.
        3. Toggle the theme at the top of the page for a dark/light mode.
        
        **Additional Tips:**
        - Hover over points in graphs to see details.
        - Filter the data based on your requirements using the sidebar filters.
        - Check summary stats to understand data distribution.
        """
    )

elif page == "Forecast Data":
    st.subheader("Forecasts for the Selected Period")
    if predictions_valid:
        if not filtered_predictions.empty:
            st.dataframe(filtered_predictions, use_container_width=True)
        else:
            st.warning("No data available for the selected predictions date range.")
    else:
        st.warning("Please select a valid predictions date range (at least two different dates).")

elif page == "Trend Analysis":
    st.subheader("Interactive Trend Plot")
    if not predictions_valid:
        st.warning("Please select a valid predictions date range.")
    elif not data_valid:
        st.warning("Please select a valid actual data date range.")
    else:
        if (selected_columns_predictions or selected_columns_data) and not filtered_predictions.empty and not filtered_data.empty:
            combined_data_for_plot = pd.DataFrame({'Date': pd.to_datetime([])})
            if selected_columns_predictions:
                temp_predictions = filtered_predictions[['Date'] + selected_columns_predictions]
                temp_predictions = temp_predictions.melt(id_vars='Date', var_name='Category', value_name='Value')
                temp_predictions['Type'] = 'Prediction'
                combined_data_for_plot = pd.concat([combined_data_for_plot, temp_predictions], ignore_index=True)

            if selected_columns_data:
                temp_actual = filtered_data[['Date'] + selected_columns_data]
                temp_actual = temp_actual.melt(id_vars='Date', var_name='Category', value_name='Value')
                temp_actual['Type'] = 'Actual'
                combined_data_for_plot = pd.concat([combined_data_for_plot, temp_actual], ignore_index=True)

            if not combined_data_for_plot.empty:
                fig = px.line(
                    combined_data_for_plot,
                    x='Date',
                    y='Value',
                    color='Category',
                    line_dash='Type',
                    title="Trends for Selected Categories",
                    labels={'Value': 'Product Quantities', 'Date': 'Date'},
                    hover_data={'Type': True}
                )
                fig.update_layout(legend_title="Category and Type")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Sum of Quantities for Selected Categories")
                sum_data = {}
                if selected_columns_data:
                    for col in selected_columns_data:
                        sum_data[f"Actual: {col}"] = filtered_data[col].sum()
                if selected_columns_predictions:
                    for col in selected_columns_predictions:
                        sum_data[f"Prediction: {col}"] = filtered_predictions[col].sum()

                if sum_data:
                    sum_df = pd.DataFrame(list(sum_data.items()), columns=["Category", "Total Quantity"])
                    fig_bar = px.bar(
                        sum_df,
                        x="Category",
                        y="Total Quantity",
                        color="Category",
                        title="Sum of Quantities for Selected Categories",
                        labels={"Total Quantity": "Total Quantity", "Category": "Category"},
                        text_auto=True
                    )
                    fig_bar.update_layout(
                        xaxis_title="Category",
                        yaxis_title="Total Quantity",
                        legend_title="Category",
                        barmode="group"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No categories selected for sum visualization.")
            else:
                st.info("No data to plot. Please adjust your selections.")
        else:
            st.info("Select categories and valid date ranges for both predictions and actual data.")

elif page == "Promo Trends":
    st.subheader("Promo Data Trends")
    if not data_valid:
        st.warning("Please select a valid actual data date range.")
    else:
        if selected_promo_columns and not filtered_data.empty:
            temp_promo = filtered_data[['Date'] + selected_promo_columns]
            temp_promo = temp_promo.melt(id_vars='Date', var_name='Promo Category', value_name='Value')
            fig_promo = px.line(
                temp_promo,
                x='Date',
                y='Value',
                color='Promo Category',
                title="Promo Trends Over Time",
                labels={'Value': 'Promo Quantities', 'Date': 'Date'}
            )
            st.plotly_chart(fig_promo, use_container_width=True)
        else:
            st.info("Select at least one promo column and ensure a valid actual date range.")

elif page == "Summary Statistics":
    st.subheader("Summary Statistics & Visualizations")
    if not data_valid:
        st.warning("Please select a valid actual data date range.")
    else:
        if selected_columns_data and not filtered_data.empty:
            summary_stats = filtered_data[selected_columns_data].describe().transpose()
            st.write("### Descriptive Statistics for Selected Categories")
            st.dataframe(summary_stats, use_container_width=True)

            st.subheader("Histogram of Selected Categories")
            temp_hist = filtered_data[selected_columns_data]
            fig_hist = go.Figure()
            for col in selected_columns_data:
                fig_hist.add_trace(go.Histogram(
                    x=temp_hist[col],
                    opacity=0.4,
                    name=col,
                    histnorm='probability'
                ))
            fig_hist.update_layout(
                barmode='overlay',
                title_text='Histogram of Selected Categories',
                xaxis_title_text='Value',
                yaxis_title_text='Probability',
                legend_title_text='Categories'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Box Plots for Selected Categories")
            temp_box = filtered_data[['Date'] + selected_columns_data]
            temp_box = temp_box.melt(id_vars='Date', var_name='Category', value_name='Value')
            fig_box = px.box(
                temp_box,
                x='Category',
                y='Value',
                title="Box Plots for Selected Categories",
                labels={'Value': 'Values', 'Category': 'Categories'},
                color='Category'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Select categories from the actual inbound values and a valid actual date range.")

elif page == "Time Series Decomposition":
    st.subheader("Time Series Decomposition")
    if not data_valid:
        st.warning("Please select a valid actual data date range.")
    else:
        if selected_columns_data and not filtered_data.empty:
            selected_decomposition_col = st.selectbox(
                "Select a Column for Time Series Decomposition",
                options=selected_columns_data
            )
            if selected_decomposition_col:
                decomposition_data = filtered_data.set_index('Date')
                decomposition_result = seasonal_decompose(decomposition_data[selected_decomposition_col], model='additive', period=7)
                fig_decomposition = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                                  subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                fig_decomposition.add_trace(go.Scatter(x=decomposition_data.index, y=decomposition_result.observed, name='Observed'), row=1, col=1)
                fig_decomposition.add_trace(go.Scatter(x=decomposition_data.index, y=decomposition_result.trend, name='Trend'), row=2, col=1)
                fig_decomposition.add_trace(go.Scatter(x=decomposition_data.index, y=decomposition_result.seasonal, name='Seasonal'), row=3, col=1)
                fig_decomposition.add_trace(go.Scatter(x=decomposition_data.index, y=decomposition_result.resid, name='Residual'), row=4, col=1)
                fig_decomposition.update_layout(height=800, title_text="Time Series Decomposition")
                st.plotly_chart(fig_decomposition, use_container_width=True)
        else:
            st.info("Select a category and ensure a valid actual date range.")

elif page == "Metrics":
    st.subheader("Model Performance Metrics")

    # Include Existing MAPE and Existing Accuracy columns as requested
    # Columns: Category, MAE, RMSE, MAPE, MAPE CI, Accuracy, Existing MAPE, Existing Accuracy, Difference
    # Data as previously given:

    # DRY:
    # Model: MAPE=9.83%, Acc=90.17%
    # Existing: MAPE=12.38%, Acc=87.62%
    # Diff=+2.55%
    # MAPE CI=[7.53%, 12.29%]
    # MAE=8198.24, RMSE=9525.85

    # FROZEN:
    # Model: MAPE=17.17%, Acc=82.83%
    # Existing: MAPE=21.12%, Acc=78.88%
    # Diff=+3.95%
    # MAPE CI=[11.11%, 23.40%]
    # MAE=1261.08, RMSE=1609.62

    # ULTRAFRESH:
    # Model: MAPE=23.38%, Acc=76.62%
    # Existing: MAPE=26.08%, Acc=73.92%
    # Diff=+2.70%
    # MAPE CI=[12.03%, 40.30%]
    # MAE=942.18, RMSE=1272.51

    # FRESH:
    # Model: MAPE=6.42%, Acc=93.58%
    # Existing: MAPE=5.06%, Acc=94.94%
    # Diff=-1.36%
    # MAPE CI=[4.31%, 8.85%]
    # MAE=4387.42, RMSE=5585.08

    metrics_data = [
        {
            "Category": "Dry",
            "MAE": "8198.24",
            "RMSE": "9525.85",
            "MAPE": "9.83%",
            "MAPE CI": "[7.53%, 12.29%]",
            "Accuracy": "90.17%",
            "Existing MAPE": "12.38%",
            "Existing Accuracy": "87.62%",
            "Difference": "<span>+2.55%</span>"
        },
        {
            "Category": "Frozen",
            "MAE": "1261.08",
            "RMSE": "1609.62",
            "MAPE": "17.17%",
            "MAPE CI": "[11.11%, 23.40%]",
            "Accuracy": "82.83%",
            "Existing MAPE": "21.12%",
            "Existing Accuracy": "78.88%",
            "Difference": "<span>+3.95%</span>"
        },
        {
            "Category": "Ultrafresh",
            "MAE": "942.18",
            "RMSE": "1272.51",
            "MAPE": "23.38%",
            "MAPE CI": "[12.03%, 40.30%]",
            "Accuracy": "76.62%",
            "Existing MAPE": "26.08%",
            "Existing Accuracy": "73.92%",
            "Difference": "<span>+2.70%</span>"
        },
        {
            "Category": "Fresh",
            "MAE": "4387.42",
            "RMSE": "5585.08",
            "MAPE": "6.42%",
            "MAPE CI": "[4.31%, 8.85%]",
            "Accuracy": "93.58%",
            "Existing MAPE": "5.06%",
            "Existing Accuracy": "94.94%",
            "Difference": "<span>-1.36%</span>"
        }
    ]

    metrics_df = pd.DataFrame(metrics_data)
    metrics_html = metrics_df.to_html(escape=False, index=False, classes='styled-table')
    st.markdown(
        f"""
        <div style="width:100%; overflow-x:auto;">
        {metrics_html}
        </div>
        """,
        unsafe_allow_html=True
    )
