import pandas as pd


def preprocess_inbound_data2(inbound_df, output_csv_path):
    """
    Preprocesses the inbound data DataFrame and exports the result to a CSV file.

    Parameters:
    - inbound_df (pd.DataFrame): The input DataFrame to preprocess.
    - output_csv_path (str): The file path where the preprocessed DataFrame will be saved as a CSV.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame ready for analysis.
    """
    import pandas as pd

    df = inbound_df.copy()

    df.drop(df.columns[0], axis=1, inplace=True)
    df.columns = ['Category', 'Subcategory'] + df.columns[2:].tolist()
    df['Category'] = df['Category'].fillna(method='ffill')
    df.drop(index=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=[col for col in df.columns if str(col).lower().startswith('wk')], inplace=True)
    new_columns = df.columns[:2].tolist() + df.iloc[0, 2:].tolist()
    df.columns = new_columns
    df = df[1:].reset_index(drop=True)
    if 'Category' not in df.columns or 'Subcategory' not in df.columns:
        raise KeyError("Required columns 'Category' and 'Subcategory' are missing after column renaming.")
    if pd.isna(df.iloc[0]['Category']) and pd.isna(df.iloc[0]['Subcategory']):
        df = df[1:].reset_index(drop=True)
    melted_df = df.melt(
        id_vars=['Category', 'Subcategory'],
        var_name='Date',
        value_name='Inbound_Forecast'
    )
    melted_df['Date'] = pd.to_datetime(melted_df['Date'], errors='coerce')
    melted_df['Inbound_Forecast'] = pd.to_numeric(melted_df['Inbound_Forecast'], errors='coerce')
    melted_df.dropna(subset=['Date', 'Inbound_Forecast'], inplace=True)
    pivot_df = melted_df.pivot_table(
        index=['Date', 'Category'],
        columns='Subcategory',
        values='Inbound_Forecast',
        aggfunc='first'
    ).reset_index()
    pick_capacity_data = pivot_df[pivot_df['Category'] == 'Pick Capacity']
    pick_capacity_totals = pick_capacity_data.groupby('Date').sum(numeric_only=True).sum(axis=1).reset_index()
    pick_capacity_totals.columns = ['Date', 'Pick Capacity']
    pivot_df = pivot_df.merge(pick_capacity_totals, on='Date', how='left')
    hd_am_pm_data = pivot_df[pivot_df['Category'].isin(['HD AM', 'HD PM'])]
    if 'Basket Update' in hd_am_pm_data.columns:
        hd_totals = hd_am_pm_data.groupby('Date')['Basket Update'].sum().reset_index()
        hd_totals.columns = ['Date', 'HD Total']
        pivot_df = pivot_df.merge(hd_totals, on='Date', how='left')
    else:
        pivot_df['HD Total'] = None
    collect_data = pivot_df[pivot_df['Category'] == 'Collect']
    if 'Basket Update' in collect_data.columns:
        collect_totals = collect_data.groupby('Date')['Basket Update'].sum().reset_index()
        collect_totals.columns = ['Date', 'Collect Total']
        pivot_df = pivot_df.merge(collect_totals, on='Date', how='left')
    else:
        pivot_df['Collect Total'] = None
    final_dataset = pivot_df[pivot_df['Category'] == 'Inbound']
    columns_to_drop = [
        'Actuals', 'Basket Fc', 'Basket Update', 'Geschat', 'Leverdag', 'Middag',
        'Nacht', 'Ochtend', 'Orders Fc', 'Orders Update', 'Picking Fc', 'Picking Update',
        'Total Picks Forecast', 'Total Picks Update'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in final_dataset.columns]
    final_dataset.drop(columns=columns_to_drop, inplace=True)
    final_dataset['Date'] = pd.to_datetime(final_dataset['Date'])
    final_dataset.insert(1, 'Day of Week', final_dataset['Date'].dt.day_name())
    final_dataset = final_dataset[final_dataset['Day of Week'] != 'Sunday'].reset_index(drop=True)
    final_dataset.to_csv(output_csv_path, index=False)

    return final_dataset
