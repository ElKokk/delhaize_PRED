.



###FORECASTING SOFTWARE ####
--------------------------------------------
#### FOR FINAL_MODEL.3.1.py ####
PRE-REQUISITES


Python installed (version 3.7 or later). Install the following Python packages

pip install pandas numpy xgboost scikit-learn optuna matplotlib shap

-----------------------------------------------------~
~~
IMPORTANT NOTE: The data file the software accepts right now is the "combined_for_model2.csv" in the folder. That is a cleaned and preprocessed dataset from the origianl data. Any other data input file needs to follow the same structure and variable names.
~~

----------  python script_name.py --dataset PATH_TO_CSV --response RESPONSE_VARIABLE --epochs NUM_TRIALS --output_dir OUTPUT_DIR --forecast_weeks NUM_WEEKS   ----------


Arguments:

--dataset: Path to the input dataset (CSV file).

--response: The name of the column containing the target variable.

--epochs: (Optional) Number of trials for hyperparameter tuning. Default is 50.

--output_dir: (Optional) Directory where results will be saved. Default is the current directory.

--forecast_weeks: (Optional) Number of weeks to forecast. Default is 3.

EXAMPLE USAGE:

python FINAL_MODEL_3.1.py --dataset 'combined_for_model2.csv' --response "Dry Actuals" --epochs 1 --output_dir 'RESULTS' --forecast_weeks 3
