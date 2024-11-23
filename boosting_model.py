def train_and_evaluate(df, response_variable, n_trials=50, test_size=14, output_dir='.', n_bootstraps=5000, ci_level=95):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import optuna
    import matplotlib.pyplot as plt
    import os

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['month'] = df['Date'].dt.month
    df['Day of Week'] = df['Date'].dt.day_name()
    df['rolling_mean_7'] = df[response_variable].rolling(window=7).mean()
    df['cumulative_sum'] = df[response_variable].cumsum()
    for lag in range(1, 15):
        df[f'lag_{lag}'] = df[response_variable].shift(lag)
    df['month_lag_1'] = df['month'] * df['lag_1']
    df = df.dropna()

    all_fc_columns = ['Dry Fc', 'Fresh Fc', 'Ultrafresh Fc', 'Frozen Fc', 'Total Inbound Fc', 'Pre-orders Fc']
    response_to_fc = {
        'Dry Actuals': 'Dry Fc',
        'Fresh': 'Fresh Fc',
        'Ultrafresh': 'Ultrafresh Fc',
        'Frozen': 'Frozen Fc',
        'Total Inbound': 'Total Inbound Fc'
    }
    fc_column = response_to_fc.get(response_variable)
    if fc_column in df.columns:
        fc_series = df[fc_column].reset_index(drop=True)
        df = df.drop(fc_column, axis=1)
    else:
        fc_series = None

    if response_variable in ['Dry Actuals', 'Fresh', 'Ultrafresh', 'Frozen']:
        exclude_columns = ['Total Inbound', 'Total Inbound Fc']
        df = df.drop(columns=[col for col in exclude_columns if col in df.columns], axis=1)

    other_fc_columns = [col for col in all_fc_columns if col != fc_column]
    df = df.drop(columns=[col for col in other_fc_columns if col in df.columns], axis=1)
    if 'Category' in df.columns:
        df = df.drop('Category', axis=1)

    train_df = df.iloc[:-test_size].reset_index(drop=True)
    test_df = df.iloc[-test_size:].reset_index(drop=True)
    X_train = train_df.drop(['Date', response_variable], axis=1)
    y_train = train_df[response_variable]
    X_test = test_df.drop(['Date', response_variable], axis=1)
    y_test = test_df[response_variable]

    categorical_cols = ['Day of Week', 'month']
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"Best RMSE from Optuna: {study.best_value}")
    print(f"Best Hyperparameters: {best_params}")

    final_model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42
    )
    final_model.fit(X_train_encoded, y_train)
    y_pred = final_model.predict(X_test_encoded)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"Final RMSE on Test Set: {final_rmse}")
    print(f"Mean Absolute Percentage Error (MAPE) on Test Set: {mape:.2f}%")
    print(f"R-squared on Test Set: {r2:.4f}")

    rmse_bootstrap = []
    mape_bootstrap = []
    rng = np.random.default_rng(seed=42)

    y_test_array = y_test.values
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_test), len(y_test))
        y_test_sample = y_test_array[indices]
        y_pred_sample = y_pred[indices]

        
        mask = y_test_sample != 0
        if np.any(mask):
            rmse_sample = np.sqrt(mean_squared_error(y_test_sample, y_pred_sample))
            mape_sample = np.mean(np.abs((y_test_sample[mask] - y_pred_sample[mask]) / y_test_sample[mask])) * 100

            rmse_bootstrap.append(rmse_sample)
            mape_bootstrap.append(mape_sample)

    alpha = (100 - ci_level) / 2
    lower_rmse = np.percentile(rmse_bootstrap, alpha)
    upper_rmse = np.percentile(rmse_bootstrap, 100 - alpha)
    lower_mape = np.percentile(mape_bootstrap, alpha)
    upper_mape = np.percentile(mape_bootstrap, 100 - alpha)

    print(f"{ci_level}% Confidence Interval for RMSE: [{lower_rmse:.2f}, {upper_rmse:.2f}]")
    print(f"{ci_level}% Confidence Interval for MAPE: [{lower_mape:.2f}%, {upper_mape:.2f}%]")


    feature_importance = final_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 10))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    feature_importance_plot_path = os.path.join(output_dir, f'feature_importance_{response_variable.replace(" ", "_")}.png')
    plt.savefig(feature_importance_plot_path)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_df['Date'], y_test, label="Actual", marker='o')
    plt.plot(test_df['Date'], y_pred, label="Model's  Prediction", linestyle='--', marker='x')
    if fc_series is not None:
        plt.plot(test_df['Date'], fc_series.iloc[-test_size:].reset_index(drop=True), 
                 label=f"Existing {fc_column}", linestyle=':', marker='s')
    plt.title(f"Actual vs Predicted vs Existing {response_variable}")
    plt.xlabel("Date")
    plt.ylabel(response_variable)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    prediction_plot_path = os.path.join(output_dir, f'actual_vs_predicted_{response_variable.replace(" ", "_")}.png')
    plt.savefig(prediction_plot_path)
    plt.close()

    print(f"Feature importance plot saved to: {feature_importance_plot_path}")
    print(f"Actual vs. Predicted plot saved to: {prediction_plot_path}")

    return final_model, best_params, final_rmse, mape





if __name__ == "__main__":
    import pandas as pd
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train and evaluate the model.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file.')
    parser.add_argument('--response', type=str, required=True, help='Response variable to predict.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of trials for hyperparameter optimization.')
    parser.add_argument('--test_size', type=int, default=14, help='Number of observations to use for testing.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output plots.')

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    response_variable = args.response
    n_trials = args.epochs
    test_size = args.test_size
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    final_model, best_params, final_rmse, mape = train_and_evaluate(
        df=df,
        response_variable=response_variable,
        n_trials=n_trials,
        test_size=test_size,
        output_dir=output_dir
    )

    print(f"Best Hyperparameters: {best_params}")
    print(f"Final RMSE: {final_rmse}")
    print(f"MAPE: {mape:.2f}%")