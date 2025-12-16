import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
import matplotlib.pyplot as plt
import os
import shap
from math import pi
import warnings
warnings.filterwarnings('ignore')

def safe_forward_fill(df):
    """
    Handles missing values in numeric columns by applying forward-fill followed by median imputation.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: Processed DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill()
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
    return df

def compute_metrics(y_true, y_pred):
    """
    Computes regression metrics: MAE, RMSE, MAPE, and R-squared.
    
    Parameters:
        y_true (np.array or pd.Series): Actual target values.
        y_pred (np.array or pd.Series): Predicted target values.

    Returns:
        tuple: MAE, RMSE, MAPE, R-squared scores.
    """
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100 if np.any(y_true!=0) else np.nan
    r2 = r2_score(y_true,y_pred)
    return mae,rmse,mape,r2

def bootstrap_metrics(y_true,y_pred,n_bootstraps=5000,ci_level=95):
    """
    Computes confidence intervals for metrics using bootstrap resampling.

    Parameters:
        y_true (np.array or pd.Series): True target values.
        y_pred (np.array or pd.Series): Predicted values.
        n_bootstraps (int): Number of bootstrap resamples. Default is 5000.
        ci_level (int): Confidence interval level (e.g., 95%).

    Returns:
        tuple: Confidence intervals for MAE, RMSE, and MAPE.
    """
    mae_bs=[]
    rmse_bs=[]
    mape_bs=[]
    rng=np.random.default_rng(seed=42)
    for _ in range(n_bootstraps):
        idx=rng.integers(0,len(y_true),len(y_true))
        yt_sample=y_true[idx]
        yp_sample=y_pred[idx]
        mask=yt_sample!=0
        if np.any(mask):
            mae_s=mean_absolute_error(yt_sample[mask],yp_sample[mask])
            rmse_s=np.sqrt(mean_squared_error(yt_sample[mask],yp_sample[mask]))
            mape_s=np.mean(np.abs((yt_sample[mask]-yp_sample[mask])/yt_sample[mask]))*100
            mae_bs.append(mae_s)
            rmse_bs.append(rmse_s)
            mape_bs.append(mape_s)
    alpha=(100-ci_level)/2
    def ci(arr):
        if len(arr)>0:
            return (np.percentile(arr,alpha),np.percentile(arr,100-alpha))
        else:
            return (np.nan,np.nan)
    mae_ci=ci(mae_bs)
    rmse_ci=ci(rmse_bs)
    mape_ci=ci(mape_bs)
    return mae_ci,rmse_ci,mape_ci

def identify_peaks(df,response_variable,window=19):
    """
    Identifies peaks in a specified response variable using a rolling window.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the response variable.
        response_variable (str): The column name of the response variable to analyze.
        window (int): The size of the rolling window. Default is 19.

    Returns:
        pd.Series: A boolean Series indicating the location of peaks.
    """
    rolling_max=df[response_variable].rolling(window=window,min_periods=1,center=False).max()
    peaks=(df[response_variable]==rolling_max)
    return peaks.fillna(False)

def add_peak_features(df,response_variable):
    """
    Adds features related to peaks in the response variable, including the days since the last peak.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        response_variable (str): The column name of the response variable to analyze.

    Returns:
        pd.DataFrame: DataFrame with an additional column for days since the last peak.
    """
    peaks=identify_peaks(df,response_variable,window=19)
    df['days_since_last_peak']=np.nan
    last_peak_idx=-1
    for i in range(len(df)):
        if peaks.iloc[i]:
            last_peak_idx=i
        df.at[i,'days_since_last_peak']=i-last_peak_idx if last_peak_idx>=0 else np.nan
    df['days_since_last_peak']=df['days_since_last_peak'].ffill().fillna(len(df))
    return df

def add_fourier_terms(df,period=6,order=3):
    """
    Adds Fourier series terms to model seasonality in the data.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        period (int): The period of the seasonal cycle. Default is 6.
        order (int): The number of harmonics to include. Default is 3.

    Returns:
        pd.DataFrame: DataFrame with additional Fourier terms as columns.
    """
    t=np.arange(len(df))
    for k in range(1,order+1):
        df[f'sin_{k}_{period}']=np.sin(2*pi*k*t/period)
        df[f'cos_{k}_{period}']=np.cos(2*pi*k*t/period)
    return df

def add_polynomial_time_index(df):
    """
    Adds polynomial terms for time to capture non-linear trends.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional columns for time (t), time squared (t2), and time cubed (t3).
    """
    t=np.arange(len(df))
    df['t']=t
    df['t2']=t**2
    df['t3']=t**3
    return df

def add_interactions(df,response_variable,special_lags=[6,7,8,14,16,21]):
    """
    Adds interaction terms between lagged features and other variables such as 'Day_Monday', 'week', and 'days_since_last_peak'.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        response_variable (str): The column name of the response variable.
        special_lags (list): List of lag values for which interactions should be created. Default is [6, 7, 8, 14, 16, 21].

    Returns:
        pd.DataFrame: DataFrame with additional interaction features.
    """
    df=df.loc[:,~df.columns.duplicated()]
    day_monday_col='Day_Monday' if 'Day_Monday' in df.columns else None

    lagged_cols=[c for c in df.columns if '_lag_' in c]

    if day_monday_col:
        for lc in lagged_cols:
            inter_col=f'{lc}*Day_Monday'
            if inter_col not in df.columns:
                df[inter_col]=df[lc]*df[day_monday_col]
    if 'week' in df.columns:
        for lc in lagged_cols:
            inter_col=f'{lc}*week'
            if inter_col not in df.columns:
                df[inter_col]=df[lc]*df['week']
    if 'days_since_last_peak' in df.columns:
        for lc in lagged_cols:
            inter_col=f'{lc}*days_since_last_peak'
            if inter_col not in df.columns:
                df[inter_col]=df[lc]*df['days_since_last_peak']

    df=df.loc[:,~df.columns.duplicated()]
    return df

def add_monday_residual_feature(df):
    """
    Adds a feature representing residual effects on Mondays.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with an additional column for Monday residual effects.
    """
    if 'Day_Monday' in df.columns:
        df['monday_residual_peak']=df['Day_Monday'].values
    else:
        df['monday_residual_peak']=0
    return df

def add_two_peak_pattern(df):
    """
    Defines patterns for a two-peak week, such as the start of peaks and their offsets.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional columns for two-peak patterns.
    """
    earliest_week=df['week'].min()
    df['week_offset']=df['week']-earliest_week
    df['peak_start_day']=(1+df['week_offset'])%6
    df['peak_day1']=df['peak_start_day']
    df['peak_day2']=(df['peak_start_day']+1)%6
    df['peak_pattern_index']=df['peak_start_day']
    return df

def create_features(df, response_variable, product_cols):
    """
    Creates and engineers features for modeling, including polynomial terms, Fourier terms, lags, and interactions.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        response_variable (str): The column name of the response variable to model.
        product_cols (list): List of product-specific column names to include in lag calculations.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    day_names=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

    df=add_polynomial_time_index(df)
    df=add_fourier_terms(df,period=6,order=3)
    df=add_peak_features(df,response_variable)

    if not pd.api.types.is_categorical_dtype(df['Day of Week']):
        df['Day of Week']=pd.Categorical(df['Day of Week'],categories=day_names,ordered=True)

    df['rolling_mean_same_day_4weeks']=df.groupby('Day of Week')[response_variable].transform(
        lambda x:x.shift(6).rolling(window=24,min_periods=1).mean()
    )
    df['ema_same_day_4weeks']=df.groupby('Day of Week')[response_variable].transform(
        lambda x:x.shift(6).ewm(span=24,min_periods=1,adjust=False).mean()
    )

    df['prev_day_of_week_idx']=(df['Day of Week'].cat.codes-1)%6
    day_map={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday'}
    df['Prev Day Name']=df['prev_day_of_week_idx'].map(day_map)

    df['rolling_mean_prev_day_4weeks']=df.groupby('Prev Day Name')[response_variable].transform(
        lambda x:x.shift(6).rolling(window=24,min_periods=1).mean()
    )
    df['ema_prev_day_4weeks']=df.groupby('Prev Day Name')[response_variable].transform(
        lambda x:x.shift(6).ewm(span=24,min_periods=1,adjust=False).mean()
    )

    day_dummies=pd.get_dummies(df['Day of Week'],prefix='Day',drop_first=False)
    df=pd.concat([df,day_dummies],axis=1)

    df['season']=df['month']%12//3+1

    df=add_monday_residual_feature(df)
    df=add_two_peak_pattern(df)

    vars_to_lag=[response_variable]+product_cols
    response_orig_col=response_variable+'_orig'
    if response_orig_col in df.columns:
        vars_to_lag.append(response_orig_col)

    special_lags=[6,7,8,14,16,21]
    for var in vars_to_lag:
        if var in df.columns:
            for lag in special_lags:
                df[f'{var}_lag_{lag}']=df[var].shift(lag)

    # Drop product_cols and response_variable_orig after lagging
    for pc in product_cols:
        if pc in df.columns:
            df.drop(pc,axis=1,inplace=True,errors='ignore')

    # If response_variable_orig was lagged, drop its original now
    if response_orig_col in df.columns:
        df.drop(response_orig_col,axis=1,inplace=True,errors='ignore')

    # We do NOT drop the main response_variable here.

    df=add_interactions(df,response_variable,special_lags=special_lags)

    df.replace([np.inf,-np.inf],0,inplace=True)
    df=safe_forward_fill(df)

    if pd.api.types.is_categorical_dtype(df['Day of Week']):
        df['Day of Week']=df['Day of Week'].cat.codes
    if 'Prev Day Name' in df.columns:
        df['Prev Day Name']=df['Prev Day Name'].astype('category').cat.codes

    for col in df.columns:
        if df[col].dtype=='O':
            df[col]=df[col].astype('category').cat.codes

    return df

def train_model(X_train,y_train,n_trials=50):
    def objective(trial):
        params={
            'n_estimators':trial.suggest_int('n_estimators',20,700),
            'max_depth':trial.suggest_int('max_depth',3,10),
            'learning_rate':trial.suggest_float('learning_rate',0.001,0.4,log=True),
            'subsample':trial.suggest_float('subsample',0.6,1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
            'min_child_weight':trial.suggest_int('min_child_weight',1,10),
            'gamma':trial.suggest_float('gamma',0,10),
            'reg_alpha':trial.suggest_float('reg_alpha',0,1),
            'reg_lambda':trial.suggest_float('reg_lambda',0,1),
            'objective':'reg:squarederror',
            'random_state':42,
            'tree_method':'hist',
            'device':'cuda',
            'verbosity':0
        }

        model=XGBRegressor(**params)
        tscv=TimeSeriesSplit(n_splits=5)
        scores=[]
        for tr_idx,val_idx in tscv.split(X_train):
            X_t,X_v=X_train.iloc[tr_idx],X_train.iloc[val_idx]
            y_t,y_v=y_train.iloc[tr_idx],y_train.iloc[val_idx]

            X_t=safe_forward_fill(X_t)
            X_v=safe_forward_fill(X_v)
            y_t=y_t.ffill().fillna(y_t.median())
            y_v=y_v.ffill().fillna(y_v.median())

            model.fit(X_t,y_t,eval_set=[(X_v,y_v)],verbose=False)
            y_pred=model.predict(X_v)
            mae=mean_absolute_error(y_v,y_pred)
            scores.append(mae)
        return np.mean(scores)

    study=optuna.create_study(direction='minimize')
    study.optimize(objective,n_trials=n_trials)
    best_params=study.best_params
    print(f"Best MAE from Optuna: {study.best_value:.2f}")
    print(f"Best Hyperparameters: {best_params}")

    if 'reg_alpha' in best_params:
        best_params['reg_alpha']=best_params['reg_alpha']+0.1
    if 'reg_lambda' in best_params:
        best_params['reg_lambda']=best_params['reg_lambda']+0.1

    X_train=safe_forward_fill(X_train)
    y_train=y_train.ffill().fillna(y_train.median())

    final_model=XGBRegressor(**best_params,objective='reg:squarederror',random_state=42,tree_method='hist',verbosity=0)
    final_model.fit(X_train,y_train,eval_set=[(X_train,y_train)],verbose=False)
    return final_model,best_params

def map_response_to_fc(response_variable):
    mapping={
        'Dry Actuals':'Dry Fc',
        'Fresh':'Fresh Fc',
        'Ultrafresh':'Ultrafresh Fc',
        'Frozen':'Frozen Fc',
        'Total Inbound':'Total Inbound Fc'
    }
    return mapping.get(response_variable,None)

def train_and_forecast(df,response_variable,n_trials=50,output_dir='.',n_bootstraps=5000,ci_level=95,forecast_weeks=3):
    forecast_days_per_week=6

    df['Date']=pd.to_datetime(df['Date'],errors='coerce').dt.tz_localize(None)
    df=df.dropna(subset=['Date']).reset_index(drop=True)
    df=df.sort_values('Date').reset_index(drop=True)

    df_original=df.copy()
    fc_column=map_response_to_fc(response_variable)
    fc_available=(fc_column in df_original.columns) if fc_column else False
    if fc_available:
        fc_values_all=df_original[fc_column].values
    else:
        fc_values_all=None

    all_fc_columns=['Dry Fc','Fresh Fc','Ultrafresh Fc','Frozen Fc','Total Inbound Fc','Pre-orders Fc']
    drop_cols=['Date','Category','year','month','week','day_of_year','t','t2','t3']+all_fc_columns
    base_numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    product_cols = [c for c in base_numeric_cols if c not in drop_cols and c!=response_variable]

    df = df.drop(columns=[c for c in all_fc_columns if c in df.columns], errors='ignore')
    if 'Category' in df.columns:
        df=df.drop('Category',axis=1,errors='ignore')

    df['year']=df['Date'].dt.year
    df['month']=df['Date'].dt.month
    df['week']=df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_year']=df['Date'].dt.dayofyear

    df=create_features(df,response_variable,product_cols)

    overall_actuals=[]
    overall_predictions=[]
    overall_fc_forecasts=[] if fc_available else None

    forecast_days=forecast_weeks*forecast_days_per_week
    for week_num in range(1,forecast_weeks+1):
        test_start=len(df)-forecast_days+(week_num-1)*forecast_days_per_week
        test_end=test_start+forecast_days_per_week
        if test_start<0 or test_end>len(df):
            print(f"Not enough data for week {week_num}")
            break

        train_data=df.iloc[:test_start]
        y_train_full=train_data[response_variable]
        X_train_full=train_data.drop(columns=['Date',response_variable],errors='ignore')

        X_train_full=safe_forward_fill(X_train_full)
        y_train_full=y_train_full.ffill().fillna(y_train_full.median())

        model_all,params_all=train_model(X_train_full,y_train_full,n_trials=n_trials)
        model_feature_names=model_all.get_booster().feature_names

        test_week_df=df.iloc[test_start:test_end].copy().reset_index(drop=True)
        original_actuals_week=test_week_df[response_variable].values

        if fc_available:
            week_dates=test_week_df['Date'].values
            fc_week_mask=df_original['Date'].isin(week_dates)
            existing_forecast_week=df_original.loc[fc_week_mask,fc_column].values
        else:
            existing_forecast_week=None

        X_test_week=test_week_df.drop(columns=['Date',response_variable],errors='ignore')
        X_test_week=safe_forward_fill(X_test_week)
        predictions=[]
        for i in range(len(test_week_df)):
            current_features=X_test_week.iloc[[i]].copy()
            missing_cols=set(model_feature_names)-set(current_features.columns)
            for col in missing_cols:
                current_features[col]=0
            current_features=current_features[model_feature_names].astype(float)
            y_pred=model_all.predict(current_features.values)[0]
            predictions.append(y_pred)

        predictions=np.array(predictions)
        start_date=test_week_df['Date'].iloc[0]
        end_date=test_week_df['Date'].iloc[-1]

        mae_w,rmse_w,mape_w,r2_w=compute_metrics(original_actuals_week,predictions)
        print(f"\n===== Forecasting Week {week_num} ({start_date.date()} to {end_date.date()}) =====")
        print(f"MAE: {mae_w:.2f}, RMSE: {rmse_w:.2f}, MAPE: {mape_w:.2f}%, R2: {r2_w:.4f}")
        mae_ci_w,rmse_ci_w,mape_ci_w=bootstrap_metrics(original_actuals_week,predictions,n_bootstraps,ci_level)
        print(f"{ci_level}% CI MAE: [{mae_ci_w[0]:.2f}, {mae_ci_w[1]:.2f}]")
        print(f"{ci_level}% CI RMSE: [{rmse_ci_w[0]:.2f}, {rmse_ci_w[1]:.2f}]")
        print(f"{ci_level}% CI MAPE: [{mape_ci_w[0]:.2f}%, {mape_ci_w[1]:.2f}%]")

        week_pred_df=pd.DataFrame({
            'Date':test_week_df['Date'],
            'Original_Actual':original_actuals_week,
            'Model_Prediction':predictions
        })
        if fc_available and existing_forecast_week is not None:
            week_pred_df['Existing_Forecast']=existing_forecast_week
            fc_mae_w,fc_rmse_w,fc_mape_w,fc_r2_w=compute_metrics(original_actuals_week,existing_forecast_week)
            print(f"Existing {fc_column} - MAE: {fc_mae_w:.2f}, RMSE: {fc_rmse_w:.2f}, MAPE: {fc_mape_w:.2f}%, R2: {fc_r2_w:.4f}")

        week_pred_csv=os.path.join(output_dir,f'predictions_week_{week_num}_{response_variable.replace(" ","_")}.csv')
        week_pred_df.to_csv(week_pred_csv,index=False)
        print(f"Predictions for Week {week_num} saved to: {week_pred_csv}")

        plt.figure(figsize=(12,6))
        plt.plot(week_pred_df['Date'],week_pred_df['Original_Actual'],label="Actual",marker='o')
        plt.plot(week_pred_df['Date'],week_pred_df['Model_Prediction'],label="Prediction",linestyle='--',marker='x')
        if fc_available and 'Existing_Forecast' in week_pred_df.columns:
            plt.plot(week_pred_df['Date'],week_pred_df['Existing_Forecast'],label=f"Existing {fc_column}",linestyle=':',marker='s')
        plt.title(f"Week {week_num}: Actual vs Predicted vs Existing")
        plt.xlabel("Date")
        plt.ylabel(response_variable)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        week_plot_path=os.path.join(output_dir,f'actual_vs_predicted_week_{week_num}_{response_variable.replace(" ","_")}.png')
        plt.savefig(week_plot_path)
        plt.close()
        print(f"Plot for Week {week_num} saved to: {week_plot_path}")

        explainer=shap.Explainer(model_all)
        X_test_full = safe_forward_fill(test_week_df.drop(columns=['Date',response_variable],errors='ignore'))
        missing_cols=set(model_feature_names)-set(X_test_full.columns)
        for col in missing_cols:
            X_test_full[col]=0
        X_test_full=X_test_full[model_feature_names].astype(float)
        shap_values_week=explainer(X_test_full)

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values_week.values,X_test_full,plot_type="dot",show=False)
        shap_summary_path=os.path.join(output_dir,f'shap_summary_beeswarm_week_{week_num}_{response_variable.replace(" ","_")}.png')
        plt.tight_layout()
        plt.savefig(shap_summary_path,bbox_inches='tight')
        plt.close()
        print(f"SHAP summary beeswarm for Week {week_num} saved to: {shap_summary_path}")

        for i in range(len(predictions)):
            day_date=test_week_df['Date'].iloc[i]
            day_instance=X_test_full.iloc[[i]]
            day_shap=explainer(day_instance)

            abs_values=np.abs(day_shap.values[0,:])
            top5_inds=np.argsort(abs_values)[-5:]
            day_shap_values_top5=day_shap.values[0, top5_inds]
            day_instance_top5=day_instance.iloc[:, top5_inds]

            force_plot_path=os.path.join(output_dir,f'shap_force_plot_week_{week_num}_day_{i+1}_{response_variable.replace(" ","_")}.png')
            shap.force_plot(explainer.expected_value,day_shap_values_top5,day_instance_top5,matplotlib=True,show=False)
            plt.savefig(force_plot_path,bbox_inches='tight')
            plt.close()
            print(f"SHAP force plot for Week {week_num}, Day {i+1} ({day_date.date()}) saved to: {force_plot_path}")

            day_shap_df=pd.DataFrame(day_shap.values,columns=X_test_full.columns)
            day_shap_df['Model_Prediction']=predictions[i]
            day_shap_df['Original_Actual']=original_actuals_week[i]
            day_shap_csv_path=os.path.join(output_dir,f'shap_values_week_{week_num}_day_{i+1}_{response_variable.replace(" ","_")}.csv')
            day_shap_df.to_csv(day_shap_csv_path,index=False)
            print(f"SHAP values for Week {week_num}, Day {i+1} exported to: {day_shap_csv_path}")

        fi=model_all.feature_importances_
        importance_df=pd.DataFrame({'Feature':model_feature_names,'Importance':fi}).sort_values(by='Importance',ascending=False)

        fi_csv_path=os.path.join(output_dir,f'feature_importance_week_{week_num}_{response_variable.replace(" ","_")}.csv')
        importance_df.to_csv(fi_csv_path,index=False)
        print(f"Feature importances for Week {week_num} exported to: {fi_csv_path}")

        # Integrate predictions as actuals into df
        for i in range(forecast_days_per_week):
            overwrite_date=test_week_df['Date'].iloc[i]
            pred_val=predictions[i]
            df.loc[df['Date']==overwrite_date,response_variable]=pred_val

        overall_actuals.extend(original_actuals_week)
        overall_predictions.extend(predictions)
        if fc_available and existing_forecast_week is not None:
            overall_fc_forecasts.extend(existing_forecast_week)
        elif fc_available:
            overall_fc_forecasts.extend([np.nan]*len(original_actuals_week))

        updated_response=df[['Date',response_variable]].copy()
        df=df_original.copy()
        df=pd.merge(df,updated_response,on='Date',how='left',suffixes=('_orig',''))
        df[response_variable]=df[response_variable]

        # After merging, drop response_variable_orig if it exists to avoid double lagging next time
        response_orig_col=response_variable+'_orig'
        if response_orig_col in df.columns:
            df.drop(response_orig_col,axis=1,inplace=True,errors='ignore')

        all_fc_columns=['Dry Fc','Fresh Fc','Ultrafresh Fc','Frozen Fc','Total Inbound Fc','Pre-orders Fc']
        df = df.drop(columns=[c for c in all_fc_columns if c in df.columns], errors='ignore')
        if 'Category' in df.columns:
            df=df.drop('Category',axis=1,errors='ignore')

        df['year']=df['Date'].dt.year
        df['month']=df['Date'].dt.month
        df['week']=df['Date'].dt.isocalendar().week.astype(int)
        df['day_of_year']=df['Date'].dt.dayofyear

        df=create_features(df,response_variable,product_cols)

    if len(overall_actuals)>0:
        overall_actuals_array=np.array(overall_actuals)
        overall_predictions_array=np.array(overall_predictions)
        final_mae,final_rmse,final_mape,final_r2=compute_metrics(overall_actuals_array,overall_predictions_array)
        print(f"\n===== Overall Performance Across All {forecast_weeks} Forecasted Weeks =====")
        print(f"Overall - MAE: {final_mae:.2f}, RMSE: {final_rmse:.2f}, MAPE: {final_mape:.2f}%, R2: {final_r2:.4f}")
        mae_ci,rmse_ci,mape_ci=bootstrap_metrics(overall_actuals_array,overall_predictions_array,5000,95)
        print(f"95% Confidence Interval for MAE: [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
        print(f"95% Confidence Interval for RMSE: [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]")
        print(f"95% Confidence Interval for MAPE: [{mape_ci[0]:.2f}%, {mape_ci[1]:.2f}%]")

        final_predictions_df=pd.DataFrame({
            'Actual':overall_actuals_array,
            'Model_Prediction':overall_predictions_array
        })

        if fc_available and overall_fc_forecasts is not None:
            overall_fc_array=np.array(overall_fc_forecasts)
            final_predictions_df['Existing_Forecast']=overall_fc_array
            fc_mae,fc_rmse,fc_mape,fc_r2=compute_metrics(overall_actuals_array,overall_fc_array)
            print(f"\n===== Comparison with Existing Fc Forecast ({fc_column}) Across All Forecasted Weeks =====")
            print(f"Existing {fc_column} - MAE: {fc_mae:.2f}, RMSE: {fc_rmse:.2f}, MAPE: {fc_mape:.2f}%, R2: {fc_r2:.4f}")

        final_predictions_df=final_predictions_df.reset_index(drop=True)
        final_pred_csv_path=os.path.join(output_dir,'final_predictions_all_weeks.csv')
        final_predictions_df.to_csv(final_pred_csv_path,index=False)
        print(f"All weeks predictions saved to: {final_pred_csv_path}")
    else:
        print("No forecasts were generated. Check your data and parameters.")

    return {
        'model':{
            'mae':final_mae if len(overall_actuals)>0 else np.nan,
            'rmse':final_rmse if len(overall_actuals)>0 else np.nan,
            'mape':final_mape if len(overall_actuals)>0 else np.nan,
            'r2':final_r2 if len(overall_actuals)>0 else np.nan,
            'mae_ci':mae_ci if len(overall_actuals)>0 else (np.nan,np.nan),
            'rmse_ci':rmse_ci if len(overall_actuals)>0 else (np.nan,np.nan),
            'mape_ci':mape_ci if len(overall_actuals)>0 else (np.nan,np.nan)
        },
        'predictions':final_predictions_df if len(overall_actuals)>0 else pd.DataFrame()
    }

if __name__=='__main__':
    import argparse
    import os

    parser=argparse.ArgumentParser(description='Train and recursively forecast, ensuring only one set of lagged response variables and removing response_variable_orig.')
    parser.add_argument('--dataset',type=str,required=True,help='Path to dataset CSV file.')
    parser.add_argument('--response',type=str,required=True,help='Response variable to predict.')
    parser.add_argument('--epochs',type=int,default=50,help='Number of trials for hyperparameter optimization.')
    parser.add_argument('--output_dir',type=str,default='.',help='Directory for outputs.')
    parser.add_argument('--forecast_weeks',type=int,default=3,help='Number of weeks to forecast.')

    args=parser.parse_args()

    df=pd.read_csv(args.dataset,index_col=0)
    print(f"Columns after reading CSV: {df.columns.tolist()}")

    response_variable=args.response
    n_trials=args.epochs
    output_dir=args.output_dir
    forecast_weeks=args.forecast_weeks

    os.makedirs(output_dir,exist_ok=True)

    results=train_and_forecast(
        df=df,
        response_variable=response_variable,
        n_trials=n_trials,
        output_dir=output_dir,
        n_bootstraps=5000,
        ci_level=95,
        forecast_weeks=forecast_weeks
    )

    ci_level=95
    model_mae=results['model']['mae']
    model_rmse=results['model']['rmse']
    model_mape=results['model']['mape']
    model_r2=results['model']['r2']
    mae_ci=results['model']['mae_ci']
    rmse_ci=results['model']['rmse_ci']
    mape_ci=results['model']['mape_ci']

    print(f"\nFinal Results After {forecast_weeks} Weeks:")
    print(f"MAE: {model_mae:.2f}, RMSE: {model_rmse:.2f}, MAPE: {model_mape:.2f}%, R2: {model_r2:.4f}")
    print(f"{ci_level}% CI for MAE: [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")
    print(f"{ci_level}% CI for RMSE: [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]")
    print(f"{ci_level}% CI for MAPE: [{mape_ci[0]:.2f}%, {mape_ci[1]:.2f}%]")
