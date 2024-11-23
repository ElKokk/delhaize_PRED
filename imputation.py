import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from missforest.missforest import MissForest

def impute_missforest(data):
    data = np.where(data == 0, np.nan, data)
    data = pd.DataFrame(data)
    imputer = MissForest()
    imputed_data = imputer.fit_transform(data)
    return imputed_data