import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox
from catboost import CatBoostRegressor, Pool
import pickle
from transform_data import df_selected, param_1


def run_model(model):
    """Method for training and evaluating models.

        Arguments:
            model (model_object): A single object of model class.

        """
    if model == 'catboost':
        model, pred, feature_importance = run_cat(categorical_indices, X_train, y_train, X_test, y_test, cat_params)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    pred_inv = inv_boxcox(pred, param_1)
    test_inv = inv_boxcox(y_test, param_1)
    rmse = mean_squared_error(pred_inv, test_inv, squared=False)
    print(f'Model type: {model}')
    print(f'RMSE score: {np.round(rmse, 2)}')
    print(f'R2 score: {np.round(model.score(X_test, y_test), 2)}')
    try:
        return model, pred_inv, feature_importance
    except:
        return model, pred_inv


def run_cat(categorical_indices, xtrain, ytrain, xval, yval, params):
    """Method for training and evaluating models.

        Arguments:
            categorical_indices (list): List of category type columns indices.
            xtrain (dataframe): train part of df with features.
            ytrain (series): train part of df with target.
            xval (dataframe): test part of df with features.
            yval (series): test part of df with target.
            params (dict): params for catboost model.

        """
    feature_importance = pd.DataFrame()
    feature_importance['feature'] = xtrain.columns
    # Pool the data and specify the categorical feature indices
    print('Load Data')
    _train = Pool(xtrain, label=ytrain, cat_features=categorical_indices)
    _valid = Pool(xval, label=yval, cat_features=categorical_indices)
    print('Train CAT')
    model = CatBoostRegressor(**params)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=100,
                          plot=False)
    feature_im = fit_model.feature_importances_
    feature_importance['importance'] = feature_im
    pred = fit_model.predict(xval)
    return model, pred, feature_importance


X = df_selected.copy()
y = X.pop('Price_log')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2022)

# train linear regression
model_reg, pred_reg = run_model(LinearRegression())

# train GradientBoostingRegressor
model_gbr, pred_gbr = run_model(GradientBoostingRegressor())

# do grid search on GradientBoostingRegressor
param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [4, 5, 6],
              'n_estimators': [100, 200, 300]}  # {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 300}

grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=2).fit(X_train, y_train)

# train GradientBoostingRegressor with selected params
model_gbr_best, pred_gbr_best = run_model(grid.best_estimator_)

# prepare data for catboost
categorical_names = ["Airline", "Additional Info_1 Long layover",
                     "Additional Info_Change airports", "Additional Info_Business class",
                     "Additional Info_No check-in baggage included", "Additional Info_In-flight meal not included",
                     "Airline_group"]

feats = [f for f in df_selected.columns if f != 'Price_log']

categorical_indices = [df_selected[feats].columns.get_loc(c) for c in df_selected[feats].columns
                       if c in categorical_names]


cat_params = {'loss_function': 'RMSE',
              'eval_metric': "RMSE",
              'learning_rate': 0.4,
              'l2_leaf_reg': 2,
              'iterations': 500,
              'depth': 6,
              'random_seed': 42,
              'early_stopping_rounds': 50
             }

# train CatBoostRegressor
model_cat, pred_cat, f_i = run_model('catboost')

# blend predictions from best models
test_inv = inv_boxcox(y_test, param_1)
rmse = mean_squared_error((pred_cat + pred_gbr_best)/2, test_inv, squared=False)
print(f'Mixed RMSE score: {np.round(rmse, 2)}')

# save best models
filename_cat = 'models/catboost.sav'
pickle.dump(model_cat, open(filename_cat, 'wb'))

filename_gbr = 'models/gbr.sav'
pickle.dump(model_gbr_best, open(filename_gbr, 'wb'))

