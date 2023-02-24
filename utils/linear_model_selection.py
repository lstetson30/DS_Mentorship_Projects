import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from itertools import combinations

def processSubsetOLS(subset, X_train, y_train, X_test, y_test):
    
    """Given the data split into training and test sets, this will fit a linear regression model on a subset of the dataset's features

    Args:
        subset (_type_): subset to train model on or None/0
        X_train (_type_): training set of predictor features
        y_train (_type_): training set of target feature
        X_test (_type_): test set of predictor features
        y_test (_type_): test set of target feature

    Returns:
        dict: dictionary containing the model, its parameters, its coefficients, its intercept, and the test set mean squared error 
    """
    X_train_sub = X_train[list(subset)]
    n_params = len(subset)
    
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_train_sub, y_train, cv = 5, scoring = 'neg_mean_squared_error')
    cv_score_mean = cv_scores.mean()
    cv_score_std = cv_scores.std()
    
    model.fit(X_train_sub, y_train)
    y_hat = model.predict(X_test[list(subset)])
    mse = mean_squared_error(y_test, y_hat)
    
    parameters = model.feature_names_in_
    coefficients = model.coef_
    intercept = model.intercept_
        
    return {'model': model, 'is_ols': True, 'scaled_data': False, 'n_params': n_params, 'parameters': parameters, 
            'coefficients': coefficients, 'intercept': intercept, 
            'cv_score': cv_score_mean, 'cv_score_std': cv_score_std, 'test_mse': mse}

def processSubsetRidgeLasso(subset, X_train, y_train, X_test, y_test):
    """_summary_
    Reference: https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn/51629917#51629917

    Args:
        subset (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    
    X_train_sub = X_train[list(subset)]
    X_test_sub = X_test[list(subset)]
    
    alphas = list(10**np.linspace(10,-2,100)*0.5)
    
    pipe = Pipeline(steps=[('standardscaler', StandardScaler()),
                           ('estimator', Ridge())
                           ])
    
    params_grid = [{
                'standardscaler': [StandardScaler(), None],
                'estimator': [Ridge()],
                'estimator__alpha': alphas
                },
                {
                'standardscaler': [StandardScaler(), None],
                'estimator': [Lasso()],
                'estimator__alpha': alphas,
                }
              ]
    
    grid_cv = GridSearchCV(pipe, params_grid, scoring = 'neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_cv.fit(X_train_sub, y_train)
    
    model = grid_cv.best_estimator_['estimator']
    scaled_data = False if (grid_cv.best_params_['standardscaler'] == None) else True
    n_params = len(subset)
    parameters = list(subset)
    coefficients = model.coef_
    intercept = model.intercept_
    
    cv_score_mean = grid_cv.best_score_
    cv_score_std = grid_cv.cv_results_['std_test_score'][grid_cv.best_index_]
    
    y_hat = grid_cv.predict(X_test_sub)
    mse = mean_squared_error(y_test, y_hat)
    
    return {'model': model, 'is_ols': False, 'scaled_data': scaled_data, 'n_params': n_params, 'parameters': parameters, 
        'coefficients': coefficients, 'intercept': intercept, 
        'cv_score': cv_score_mean, 'cv_score_std': cv_score_std, 'test_mse': mse}
                
def getBestKModel(X_train, y_train, X_test, y_test, k: int, regressor = 'OLS'):
    
    results = []
    
    for combo in combinations(X_train.columns, k):
        if regressor == 'OLS':
            results.append(processSubsetOLS(combo, X_train, y_train, X_test, y_test))
        elif regressor == 'RidgeLasso':
            results.append(processSubsetRidgeLasso(combo, X_train, y_train, X_test, y_test))
        
    models = pd.DataFrame(results)
    
    best_model = models.iloc[[models['cv_score'].argmax()]]
    
    print("Processed", models.shape[0], "model(s) on", k, "predictor(s)")
    
    return best_model

def forwardSelection(X_train, y_train, X_test, y_test, predictors, regressor = 'OLS'):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]

    results = []
    
    for p in remaining_predictors:
        subset = list(predictors)+[p]
        if regressor == 'OLS':
            results.append(processSubsetOLS(subset, X_train, y_train, X_test, y_test))
        elif regressor == 'RidgeLasso':
            results.append(processSubsetRidgeLasso(subset, X_train, y_train, X_test, y_test))

    models = pd.DataFrame(results)

    best_model = models.iloc[[models['cv_score'].argmax()]]

    print("Processed", models.shape[0], "model(s) on", len(predictors)+1, "predictor(s).")

    return best_model

def backwardSelection(X_train, y_train, X_test, y_test, predictors, regressor = 'OLS'):

    results = []
    
    for combo in combinations(predictors, len(predictors)-1):
        if regressor == 'OLS':
            results.append(processSubsetOLS(combo, X_train, y_train, X_test, y_test))
        elif regressor == 'RidgeLasso':
            results.append(processSubsetRidgeLasso(combo, X_train, y_train, X_test, y_test))

    models = pd.DataFrame(results)

    best_model = models.iloc[[models['cv_score'].argmax()]]

    print("Processed", models.shape[0], "model(s) on", len(predictors)-1, "predictor(s).")

    return best_model