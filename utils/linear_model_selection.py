from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def processSubset(subset, X_train, y_train, X_test, y_test):
    
    """Given the data split into training and test sets, this will fit a linear regression model on a subset of the dataset's features

    Args:
        subset (_type_): subset to train model on
        X_train (_type_): training set of predictor features
        y_train (_type_): training set of target feature
        X_test (_type_): test set of predictor features
        y_test (_type_): test set of target feature

    Returns:
        dict: dictionary containing the model, its parameters, its coefficients, its intercept, and the test set mean squared error 
    """
    model = LinearRegression()
    model.fit(X_train[list(subset)], y_train)
    y_hat = model.predict(X_test[list(subset)])
    mse = mean_squared_error(y_test, y_hat)
    return {'model': model, 'parameters': model.feature_names_in_, 'coefficients': model.coef_, 'intercept': model.intercept_, 'MSE': mse}