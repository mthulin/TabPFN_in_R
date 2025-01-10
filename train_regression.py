from tabpfn import TabPFNRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import sklearn

def train_regression(X, y, test_size=0.33):
    
    reg = TabPFNRegressor(device='auto')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    reg.fit(X_train, y_train)
    
    preds = reg.predict(X_test)
    RMSE = sklearn.metrics.root_mean_squared_error(y_test, preds)
    MAE =  sklearn.metrics.mean_absolute_error(y_test, preds)
    R2 =  sklearn.metrics.r2_score(y_test, preds)
    
    return {
                "predictions": preds,
                "RMSE": RMSE,
                "MAE": MAE,
                "R2": R2
           }
