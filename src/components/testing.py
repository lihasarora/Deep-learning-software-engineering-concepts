import pandas as pd
import numpy  as np
import pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from src.components.data_transformation import DataTransformation


if __name__ == '__main__':
    # load the model
    with open('artifacts/model.pkl','rb') as mod:
        model = pickle.load(mod)
    
    if isinstance(model, LinearRegression):
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
    
    # load the preprocessor
    with open("artifacts/preprocessor.pkl","rb") as dfm:
        dfm = pickle.load(dfm)
    
    test_df = pd.read_csv("artifacts/test.csv")
    
    if hasattr(model, 'feature_importances_'):
        print("Feature Importances:", model.feature_importances_)
        print(model.get_params())
    
    arr = dfm.transform(test_df)
    y_test = test_df['math score']
    print(arr.shape, y_test.shape)
    pred = model.predict(arr)
    r2 = r2_score(
        y_true = y_test,
        y_pred = pred
        )
    print(f"R squared  = {r2: 0.2%}")   
    
    


