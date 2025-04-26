import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report

class MetaModel:
    def __init__(self, model_type='logistic'):
        if model_type == 'logistic':
            self.model = LogisticRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor()
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error

        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {'mean_squared_error': mse}