import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from MetaModel import MetaModel

def test_train_and_predict_logistic():
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], size=100)

    model = MetaModel(model_type='logistic')
    model.train(X_train, y_train)

    X_test = np.random.randn(20, 5)
    preds = model.predict(X_test)
    assert len(preds) == 20

def test_train_and_predict_random_forest():
    X_train = np.random.randn(100, 5)
    y_train = np.random.choice([0, 1], size=100)

    model = MetaModel(model_type='random_forest')
    model.train(X_train, y_train)

    X_test = np.random.randn(20, 5)
    preds = model.predict(X_test)
    assert len(preds) == 20

def test_invalid_model_type():
    with pytest.raises(ValueError, match="model_type must be 'logistic' or 'random_forest'"):
        MetaModel(model_type='invalid_model')
