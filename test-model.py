import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH = "best_modeel.joblib"
# fixture-1:load the trained model
@pytest.fixture
def model():
  assert os.path.exists(MODEL_PATH), "Model file not found!"
  return joblib.load(MODEL_PATH)
# fixture-2:prepare sample input data
@pytest.fixture
def sample_data():
  data = load_breast_cancer()
  X = pd.DataFrame(data.data,columns=data.feature_names)
  return X.iloc[:5]
# Test-1:check model loads successfully
def test_model_loads(model):
  assert model is not None
# Test-2:check whether model generates predictions
